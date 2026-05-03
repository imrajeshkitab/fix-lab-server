"""
Fix Lab Backend Server
======================
FastAPI server that handles audio regeneration for the Fix Lab feature,
RMS → Linear status sync, and AI-powered VO triage.

Endpoints:
  GET  /                                    — Health check
  POST /api/fix-lab/regenerate              — Start audio regeneration for selected items
  GET  /api/fix-lab/jobs/{job_id}           — Poll job progress
  POST /api/fix-lab/triage                  — Run AI triage on a bite (Phase 1)
  GET  /api/fix-lab/triage/{triage_id}      — Get stored triage result
  GET  /api/fix-lab/triage                  — List triage results (with filters)
  PATCH /api/fix-lab/triage/{triage_id}     — Admin approve/reject triage
  POST /api/linear-sync/bites              — Sync completed bites → Linear Approved
  POST /api/linear-sync/summaries          — Sync completed summaries → Linear Approved
  GET  /api/linear-sync/status/{type}      — Compare RMS vs Linear statuses
  GET  /api/linear-sync/jobs/{job_id}      — Poll sync job progress

Auth: x-fix-lab-key header must match FIX_LAB_SECRET env var.

Job state is persisted in Supabase tables (fix_lab_jobs, fix_lab_job_items)
so jobs survive server restarts and Render cold-starts.

AI Triage (Phase 1) uses a hybrid pipeline:
  1. Local Whisper STT for paragraph-level timestamp alignment
  2. Code logic to map timestamped feedback → paragraphs
  3. Gemini text-only call for decision making (no audio sent to LLM)
Results are stored in bite_audio_triage table for admin review.
"""


import os
import uuid
import asyncio
import json
import re
import logging
from typing import Optional, List, Dict, Any
from datetime import datetime, timezone

from fastapi import FastAPI, Header, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

import httpx

# ── Load env & configure paths ──────────────────────────────────────────────

load_dotenv()

from voice_config import get_voice_id

# ── Settings ────────────────────────────────────────────────────────────────

SUPABASE_URL = (os.getenv("SUPABASE_URL") or "").rstrip("/").removesuffix("/rest/v1")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
ELEVEN_LABS_API_KEY = os.getenv("ELEVEN_LABS_API_KEY")
FIX_LAB_SECRET = os.getenv("FIX_LAB_SECRET", "kitab-fix-lab-2024")
LINEAR_API_KEY = os.getenv("LINEAR_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
PORT = int(os.getenv("PORT", "8642"))

# Storage Supabase — may point to a different project than the DB.
# Falls back to the main DB creds if not explicitly set.
STORAGE_SUPABASE_URL = os.getenv("STORAGE_SUPABASE_URL") or SUPABASE_URL
STORAGE_SUPABASE_SERVICE_KEY = os.getenv("STORAGE_SUPABASE_SERVICE_KEY") or SUPABASE_SERVICE_KEY

# Supabase REST headers (service role bypasses RLS)
SB_HEADERS = {
    "apikey": SUPABASE_SERVICE_KEY,
    "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
    "Content-Type": "application/json",
}

# ── Logging ─────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("fix-lab")

# ── FastAPI App ─────────────────────────────────────────────────────────────

app = FastAPI(title="Fix Lab Server", version="2.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Shared HTTP Client (connection pooling) ─────────────────────────────────

http_client: httpx.AsyncClient = None


@app.on_event("startup")
async def startup():
    global http_client
    http_client = httpx.AsyncClient(timeout=120.0)
    logger.info("Shared HTTP client created")
    # Check for interrupted jobs on startup
    asyncio.create_task(resume_interrupted_jobs())


@app.on_event("shutdown")
async def shutdown():
    global http_client
    if http_client:
        await http_client.aclose()
        logger.info("Shared HTTP client closed")


# ── Pydantic Models ─────────────────────────────────────────────────────────

class RegenerateItem(BaseModel):
    bite_id: str
    language: str
    assignment_id: Optional[str] = None


class RegenerateRequest(BaseModel):
    items: List[RegenerateItem]  # [{bite_id, language, assignment_id}, ...]


class TriageRequest(BaseModel):
    assignment_id: str


class TriageAdminAction(BaseModel):
    admin_action: str           # approved, rejected, modified
    admin_notes: Optional[str] = None


# ── Auth Dependency ─────────────────────────────────────────────────────────

def verify_secret(x_fix_lab_key: str = Header(...)):
    if x_fix_lab_key != FIX_LAB_SECRET:
        raise HTTPException(status_code=403, detail="Invalid Fix Lab key")
    return True


# ── Helper: Supabase queries via REST API ───────────────────────────────────

async def sb_get(path: str, params: dict = None) -> Any:
    """GET request to Supabase REST API."""
    # Split embedded query params from path (e.g., "table?id=eq.x&select=y")
    if "?" in path:
        table_path, query_string = path.split("?", 1)
        from urllib.parse import parse_qs
        parsed = parse_qs(query_string, keep_blank_values=True)
        # parse_qs returns lists; flatten single values
        url_params = {k: v[0] if len(v) == 1 else v for k, v in parsed.items()}
        # Merge: explicit params override URL params
        merged_params = {**url_params, **(params or {})}
    else:
        table_path = path
        merged_params = params or {}

    r = await http_client.get(
        f"{SUPABASE_URL}/rest/v1/{table_path}",
        headers=SB_HEADERS,
        params=merged_params,
    )
    if r.status_code not in (200, 206):
        logger.error(f"Supabase GET {path} failed: {r.status_code} {r.text[:300]}")
        raise Exception(f"Supabase error: {r.status_code}")
    return r.json()


async def sb_patch(table: str, row_id: str, data: dict) -> Any:
    """PATCH (update) a row in Supabase."""
    r = await http_client.patch(
        f"{SUPABASE_URL}/rest/v1/{table}?id=eq.{row_id}",
        headers={**SB_HEADERS, "Prefer": "return=minimal"},
        json=data,
    )
    if r.status_code not in (200, 204):
        logger.error(f"Supabase PATCH {table}/{row_id} failed: {r.status_code} {r.text[:300]}")
        raise Exception(f"Supabase update error: {r.status_code}")
    return True


async def sb_insert(table: str, data: dict) -> Any:
    """INSERT a row into Supabase and return it."""
    r = await http_client.post(
        f"{SUPABASE_URL}/rest/v1/{table}",
        headers={**SB_HEADERS, "Prefer": "return=representation"},
        json=data,
    )
    if r.status_code not in (200, 201):
        logger.error(f"Supabase INSERT {table} failed: {r.status_code} {r.text[:300]}")
        raise Exception(f"Supabase insert error: {r.status_code}")
    result = r.json()
    return result[0] if isinstance(result, list) else result


async def sb_insert_many(table: str, rows: list) -> Any:
    """INSERT multiple rows into Supabase."""
    r = await http_client.post(
        f"{SUPABASE_URL}/rest/v1/{table}",
        headers={**SB_HEADERS, "Prefer": "return=representation"},
        json=rows,
    )
    if r.status_code not in (200, 201):
        logger.error(f"Supabase INSERT MANY {table} failed: {r.status_code} {r.text[:300]}")
        raise Exception(f"Supabase insert error: {r.status_code}")
    return r.json()


async def sb_patch_where(table: str, filter_query: str, data: dict) -> Any:
    """PATCH multiple rows by a filter query string (e.g. 'bite_id=eq.X&status=eq.completed')."""
    r = await http_client.patch(
        f"{SUPABASE_URL}/rest/v1/{table}?{filter_query}",
        headers={**SB_HEADERS, "Prefer": "return=minimal"},
        json=data,
    )
    if r.status_code not in (200, 204):
        logger.error(f"Supabase PATCH WHERE {table} failed: {r.status_code} {r.text[:300]}")
        raise Exception(f"Supabase update error: {r.status_code}")
    return True


async def sb_rpc(function_name: str, params: dict) -> Any:
    """Call a Supabase RPC function via REST API."""
    r = await http_client.post(
        f"{SUPABASE_URL}/rest/v1/rpc/{function_name}",
        headers=SB_HEADERS,
        json=params,
    )
    if r.status_code != 200:
        logger.error(f"RPC {function_name} failed: {r.status_code} {r.text[:300]}")
        raise Exception(f"RPC error: {r.status_code}")
    return r.json()


async def sb_upload_storage(bucket: str, path: str, data: bytes, content_type: str = "audio/mpeg") -> str:
    """Upload a file to Supabase Storage and return its public URL.

    Uses STORAGE_SUPABASE_URL / STORAGE_SUPABASE_SERVICE_KEY so the storage
    bucket can live on a different Supabase project than the DB tables.
    """
    r = await http_client.post(
        f"{STORAGE_SUPABASE_URL}/storage/v1/object/{bucket}/{path}",
        headers={
            "apikey": STORAGE_SUPABASE_SERVICE_KEY,
            "Authorization": f"Bearer {STORAGE_SUPABASE_SERVICE_KEY}",
            "Content-Type": content_type,
            "x-upsert": "true",
        },
        content=data,
    )
    if r.status_code not in (200, 201):
        logger.error(f"Storage upload failed: {r.status_code} {r.text[:300]}")
        raise Exception(f"Storage upload failed: {r.status_code}")

    # Public URL is constructed from the STORAGE project's URL (not the DB project)
    public_url = f"{STORAGE_SUPABASE_URL}/storage/v1/object/public/{bucket}/{path}"
    return public_url


# ── Helper: TTS Generation ─────────────────────────────────────────────────

def generate_audio(text: str, voice_id: str, language: str) -> bytes:
    """Generate TTS audio using the appropriate ElevenLabs model."""
    from tts_engine import generate_audio as _generate
    return _generate(text, voice_id, language)


# ── Helper: Parse audio URL to extract current round ────────────────────────

def parse_audio_round(url: str) -> int:
    """Extract the current round number from a bite audio URL."""
    if not url:
        return 1
    match = re.search(r'/round(\d+)/', url)
    if match:
        return int(match.group(1))
    match = re.search(r'/v(\d+)/', url)
    if match:
        return int(match.group(1))
    return 1


def build_new_audio_path(source_id: str, language: str, new_round: int) -> str:
    """Build the storage path for a new audio file."""
    lang_folder = "english" if language == "en" else "hindi"
    return f"bites/audio/{lang_folder}/round{new_round}/{source_id}.mp3"


# ── Job State Helpers (DB-persisted) ────────────────────────────────────────

async def create_job(total: int) -> dict:
    """Create a job in fix_lab_jobs table."""
    return await sb_insert("fix_lab_jobs", {
        "status": "running",
        "total": total,
        "completed": 0,
        "failed": 0,
    })


async def create_job_items(job_id: str, items: List[RegenerateItem]) -> list:
    """Create job items in fix_lab_job_items table."""
    rows = [
        {
            "job_id": job_id,
            "bite_id": item.bite_id,
            "language": item.language,
            "assignment_id": item.assignment_id,
            "status": "pending",
        }
        for item in items
    ]
    return await sb_insert_many("fix_lab_job_items", rows)


async def get_job(job_id: str) -> dict:
    """Get job from fix_lab_jobs table."""
    rows = await sb_get(f"fix_lab_jobs?id=eq.{job_id}&select=*")
    if not rows:
        return None
    return rows[0]


async def get_job_items(job_id: str) -> list:
    """Get all job items for a given job."""
    return await sb_get(f"fix_lab_job_items?job_id=eq.{job_id}&select=*&order=created_at.asc")


async def get_pending_items(job_id: str) -> list:
    """Get pending items for a job (for resumption)."""
    return await sb_get(
        f"fix_lab_job_items?job_id=eq.{job_id}&status=eq.pending&select=*&order=created_at.asc"
    )


async def update_job_item(item_id: str, data: dict):
    """Update a job item's status and result."""
    await sb_patch("fix_lab_job_items", item_id, data)


async def update_job(job_id: str, data: dict):
    """Update job status/counters."""
    await sb_patch("fix_lab_jobs", job_id, {**data, "updated_at": datetime.now(timezone.utc).isoformat()})


# ── Single Active Job Check ────────────────────────────────────────────────

async def get_active_job() -> Optional[dict]:
    """Check if there's an active (running) job."""
    rows = await sb_get("fix_lab_jobs?status=eq.running&select=id,status,total,completed,failed&limit=1")
    return rows[0] if rows else None


# ── Resume Interrupted Jobs ────────────────────────────────────────────────

async def resume_interrupted_jobs():
    """On startup, check for running jobs and resume them."""
    try:
        active = await get_active_job()
        if active:
            logger.info(f"Found interrupted job {active['id']}, resuming...")
            asyncio.create_task(run_regeneration_job(active["id"]))
    except Exception as e:
        logger.error(f"Error checking for interrupted jobs: {e}")


# ── Regeneration Job Worker ─────────────────────────────────────────────────

async def run_regeneration_job(job_id: str):
    """Process pending items one at a time with DB checkpointing."""
    logger.info(f"Job {job_id}: Starting worker")

    try:
        pending_items = await get_pending_items(job_id)
        logger.info(f"Job {job_id}: {len(pending_items)} pending items to process")

        for job_item in pending_items:
            bite_id = job_item["bite_id"]
            lang = job_item["language"]
            assignment_id = job_item.get("assignment_id")
            item_id = job_item["id"]

            # Mark item as processing
            await update_job_item(item_id, {"status": "processing"})

            try:
                # 1. Fetch bite data (one at a time — memory safe)
                bites = await sb_get(f"bites?id=eq.{bite_id}&select=*")
                if not bites:
                    await update_job_item(item_id, {
                        "status": "skipped",
                        "error": "Bite not found",
                    })
                    await update_job(job_id, {"failed": (await get_job(job_id))["failed"] + 1})
                    continue

                bite = bites[0]
                source_id = bite.get("source_id", bite_id)
                audio_data = bite.get("audio", {}) or {}
                content_data = bite.get("content", {}) or {}
                audio_version = bite.get("audio_version", {}) or {"en": 1, "hi": 1}
                title = bite.get("title", "Unknown")

                # 2. Get the text content
                lang_content = content_data.get(lang, {})
                if isinstance(lang_content, dict):
                    text = lang_content.get("text", "") or lang_content.get("body", "")
                else:
                    text = str(lang_content) if lang_content else ""

                if not text:
                    await update_job_item(item_id, {
                        "status": "skipped",
                        "error": f"No {lang} content text found",
                        "result": json.dumps({"title": title}),
                    })
                    await update_job(job_id, {"failed": (await get_job(job_id))["failed"] + 1})
                    logger.warning(f"Job {job_id}: Skipped {bite_id}/{lang} — no text")
                    continue

                # 3. Get the voice ID
                lang_audio = audio_data.get(lang, {}) or {}
                vo_artist = lang_audio.get("vo_artist", "")
                voice_id = get_voice_id(vo_artist, lang)

                # 4. Determine the new round number
                current_url = lang_audio.get("url", "")
                current_round = parse_audio_round(current_url)
                new_round = current_round + 1
                current_version = audio_version.get(lang, 1)
                new_version = current_version + 1

                logger.info(f"Job {job_id}: Generating {bite_id}/{lang} "
                           f"(voice={vo_artist}, round {current_round}→{new_round})")

                # 5. Generate TTS audio (blocking — run in thread pool)
                audio_bytes = await asyncio.to_thread(
                    generate_audio, text, voice_id, lang
                )

                logger.info(f"Job {job_id}: Generated {len(audio_bytes)} bytes for {bite_id}/{lang}")

                # 5b. Calculate audio duration using mutagen
                duration_str = None
                try:
                    from io import BytesIO
                    from mutagen.mp3 import MP3
                    mp3 = MP3(BytesIO(audio_bytes))
                    total_seconds = mp3.info.length
                    minutes = int(total_seconds // 60)
                    seconds = int(total_seconds % 60)
                    duration_str = f"{minutes:02d}:{seconds:02d}"
                    logger.info(f"Job {job_id}: Audio duration: {duration_str}")
                except Exception as e:
                    logger.warning(f"Job {job_id}: Could not calculate duration: {e}")

                # 6. Upload to Supabase Storage
                storage_path = build_new_audio_path(source_id, lang, new_round)
                new_url = await sb_upload_storage(
                    bucket="RMS-content",
                    path=storage_path,
                    data=audio_bytes,
                )

                # Free memory immediately
                audio_size = len(audio_bytes)
                del audio_bytes

                # 7. Update the bites table
                updated_audio = dict(audio_data)
                lang_audio_updated = dict(lang_audio)
                lang_audio_updated["url"] = new_url
                if duration_str is not None:
                    lang_audio_updated["duration"] = duration_str
                updated_audio[lang] = lang_audio_updated

                updated_version = dict(audio_version)
                updated_version[lang] = new_version

                await sb_patch("bites", bite_id, {
                    "audio": updated_audio,
                    "audio_version": updated_version,
                })

                # 8. Auto-mark assignment as 'fixed'
                if assignment_id:
                    try:
                        await sb_patch("content_assignments", assignment_id, {
                            "status": "fixed",
                        })
                        logger.info(f"Job {job_id}: Marked assignment {assignment_id} as 'fixed'")
                    except Exception as e:
                        logger.warning(f"Job {job_id}: Could not mark assignment as fixed: {e}")

                # 9. Update job item as completed
                result_data = {
                    "title": title,
                    "new_url": new_url,
                    "new_round": new_round,
                    "new_version": new_version,
                    "audio_size": audio_size,
                    "duration": duration_str,
                    "char_count": len(text),
                }
                await update_job_item(item_id, {
                    "status": "completed",
                    "result": json.dumps(result_data),
                })

                # Update job counters
                job = await get_job(job_id)
                await update_job(job_id, {"completed": job["completed"] + 1})

                logger.info(f"Job {job_id}: ✅ {bite_id}/{lang} → round{new_round}, v{new_version}, {duration_str}")

            except Exception as e:
                await update_job_item(item_id, {
                    "status": "failed",
                    "error": str(e)[:500],
                })
                job = await get_job(job_id)
                await update_job(job_id, {"failed": job["failed"] + 1})
                logger.error(f"Job {job_id}: ❌ {bite_id}/{lang} — {e}")

            # Breathe on 0.1 CPU — longer delay between items
            await asyncio.sleep(2)

    except Exception as e:
        logger.error(f"Job {job_id}: Fatal error — {e}")

    # Mark job as complete
    job = await get_job(job_id)
    if job:
        if job["failed"] == job["total"]:
            final_status = "failed"
        elif job["failed"] > 0:
            final_status = "completed_with_errors"
        else:
            final_status = "completed"
        await update_job(job_id, {"status": final_status})
        logger.info(f"Job {job_id}: Done — {job['completed']}/{job['total']} succeeded, "
                   f"{job['failed']} failed")


# ═══════════════════════════════════════════════════════════════════════════
# API ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════

@app.get("/")
async def health():
    triage_ready = bool(GEMINI_API_KEY)
    return {
        "status": "ok",
        "service": "fix-lab-server",
        "version": "2.1.0",
        "triage_enabled": triage_ready,
    }


@app.post("/api/fix-lab/regenerate")
async def start_regeneration(
    request: RegenerateRequest,
    background_tasks: BackgroundTasks,
    x_fix_lab_key: str = Header(...),
):
    """Start a regeneration job for selected bites."""
    verify_secret(x_fix_lab_key)

    if not request.items:
        raise HTTPException(status_code=400, detail="No items provided")

    # Single active job enforcement
    active = await get_active_job()
    if active:
        raise HTTPException(
            status_code=409,
            detail=f"A job is already running (id: {active['id']}, {active['completed']}/{active['total']} done). Please wait."
        )

    total_items = len(request.items)

    # Create job in DB
    job = await create_job(total_items)
    job_id = job["id"]

    # Create job items in DB
    await create_job_items(job_id, request.items)

    # Run in background
    background_tasks.add_task(run_regeneration_job, job_id)

    logger.info(f"Job {job_id}: Created with {total_items} items")
    return {"job_id": job_id, "status": "running", "total": total_items}


@app.get("/api/fix-lab/jobs/{job_id}")
async def get_job_status(job_id: str, x_fix_lab_key: str = Header(...)):
    """Poll job progress. Returns job info + all item results."""
    verify_secret(x_fix_lab_key)

    # Validate UUID format
    try:
        uuid.UUID(job_id)
    except ValueError:
        raise HTTPException(status_code=404, detail="Invalid job ID format")

    job = await get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    # Get items for this job
    items = await get_job_items(job_id)

    # Build results list matching the frontend's expected format
    results = []
    for item in items:
        result_data = json.loads(item.get("result") or "{}") if item.get("result") else {}
        results.append({
            "bite_id": item["bite_id"],
            "language": item["language"],
            "status": item["status"],
            "title": result_data.get("title", ""),
            "error": item.get("error"),
            "new_url": result_data.get("new_url"),
            "new_round": result_data.get("new_round"),
            "new_version": result_data.get("new_version"),
            "audio_size": result_data.get("audio_size", 0),
            "duration": result_data.get("duration"),
        })

    return {
        "job_id": job["id"],
        "status": job["status"],
        "total": job["total"],
        "completed": job["completed"],
        "failed": job["failed"],
        "results": results,
    }


# ═══════════════════════════════════════════════════════════════════════════
# LINEAR SYNC — RMS content_assignments → Linear issue status
# ═══════════════════════════════════════════════════════════════════════════

# ── Linear Constants ────────────────────────────────────────────────────────

LINEAR_API_URL = "https://api.linear.app/graphql"

# Approved status IDs per team
LINEAR_APPROVED_STATUS = {
    "BYT":   "f7540a43-23cd-47db-92c2-d345b804b325",
    "BYTHN": "5f7b0e54-9c37-49c8-a8cf-ef4a6cded613",
    "SUM":   "c392b2c9-9487-463b-9c0d-2a86049ae3e8",
    "SUMHN": "66e0fa4f-14a4-42f5-89fc-67c5a77e158c",
}

# Statuses to skip (already final)
LINEAR_SKIP_STATUSES = {"Approved", "Published", "Canceled", "Rejected", "Duplicate"}

# Max concurrent Linear API calls
LINEAR_SEMAPHORE_LIMIT = 20

# ── Linear Sync: In-memory job store ────────────────────────────────────────
# Lightweight — no DB table needed, jobs are short-lived

sync_jobs: Dict[str, Dict[str, Any]] = {}


# ── Linear API Helpers ──────────────────────────────────────────────────────

async def linear_graphql(query: str, variables: dict = None) -> Any:
    """Execute a GraphQL query against the Linear API."""
    r = await http_client.post(
        LINEAR_API_URL,
        headers={
            "Authorization": LINEAR_API_KEY,
            "Content-Type": "application/json",
        },
        json={"query": query, "variables": variables or {}},
    )
    if r.status_code != 200:
        raise Exception(f"Linear API error: {r.status_code} {r.text[:300]}")
    data = r.json()
    if "errors" in data:
        raise Exception(f"Linear GraphQL error: {data['errors']}")
    return data.get("data")


async def linear_get_issue(identifier: str) -> Optional[dict]:
    """Fetch a Linear issue by its identifier (e.g. BYT-123)."""
    query = """
    query($filter: IssueFilter) {
        issues(filter: $filter, first: 1) {
            nodes {
                id
                identifier
                state { name }
            }
        }
    }
    """
    variables = {
        "filter": {
            "number": {"eq": int(identifier.split("-")[1])},
            "team": {"key": {"eq": identifier.split("-")[0]}},
        }
    }
    data = await linear_graphql(query, variables)
    nodes = data.get("issues", {}).get("nodes", [])
    return nodes[0] if nodes else None


async def linear_update_issue_status(issue_id: str, status_id: str) -> bool:
    """Update a Linear issue's status."""
    query = """
    mutation($id: String!, $stateId: String!) {
        issueUpdate(id: $id, input: { stateId: $stateId }) {
            success
        }
    }
    """
    data = await linear_graphql(query, {"id": issue_id, "stateId": status_id})
    return data.get("issueUpdate", {}).get("success", False)


# ── Linear Sync: Build identifier for target team ──────────────────────────

def build_linear_identifier(base_identifier: str, language: str, content_type: str) -> Optional[str]:
    """
    Build the target Linear identifier based on language.
    e.g. BYT-123 + hi → BYTHN-123, SUM-456 + en → SUM-456
    """
    if not base_identifier or "-" not in base_identifier:
        return None

    parts = base_identifier.split("-", 1)
    prefix = parts[0]  # BYT or SUM
    number = parts[1]

    if language == "hi":
        # Map to Hindi team
        if prefix == "BYT":
            return f"BYTHN-{number}"
        elif prefix == "SUM":
            return f"SUMHN-{number}"
        else:
            return None
    elif language == "en":
        return base_identifier  # Already the right team
    else:
        return None


def get_team_key_from_identifier(identifier: str) -> Optional[str]:
    """Extract team key from Linear identifier, e.g. BYTHN-123 → BYTHN"""
    if not identifier or "-" not in identifier:
        return None
    return identifier.split("-")[0]


# ── Linear Sync: Worker ────────────────────────────────────────────────────

async def process_single_sync_item(
    sem: asyncio.Semaphore,
    job_id: str,
    assignment: dict,
    content_row: dict,
    language: str,
    content_type: str,
    dry_run: bool,
) -> dict:
    """Process a single assignment→Linear sync. Returns a result dict."""
    async with sem:
        base_identifier = content_row.get("linear_identifier")
        target_identifier = build_linear_identifier(base_identifier, language, content_type)

        result = {
            "assignment_id": assignment["id"],
            "content_id": assignment["content_id"],
            "base_identifier": base_identifier,
            "target_identifier": target_identifier,
            "language": language,
            "status": "pending",
            "action": None,
            "error": None,
        }

        if not target_identifier:
            result["status"] = "skipped"
            result["action"] = "no_identifier"
            return result

        try:
            # Fetch current Linear issue status
            issue = await linear_get_issue(target_identifier)
            if not issue:
                result["status"] = "skipped"
                result["action"] = "issue_not_found"
                return result

            current_status = issue.get("state", {}).get("name", "")
            result["current_status"] = current_status

            if current_status in LINEAR_SKIP_STATUSES:
                result["status"] = "skipped"
                result["action"] = f"already_{current_status.lower()}"
                return result

            team_key = get_team_key_from_identifier(target_identifier)
            approved_status_id = LINEAR_APPROVED_STATUS.get(team_key)

            if not approved_status_id:
                result["status"] = "error"
                result["error"] = f"No approved status ID for team {team_key}"
                return result

            if dry_run:
                result["status"] = "would_update"
                result["action"] = f"{current_status} → Approved"
                return result

            # Actually update
            success = await linear_update_issue_status(issue["id"], approved_status_id)
            if success:
                result["status"] = "updated"
                result["action"] = f"{current_status} → Approved"
            else:
                result["status"] = "error"
                result["error"] = "Update returned success=false"

        except Exception as e:
            result["status"] = "error"
            result["error"] = str(e)[:300]

        return result


async def run_linear_sync_job(job_id: str, content_type: str, dry_run: bool):
    """Background worker: sync completed assignments → Linear Approved."""
    job = sync_jobs[job_id]
    job["status"] = "running"
    logger.info(f"Sync job {job_id}: Starting ({content_type}, dry_run={dry_run})")

    try:
        # 1. Fetch completed assignments for this content type
        assignments = await sb_get(
            f"content_assignments?content_type=eq.{content_type}&status=eq.completed&select=id,content_id,assigned_languages"
        )
        logger.info(f"Sync job {job_id}: Found {len(assignments)} completed {content_type} assignments")

        if not assignments:
            job["status"] = "completed"
            job["message"] = "No completed assignments found"
            return

        # 2. Collect all content_ids and fetch content rows in bulk
        content_ids = [a["content_id"] for a in assignments if a.get("content_id")]
        table = "bites" if content_type == "bites" else "summaries"

        # Fetch in batches (Supabase URL length limits)
        content_map = {}
        batch_size = 50
        for i in range(0, len(content_ids), batch_size):
            batch_ids = content_ids[i:i + batch_size]
            ids_filter = ",".join(batch_ids)
            rows = await sb_get(
                f"{table}?id=in.({ids_filter})&select=id,linear_identifier,source_id"
            )
            for row in rows:
                content_map[row["id"]] = row

        logger.info(f"Sync job {job_id}: Fetched {len(content_map)} {table} rows")

        # 3. Build list of sync tasks
        sem = asyncio.Semaphore(LINEAR_SEMAPHORE_LIMIT)
        tasks = []

        for assignment in assignments:
            content_id = assignment.get("content_id")
            content_row = content_map.get(content_id)
            if not content_row:
                job["results"].append({
                    "assignment_id": assignment["id"],
                    "content_id": content_id,
                    "status": "skipped",
                    "action": "content_not_found",
                })
                job["skipped"] += 1
                continue

            # Get language from assigned_languages
            langs = assignment.get("assigned_languages", [])
            if isinstance(langs, str):
                import json as _json
                langs = _json.loads(langs)
            language = langs[0] if langs else None

            if not language:
                job["results"].append({
                    "assignment_id": assignment["id"],
                    "content_id": content_id,
                    "status": "skipped",
                    "action": "no_language",
                })
                job["skipped"] += 1
                continue

            tasks.append(
                process_single_sync_item(
                    sem, job_id, assignment, content_row, language, content_type, dry_run
                )
            )

        job["total"] = len(tasks) + job["skipped"]

        # 4. Run all tasks in parallel
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for r in results:
                if isinstance(r, Exception):
                    job["errors"] += 1
                    job["results"].append({
                        "status": "error",
                        "error": str(r)[:300],
                    })
                else:
                    job["results"].append(r)
                    if r["status"] == "updated":
                        job["updated"] += 1
                    elif r["status"] == "would_update":
                        job["would_update"] += 1
                    elif r["status"] == "skipped":
                        job["skipped"] += 1
                    elif r["status"] == "error":
                        job["errors"] += 1

                # Update progress for polling
                job["processed"] += 1

        job["status"] = "completed"
        logger.info(
            f"Sync job {job_id}: Done — "
            f"updated={job['updated']}, would_update={job['would_update']}, "
            f"skipped={job['skipped']}, errors={job['errors']}"
        )

    except Exception as e:
        job["status"] = "failed"
        job["error"] = str(e)[:500]
        logger.error(f"Sync job {job_id}: Fatal error — {e}")


# ── Linear Sync: API Endpoints ─────────────────────────────────────────────

@app.post("/api/linear-sync/bites")
async def sync_bites(
    background_tasks: BackgroundTasks,
    dry_run: bool = True,
    x_fix_lab_key: str = Header(...),
):
    """Sync completed bites assignments → Linear 'Approved' status."""
    verify_secret(x_fix_lab_key)

    job_id = str(uuid.uuid4())
    sync_jobs[job_id] = {
        "id": job_id,
        "content_type": "bites",
        "dry_run": dry_run,
        "status": "starting",
        "total": 0,
        "processed": 0,
        "updated": 0,
        "would_update": 0,
        "skipped": 0,
        "errors": 0,
        "error": None,
        "message": None,
        "results": [],
    }

    background_tasks.add_task(run_linear_sync_job, job_id, "bites", dry_run)
    logger.info(f"Sync job {job_id}: Created for bites (dry_run={dry_run})")

    return {"job_id": job_id, "status": "starting", "dry_run": dry_run, "content_type": "bites"}


@app.post("/api/linear-sync/summaries")
async def sync_summaries(
    background_tasks: BackgroundTasks,
    dry_run: bool = True,
    x_fix_lab_key: str = Header(...),
):
    """Sync completed summaries assignments → Linear 'Approved' status."""
    verify_secret(x_fix_lab_key)

    job_id = str(uuid.uuid4())
    sync_jobs[job_id] = {
        "id": job_id,
        "content_type": "summaries",
        "dry_run": dry_run,
        "status": "starting",
        "total": 0,
        "processed": 0,
        "updated": 0,
        "would_update": 0,
        "skipped": 0,
        "errors": 0,
        "error": None,
        "message": None,
        "results": [],
    }

    background_tasks.add_task(run_linear_sync_job, job_id, "summaries", dry_run)
    logger.info(f"Sync job {job_id}: Created for summaries (dry_run={dry_run})")

    return {"job_id": job_id, "status": "starting", "dry_run": dry_run, "content_type": "summaries"}


@app.get("/api/linear-sync/jobs/{job_id}")
async def get_sync_job_status(job_id: str, x_fix_lab_key: str = Header(...)):
    """Poll sync job progress."""
    verify_secret(x_fix_lab_key)

    job = sync_jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Sync job not found")

    response = {
        "job_id": job["id"],
        "content_type": job["content_type"],
        "dry_run": job["dry_run"],
        "status": job["status"],
        "total": job["total"],
        "processed": job["processed"],
        "updated": job["updated"],
        "would_update": job["would_update"],
        "skipped": job["skipped"],
        "errors": job["errors"],
        "error": job.get("error"),
        "message": job.get("message"),
        "results": job["results"],
    }

    # Include summary for status check jobs
    if "summary" in job and job["summary"]:
        response["summary"] = job["summary"]

    return response


@app.get("/api/linear-sync/status/{content_type}")
async def get_sync_status(
    content_type: str,
    background_tasks: BackgroundTasks,
    x_fix_lab_key: str = Header(...),
):
    """
    Compare RMS completed assignments vs Linear statuses.
    Returns a job_id — poll /api/linear-sync/jobs/{job_id} for results.
    content_type must be 'bites' or 'summaries'.
    """
    verify_secret(x_fix_lab_key)

    if content_type not in ("bites", "summaries"):
        raise HTTPException(status_code=400, detail="content_type must be 'bites' or 'summaries'")

    job_id = str(uuid.uuid4())
    sync_jobs[job_id] = {
        "id": job_id,
        "content_type": content_type,
        "dry_run": True,
        "status": "starting",
        "total": 0,
        "processed": 0,
        "updated": 0,
        "would_update": 0,
        "skipped": 0,
        "errors": 0,
        "error": None,
        "message": "status_check",
        "results": [],
        # Extra fields for status check
        "summary": {},
    }

    background_tasks.add_task(run_status_check_job, job_id, content_type)
    logger.info(f"Status check job {job_id}: Created for {content_type}")

    return {"job_id": job_id, "status": "starting", "content_type": content_type}


async def run_status_check_job(job_id: str, content_type: str):
    """Background worker: compare RMS completed vs Linear statuses."""
    job = sync_jobs[job_id]
    job["status"] = "running"
    logger.info(f"Status check {job_id}: Starting ({content_type})")

    try:
        # 1. Fetch completed assignments
        assignments = await sb_get(
            f"content_assignments?content_type=eq.{content_type}&status=eq.completed&select=id,content_id,assigned_languages"
        )

        if not assignments:
            job["status"] = "completed"
            job["summary"] = {"completed_in_rms": 0, "synced": 0, "not_synced": 0, "errors": 0}
            job["message"] = "No completed assignments found"
            return

        # 2. Fetch content rows in bulk
        content_ids = [a["content_id"] for a in assignments if a.get("content_id")]
        table = "bites" if content_type == "bites" else "summaries"

        content_map = {}
        batch_size = 50
        for i in range(0, len(content_ids), batch_size):
            batch_ids = content_ids[i:i + batch_size]
            ids_filter = ",".join(batch_ids)
            rows = await sb_get(
                f"{table}?id=in.({ids_filter})&select=id,linear_identifier,source_id,title"
            )
            for row in rows:
                content_map[row["id"]] = row

        # 3. Check each assignment's Linear status in parallel
        sem = asyncio.Semaphore(LINEAR_SEMAPHORE_LIMIT)
        synced = []
        not_synced = []
        check_errors = []
        no_identifier = []

        job["total"] = len(assignments)

        async def check_one(assignment):
            async with sem:
                content_id = assignment.get("content_id")
                content_row = content_map.get(content_id)

                if not content_row:
                    job["processed"] += 1
                    return {"type": "error", "data": {
                        "assignment_id": assignment["id"],
                        "content_id": content_id,
                        "reason": "content_not_found",
                    }}

                langs = assignment.get("assigned_languages", [])
                if isinstance(langs, str):
                    import json as _json
                    langs = _json.loads(langs)
                language = langs[0] if langs else None

                base_id = content_row.get("linear_identifier")
                target_id = build_linear_identifier(base_id, language, content_type) if base_id and language else None

                if not target_id:
                    job["processed"] += 1
                    return {"type": "no_identifier", "data": {
                        "assignment_id": assignment["id"],
                        "content_id": content_id,
                        "title": content_row.get("title", ""),
                        "language": language,
                        "base_identifier": base_id,
                    }}

                try:
                    issue = await linear_get_issue(target_id)
                    job["processed"] += 1

                    if not issue:
                        return {"type": "not_synced", "data": {
                            "assignment_id": assignment["id"],
                            "content_id": content_id,
                            "title": content_row.get("title", ""),
                            "linear_identifier": target_id,
                            "language": language,
                            "linear_status": "issue_not_found",
                        }}

                    status_name = issue.get("state", {}).get("name", "Unknown")

                    item = {
                        "assignment_id": assignment["id"],
                        "content_id": content_id,
                        "title": content_row.get("title", ""),
                        "linear_identifier": target_id,
                        "language": language,
                        "linear_status": status_name,
                    }

                    if status_name in ("Approved", "Published"):
                        return {"type": "synced", "data": item}
                    else:
                        return {"type": "not_synced", "data": item}

                except Exception as e:
                    job["processed"] += 1
                    return {"type": "error", "data": {
                        "assignment_id": assignment["id"],
                        "content_id": content_id,
                        "linear_identifier": target_id,
                        "error": str(e)[:300],
                    }}

        results = await asyncio.gather(*[check_one(a) for a in assignments], return_exceptions=True)

        for r in results:
            if isinstance(r, Exception):
                check_errors.append({"error": str(r)[:300]})
            elif r["type"] == "synced":
                synced.append(r["data"])
            elif r["type"] == "not_synced":
                not_synced.append(r["data"])
            elif r["type"] == "no_identifier":
                no_identifier.append(r["data"])
            elif r["type"] == "error":
                check_errors.append(r["data"])

        job["summary"] = {
            "completed_in_rms": len(assignments),
            "synced": len(synced),
            "not_synced": len(not_synced),
            "no_identifier": len(no_identifier),
            "errors": len(check_errors),
        }
        job["results"] = {
            "synced": synced,
            "not_synced": not_synced,
            "no_identifier": no_identifier,
            "errors": check_errors,
        }
        job["status"] = "completed"

        logger.info(
            f"Status check {job_id}: Done — "
            f"RMS completed={len(assignments)}, synced={len(synced)}, "
            f"not_synced={len(not_synced)}, no_id={len(no_identifier)}, errors={len(check_errors)}"
        )

    except Exception as e:
        job["status"] = "failed"
        job["error"] = str(e)[:500]
        logger.error(f"Status check {job_id}: Fatal error — {e}")


# ═══════════════════════════════════════════════════════════════════════════
# VO TRIAGE — AI-powered audio issue analysis (Phase 1)
# ═══════════════════════════════════════════════════════════════════════════
#
# Phase 1 is READ-ONLY: analyzes audio + feedback, returns a fix plan.
# No audio is regenerated — the admin reviews the plan in FixLab UI.
#
# Endpoints:
#   POST /api/fix-lab/triage             — Run triage on a bite
#   GET  /api/fix-lab/triage/{triage_id} — Get stored triage result
#   GET  /api/fix-lab/triage             — List triage results (with filters)
#   PATCH /api/fix-lab/triage/{triage_id} — Admin approve/reject/modify
# ═══════════════════════════════════════════════════════════════════════════


@app.post("/api/fix-lab/triage")
async def triage_bite(
    request: TriageRequest,
    x_fix_lab_key: str = Header(...),
):
    """
    Run AI triage on a bite's voice-over audio.

    Only input needed: assignment_id from content_assignments.
    The endpoint validates the assignment (must be content_type=bites,
    status=changes_requested) and derives bite_id + language automatically.

    Pipeline:
      1. Validate assignment
      2. Fetch bite data + download audio + fetch feedback
      3. Local Whisper STT → paragraph-level timestamps
      4. Code logic → map timestamped feedback to paragraphs
      5. Gemini text-only → decision (only affected paras + full feedback thread)
      6. Persist + return

    Requires GEMINI_API_KEY or GOOGLE_API_KEY in environment.
    """
    verify_secret(x_fix_lab_key)

    from triage import split_into_paragraphs, run_triage_decision
    from stt import transcribe_audio, align_paragraphs, map_feedback_to_paragraphs

    assignment_id = request.assignment_id

    # ── 1. Fetch and validate the assignment ───────────────────────────────
    logger.info(f"DEBUG: SUPABASE_URL = {SUPABASE_URL}")
    query_path = (
        f"content_assignments?id=eq.{assignment_id}"
        f"&select=id,content_id,content_type,status,assigned_languages"
    )
    logger.info(f"DEBUG: Full URL = {SUPABASE_URL}/rest/v1/{query_path}")
    assignments = await sb_get(query_path)
    logger.info(f"DEBUG: assignments response = {assignments}")
    if not assignments:
        raise HTTPException(status_code=404, detail="Assignment not found")
    assignment = assignments[0]
    logger.info(f"DEBUG: assignment[0] = {assignment}")

    # Validate content_type
    if assignment.get("content_type") != "bites":
        raise HTTPException(
            status_code=400,
            detail=f"Triage is only supported for bites, "
                   f"but this assignment is content_type='{assignment.get('content_type')}'"
        )

    # Validate status
    if assignment.get("status") != "changes_requested":
        raise HTTPException(
            status_code=400,
            detail=f"Triage requires status='changes_requested', "
                   f"but this assignment has status='{assignment.get('status')}'"
        )

    # Derive bite_id and language from the assignment
    bite_id = assignment["content_id"]
    assigned_langs = assignment.get("assigned_languages", [])
    if isinstance(assigned_langs, str):
        assigned_langs = json.loads(assigned_langs)
    if not assigned_langs:
        raise HTTPException(
            status_code=400,
            detail="Assignment has no assigned_languages"
        )
    lang = assigned_langs[0]

    logger.info(f"Triage: assignment={assignment_id}, bite={bite_id}, lang={lang}")

    # ── 2. Fetch bite data ─────────────────────────────────────────────────
    bites = await sb_get(
        f"bites?id=eq.{bite_id}&select=id,source_id,title,content,audio,audio_version"
    )
    if not bites:
        raise HTTPException(status_code=404, detail="Bite not found")
    bite = bites[0]

    # ── 3. Extract text content ────────────────────────────────────────────
    content_data = bite.get("content", {}) or {}
    lang_content = content_data.get(lang, {})
    if isinstance(lang_content, dict):
        text = lang_content.get("text", "") or lang_content.get("body", "")
    else:
        text = str(lang_content) if lang_content else ""

    if not text:
        raise HTTPException(
            status_code=400,
            detail=f"No {lang} content text found for this bite"
        )

    paragraphs = split_into_paragraphs(text)
    logger.info(f"Triage {bite_id}/{lang}: {len(paragraphs)} paragraphs, "
                f"{len(text)} chars")

    # ── 4. Download audio ──────────────────────────────────────────────────
    audio_data = bite.get("audio", {}) or {}
    lang_audio = audio_data.get(lang, {}) or {}
    audio_url = lang_audio.get("url")

    if not audio_url:
        raise HTTPException(
            status_code=400,
            detail=f"No {lang} audio URL found for this bite"
        )

    logger.info(f"Triage {bite_id}/{lang}: Downloading audio...")
    audio_resp = await http_client.get(audio_url)
    if audio_resp.status_code != 200:
        raise HTTPException(
            status_code=502,
            detail=f"Failed to download audio: HTTP {audio_resp.status_code}"
        )
    audio_bytes = audio_resp.content
    logger.info(f"Triage {bite_id}/{lang}: Audio downloaded, {len(audio_bytes)} bytes")

    # ── 5. Fetch reviewer feedback ─────────────────────────────────────────
    feedback_items = []

    # Get latest reviews for this assignment (most recent first)
    reviews = await sb_get(
        f"reviews?assignment_id=eq.{assignment_id}"
        f"&select=id,rating,feedback_details,created_at"
        f"&order=created_at.desc&limit=5"
    )

    # Extract structured feedback items, filtered by language
    for review in reviews:
        details = review.get("feedback_details")
        if isinstance(details, str):
            details = json.loads(details)
        if details and isinstance(details, list):
            for item in details:
                item_lang = item.get("language")
                # Include if language matches OR if no language tag (legacy)
                if item_lang == lang or not item_lang:
                    feedback_items.append(item)

    logger.info(f"Triage {bite_id}/{lang}: {len(feedback_items)} feedback items")

    if not feedback_items:
        logger.warning(
            f"Triage {bite_id}/{lang}: No feedback items found. "
            "Model will analyze based on general context only."
        )

    # ── 6. Whisper STT → paragraph alignment ───────────────────────────────
    logger.info(f"Triage {bite_id}/{lang}: Running Whisper STT...")
    try:
        segments = transcribe_audio(audio_bytes, lang)
        paragraph_timings = align_paragraphs(segments, paragraphs)
    except Exception as e:
        logger.error(f"Triage {bite_id}/{lang}: STT failed: {e}")
        # If STT fails, we can still run triage without paragraph timings
        # All feedback becomes "unmapped" and goes to Gemini as-is
        paragraph_timings = []
    finally:
        del audio_bytes  # Free memory

    # ── 7. Map feedback to paragraphs (code logic, no LLM) ─────────────────
    mapped_feedback, unmapped_feedback, affected_indices = map_feedback_to_paragraphs(
        feedback_items, paragraph_timings
    )

    logger.info(
        f"Triage {bite_id}/{lang}: {len(mapped_feedback)} mapped, "
        f"{len(unmapped_feedback)} unmapped, "
        f"{len(affected_indices)} affected paragraphs"
    )

    # ── 8. Gemini text-only decision ───────────────────────────────────────
    try:
        result, usage = await run_triage_decision(
            paragraphs=paragraphs,
            affected_indices=affected_indices,
            mapped_feedback=mapped_feedback,
            unmapped_feedback=unmapped_feedback,
            language=lang,
            title=bite.get("title", ""),
        )
    except Exception as e:
        logger.error(f"Triage failed for {bite_id}/{lang}: {e}")

        # Store failure for audit trail
        try:
            await sb_insert("bite_audio_triage", {
                "bite_id": bite_id,
                "language": lang,
                "assignment_id": assignment_id,
                "decision": "escalate",
                "reasoning": f"Triage failed: {str(e)[:500]}",
                "status": "failed",
            })
        except Exception:
            pass  # Don't fail the request if audit insert fails

        raise HTTPException(
            status_code=500,
            detail=f"Triage failed: {str(e)[:200]}"
        )

    # ── 9. Persist result ──────────────────────────────────────────────────
    triage_record = await sb_insert("bite_audio_triage", {
        "bite_id": bite_id,
        "language": lang,
        "assignment_id": assignment_id,
        "decision": result.get("decision", "escalate"),
        "confidence": result.get("confidence"),
        "segments_to_regen": result.get("segments_to_regen", []),
        "feedback_classification": result.get("feedback_classification", []),
        "paragraph_timings": paragraph_timings,
        "reasoning": result.get("reasoning", ""),
        "model_used": usage.get("model", ""),
        "cost_input_tokens": usage.get("input_tokens", 0),
        "cost_output_tokens": usage.get("output_tokens", 0),
        "status": "completed",
    })

    # ── 9b. Expire previous triage runs for the same (bite, language) ──────
    # Mark older rows as expired so admin queue only shows the latest.
    # Only expire NON-expired rows other than the one we just inserted.
    try:
        await sb_patch_where(
            "bite_audio_triage",
            f"bite_id=eq.{bite_id}"
            f"&language=eq.{lang}"
            f"&id=neq.{triage_record['id']}"
            f"&status=neq.expired",
            {"status": "expired"},
        )
    except Exception as e:
        # Don't fail the request — the new row is already saved.
        logger.warning(f"Triage {bite_id}/{lang}: Could not expire old rows: {e}")

    # ── 10. Return ─────────────────────────────────────────────────────────
    logger.info(
        f"Triage {bite_id}/{lang}: Done — decision={result.get('decision')}, "
        f"triage_id={triage_record['id']}"
    )

    return {
        "triage_id": triage_record["id"],
        "assignment_id": assignment_id,
        "bite_id": bite_id,
        "language": lang,
        "title": bite.get("title", ""),
        "decision": result.get("decision"),
        "confidence": result.get("confidence"),
        "reasoning": result.get("reasoning"),
        "segments_to_regen": result.get("segments_to_regen", []),
        "feedback_classification": result.get("feedback_classification", []),
        "paragraph_timings": paragraph_timings,
        "paragraphs_count": len(paragraphs),
        "feedback_count": len(feedback_items),
        "mapped_feedback_count": len(mapped_feedback),
        "affected_paragraphs": sorted(affected_indices),
        "tokens": usage,
    }


@app.get("/api/fix-lab/triage/{triage_id}")
async def get_triage_result(triage_id: str, x_fix_lab_key: str = Header(...)):
    """Get a stored triage result by ID."""
    verify_secret(x_fix_lab_key)

    try:
        uuid.UUID(triage_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid triage ID format")

    rows = await sb_get(f"bite_audio_triage?id=eq.{triage_id}&select=*")
    if not rows:
        raise HTTPException(status_code=404, detail="Triage result not found")

    row = rows[0]
    # Parse JSONB fields in case they come as strings
    for field in ("segments_to_regen", "feedback_classification", "paragraph_timings"):
        if isinstance(row.get(field), str):
            row[field] = json.loads(row[field])

    return row


@app.get("/api/fix-lab/triage")
async def list_triage_results(
    bite_id: Optional[str] = None,
    language: Optional[str] = None,
    assignment_id: Optional[str] = None,
    decision: Optional[str] = None,
    pending_review: bool = False,
    limit: int = 20,
    x_fix_lab_key: str = Header(...),
):
    """
    List triage results with optional filters.

    Query params:
      - bite_id: filter by bite
      - language: filter by language
      - assignment_id: filter by assignment
      - decision: filter by decision type
      - pending_review: if true, only show results without admin_action
      - limit: max results (default 20, max 100)
    """
    verify_secret(x_fix_lab_key)

    query = "bite_audio_triage?select=*&order=created_at.desc"
    if bite_id:
        query += f"&bite_id=eq.{bite_id}"
    if language:
        query += f"&language=eq.{language}"
    if assignment_id:
        query += f"&assignment_id=eq.{assignment_id}"
    if decision:
        query += f"&decision=eq.{decision}"
    if pending_review:
        query += "&admin_action=is.null&status=eq.completed"
    query += f"&limit={min(limit, 100)}"

    rows = await sb_get(query)

    for row in rows:
        for field in ("segments_to_regen", "feedback_classification", "paragraph_timings"):
            if isinstance(row.get(field), str):
                row[field] = json.loads(row[field])

    return {"results": rows, "count": len(rows)}


@app.patch("/api/fix-lab/triage/{triage_id}")
async def update_triage_result(
    triage_id: str,
    request: TriageAdminAction,
    x_fix_lab_key: str = Header(...),
):
    """
    Admin action on a triage result.

    Actions:
      - approved: AI recommendation accepted, proceed to regeneration (Phase 2+)
      - rejected: AI recommendation rejected, no action taken
      - modified: Admin made changes, notes describe modifications
    """
    verify_secret(x_fix_lab_key)

    try:
        uuid.UUID(triage_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid triage ID format")

    if request.admin_action not in ("approved", "rejected", "modified"):
        raise HTTPException(
            status_code=400,
            detail="admin_action must be one of: approved, rejected, modified"
        )

    await sb_patch("bite_audio_triage", triage_id, {
        "admin_action": request.admin_action,
        "admin_notes": request.admin_notes,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    })

    logger.info(f"Triage {triage_id}: Admin action → {request.admin_action}")

    return {
        "triage_id": triage_id,
        "admin_action": request.admin_action,
        "admin_notes": request.admin_notes,
    }


# ── Main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    logger.info(f"Starting Fix Lab Server on port {PORT}")
    uvicorn.run(app, host="0.0.0.0", port=PORT)

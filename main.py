"""
Fix Lab Backend Server
======================
FastAPI server that handles audio regeneration for the Fix Lab feature,
and RMS → Linear status sync.

Endpoints:
  GET  /                                    — Health check
  POST /api/fix-lab/regenerate              — Start audio regeneration for selected items
  GET  /api/fix-lab/jobs/{job_id}           — Poll job progress
  POST /api/linear-sync/bites              — Sync completed bites → Linear Approved
  POST /api/linear-sync/summaries          — Sync completed summaries → Linear Approved
  GET  /api/linear-sync/status/{type}      — Compare RMS vs Linear statuses
  GET  /api/linear-sync/jobs/{job_id}      — Poll sync job progress

Auth: x-fix-lab-key header must match FIX_LAB_SECRET env var.

Job state is persisted in Supabase tables (fix_lab_jobs, fix_lab_job_items)
so jobs survive server restarts and Render cold-starts.
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

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
ELEVEN_LABS_API_KEY = os.getenv("ELEVEN_LABS_API_KEY")
FIX_LAB_SECRET = os.getenv("FIX_LAB_SECRET", "kitab-fix-lab-2024")
LINEAR_API_KEY = os.getenv("LINEAR_API_KEY")
PORT = int(os.getenv("PORT", "8642"))

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

app = FastAPI(title="Fix Lab Server", version="2.0.0")

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


# ── Auth Dependency ─────────────────────────────────────────────────────────

def verify_secret(x_fix_lab_key: str = Header(...)):
    if x_fix_lab_key != FIX_LAB_SECRET:
        raise HTTPException(status_code=403, detail="Invalid Fix Lab key")
    return True


# ── Helper: Supabase queries via REST API ───────────────────────────────────

async def sb_get(path: str, params: dict = None) -> Any:
    """GET request to Supabase REST API."""
    r = await http_client.get(
        f"{SUPABASE_URL}/rest/v1/{path}",
        headers=SB_HEADERS,
        params=params or {},
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


async def sb_upload_storage(bucket: str, path: str, data: bytes, content_type: str = "audio/mpeg") -> str:
    """Upload a file to Supabase Storage and return its public URL."""
    r = await http_client.post(
        f"{SUPABASE_URL}/storage/v1/object/{bucket}/{path}",
        headers={
            "apikey": SUPABASE_SERVICE_KEY,
            "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
            "Content-Type": content_type,
            "x-upsert": "true",
        },
        content=data,
    )
    if r.status_code not in (200, 201):
        logger.error(f"Storage upload failed: {r.status_code} {r.text[:300]}")
        raise Exception(f"Storage upload failed: {r.status_code}")

    public_url = f"{SUPABASE_URL}/storage/v1/object/public/{bucket}/{path}"
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
    return {"status": "ok", "service": "fix-lab-server", "version": "2.0.0"}


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


# ── Main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    logger.info(f"Starting Fix Lab Server on port {PORT}")
    uvicorn.run(app, host="0.0.0.0", port=PORT)

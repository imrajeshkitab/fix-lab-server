"""
Fix Lab Backend Server
======================
FastAPI server that handles audio regeneration for the Fix Lab feature.

Endpoints:
  GET  /                            — Health check
  POST /api/fix-lab/regenerate     — Start audio regeneration for selected items
  GET  /api/fix-lab/jobs/{job_id}  — Poll job progress

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
    await sb_patch("fix_lab_job_items", item_id, {**data, "updated_at": datetime.now(timezone.utc).isoformat()})


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


# ── Main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    logger.info(f"Starting Fix Lab Server on port {PORT}")
    uvicorn.run(app, host="0.0.0.0", port=PORT)

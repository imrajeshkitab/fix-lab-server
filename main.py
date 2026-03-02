"""
Fix Lab Backend Server
======================
FastAPI server that handles audio regeneration for the Fix Lab feature.

Endpoints:
  GET  /api/fix-lab/items          — Fetch bites with changes_requested status
  POST /api/fix-lab/regenerate     — Start audio regeneration for selected items
  GET  /api/fix-lab/jobs/{job_id}  — Poll job progress

Auth: x-fix-lab-key header must match FIX_LAB_SECRET env var.
"""

import os
import sys
import uuid
import asyncio
import json
import time
import logging
from typing import Optional, List, Dict, Any
from datetime import datetime

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

app = FastAPI(title="Fix Lab Server", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Dev mode — tighten for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── In-memory job store ─────────────────────────────────────────────────────

jobs: Dict[str, Dict[str, Any]] = {}

# ── Pydantic Models ─────────────────────────────────────────────────────────

class RegenerateRequest(BaseModel):
    bite_languages: Dict[str, List[str]]  # { "bite-uuid": ["en"], "bite-uuid2": ["en","hi"] }


class JobStatus(BaseModel):
    job_id: str
    status: str  # "running", "completed", "failed"
    total: int
    completed: int
    failed: int
    results: List[Dict[str, Any]]


# ── Auth Dependency ─────────────────────────────────────────────────────────

def verify_secret(x_fix_lab_key: str = Header(...)):
    if x_fix_lab_key != FIX_LAB_SECRET:
        raise HTTPException(status_code=403, detail="Invalid Fix Lab key")
    return True


# ── Helper: Supabase queries via REST API ───────────────────────────────────

async def sb_get(path: str, params: dict = None) -> Any:
    """GET request to Supabase REST API."""
    async with httpx.AsyncClient() as client:
        r = await client.get(
            f"{SUPABASE_URL}/rest/v1/{path}",
            headers=SB_HEADERS,
            params=params or {},
        )
        if r.status_code not in (200, 206):
            logger.error(f"Supabase GET {path} failed: {r.status_code} {r.text[:300]}")
            raise HTTPException(status_code=500, detail=f"Supabase error: {r.status_code}")
        return r.json()


async def sb_patch(table: str, row_id: str, data: dict) -> Any:
    """PATCH (update) a row in Supabase."""
    async with httpx.AsyncClient() as client:
        r = await client.patch(
            f"{SUPABASE_URL}/rest/v1/{table}?id=eq.{row_id}",
            headers={**SB_HEADERS, "Prefer": "return=minimal"},
            json=data,
        )
        if r.status_code not in (200, 204):
            logger.error(f"Supabase PATCH {table}/{row_id} failed: {r.status_code} {r.text[:300]}")
            raise HTTPException(status_code=500, detail=f"Supabase update error")
        return True


async def sb_upload_storage(bucket: str, path: str, data: bytes, content_type: str = "audio/mpeg") -> str:
    """Upload a file to Supabase Storage and return its public URL."""
    async with httpx.AsyncClient() as client:
        # Try to upload (create)
        r = await client.post(
            f"{SUPABASE_URL}/storage/v1/object/{bucket}/{path}",
            headers={
                "apikey": SUPABASE_SERVICE_KEY,
                "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
                "Content-Type": content_type,
                "x-upsert": "true",  # Overwrite if exists
            },
            content=data,
        )
        if r.status_code not in (200, 201):
            logger.error(f"Storage upload failed: {r.status_code} {r.text[:300]}")
            raise Exception(f"Storage upload failed: {r.status_code}")

    # Return the public URL
    public_url = f"{SUPABASE_URL}/storage/v1/object/public/{bucket}/{path}"
    return public_url


# ── Helper: TTS Generation ─────────────────────────────────────────────────

def generate_audio(text: str, voice_id: str, language: str) -> bytes:
    """
    Generate TTS audio using the appropriate ElevenLabs model.
    EN → eleven_multilingual_v2 (from tts_11v2.1.py)
    HI → eleven_v3 (from tts_11v3.1.py)
    """
    from tts_engine import generate_audio as _generate
    return _generate(text, voice_id, language)


# ── Helper: Parse audio URL to extract current round ────────────────────────

def parse_audio_round(url: str) -> int:
    """Extract the current round number from a bite audio URL like .../round3/..."""
    if not url:
        return 1
    import re
    match = re.search(r'/round(\d+)/', url)
    if match:
        return int(match.group(1))
    # Also check for /v1/, /v2/ etc.
    match = re.search(r'/v(\d+)/', url)
    if match:
        return int(match.group(1))
    return 1


def build_new_audio_path(source_id: str, language: str, new_round: int) -> str:
    """Build the storage path for a new audio file."""
    lang_folder = "english" if language == "en" else "hindi"
    return f"bites/audio/{lang_folder}/round{new_round}/{source_id}.mp3"


# ── Regeneration Job Worker ─────────────────────────────────────────────────

async def run_regeneration_job(job_id: str, bite_languages: Dict[str, List[str]]):
    """Background task: regenerate audio for each bite with its specific languages."""
    job = jobs[job_id]

    bite_ids = list(bite_languages.keys())

    # Fetch the bite data
    try:
        # Fetch all bites by IDs
        ids_filter = ",".join(bite_ids)
        bites = await sb_get(f"bites?id=in.({ids_filter})&select=*")
    except Exception as e:
        job["status"] = "failed"
        job["error"] = str(e)
        logger.error(f"Job {job_id}: Failed to fetch bites: {e}")
        return

    # Calculate exact total
    job["total"] = sum(len(langs) for langs in bite_languages.values())

    for bite in bites:
        bite_id = bite["id"]
        source_id = bite.get("source_id", bite_id)
        audio_data = bite.get("audio", {}) or {}
        content_data = bite.get("content", {}) or {}
        audio_version = bite.get("audio_version", {}) or {"en": 1, "hi": 1}

        # Only regenerate the languages requested for THIS specific bite
        languages_for_bite = bite_languages.get(bite_id, [])

        for lang in languages_for_bite:
            item_key = f"{bite_id}_{lang}"
            result = {
                "bite_id": bite_id,
                "source_id": source_id,
                "language": lang,
                "status": "processing",
                "title": bite.get("title", "Unknown"),
            }
            job["results"].append(result)

            try:
                # 1. Get the text content
                lang_content = content_data.get(lang, {})
                if isinstance(lang_content, dict):
                    text = lang_content.get("text", "") or lang_content.get("body", "")
                else:
                    text = str(lang_content) if lang_content else ""

                if not text:
                    result["status"] = "skipped"
                    result["error"] = f"No {lang} content text found"
                    job["failed"] += 1
                    logger.warning(f"Job {job_id}: Skipped {bite_id}/{lang} — no text")
                    continue

                # 2. Get the voice ID
                lang_audio = audio_data.get(lang, {}) or {}
                vo_artist = lang_audio.get("vo_artist", "")
                voice_id = get_voice_id(vo_artist, lang)

                # 3. Determine the new round number
                current_url = lang_audio.get("url", "")
                current_round = parse_audio_round(current_url)
                new_round = current_round + 1
                current_version = audio_version.get(lang, 1)
                new_version = current_version + 1

                logger.info(f"Job {job_id}: Generating {bite_id}/{lang} "
                           f"(voice={vo_artist}, round {current_round}→{new_round})")

                # 4. Generate TTS audio (blocking — run in thread pool)
                audio_bytes = await asyncio.to_thread(
                    generate_audio, text, voice_id, lang
                )

                result["status"] = "uploading"
                logger.info(f"Job {job_id}: Generated {len(audio_bytes)} bytes for {bite_id}/{lang}")

                # 4b. Calculate audio duration using mutagen (pure Python, no ffmpeg needed)
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
                    duration_str = None
                    logger.warning(f"Job {job_id}: Could not calculate duration: {e}")

                # 5. Upload to Supabase Storage
                storage_path = build_new_audio_path(source_id, lang, new_round)
                new_url = await sb_upload_storage(
                    bucket="RMS-content",
                    path=storage_path,
                    data=audio_bytes,
                )

                # 6. Update the database
                # Update audio URL + duration
                updated_audio = dict(audio_data)
                lang_audio_updated = dict(lang_audio)
                lang_audio_updated["url"] = new_url
                if duration_str is not None:
                    lang_audio_updated["duration"] = duration_str
                updated_audio[lang] = lang_audio_updated

                # Update audio_version
                updated_version = dict(audio_version)
                updated_version[lang] = new_version

                await sb_patch("bites", bite_id, {
                    "audio": updated_audio,
                    "audio_version": updated_version,
                })

                result["status"] = "completed"
                result["new_url"] = new_url
                result["new_round"] = new_round
                result["new_version"] = new_version
                result["audio_size"] = len(audio_bytes)
                result["duration"] = duration_str
                result["char_count"] = len(text)
                job["completed"] += 1

                logger.info(f"Job {job_id}: ✅ {bite_id}/{lang} → round{new_round}, v{new_version}, {duration_str}")

            except Exception as e:
                result["status"] = "failed"
                result["error"] = str(e)
                job["failed"] += 1
                logger.error(f"Job {job_id}: ❌ {bite_id}/{lang} — {e}")

            # Small delay between items to avoid rate limiting
            await asyncio.sleep(0.5)

    # Mark job as complete
    if job["failed"] == job["total"]:
        job["status"] = "failed"
    elif job["failed"] > 0:
        job["status"] = "completed_with_errors"
    else:
        job["status"] = "completed"

    logger.info(f"Job {job_id}: Done — {job['completed']}/{job['total']} succeeded, "
               f"{job['failed']} failed")


# ═══════════════════════════════════════════════════════════════════════════
# API ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════

@app.get("/")
async def health():
    return {"status": "ok", "service": "fix-lab-server", "version": "1.0.0"}


@app.post("/api/fix-lab/regenerate")
async def start_regeneration(
    request: RegenerateRequest,
    background_tasks: BackgroundTasks,
    x_fix_lab_key: str = Header(...),
):
    """Start a regeneration job for selected bites."""
    verify_secret(x_fix_lab_key)

    if not request.bite_languages:
        raise HTTPException(status_code=400, detail="No bite_languages provided")

    total_items = sum(len(langs) for langs in request.bite_languages.values())

    job_id = str(uuid.uuid4())
    jobs[job_id] = {
        "job_id": job_id,
        "status": "running",
        "total": total_items,
        "completed": 0,
        "failed": 0,
        "results": [],
        "started_at": datetime.now(datetime.UTC).isoformat() if hasattr(datetime, 'UTC') else datetime.utcnow().isoformat(),
    }

    # Run in background
    background_tasks.add_task(
        run_regeneration_job,
        job_id,
        request.bite_languages,
    )

    return {"job_id": job_id, "status": "running", "total": total_items}


@app.get("/api/fix-lab/jobs/{job_id}")
async def get_job_status(job_id: str, x_fix_lab_key: str = Header(...)):
    """Poll job progress."""
    verify_secret(x_fix_lab_key)

    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    return jobs[job_id]


# ── Main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    logger.info(f"Starting Fix Lab Server on port {PORT}")
    uvicorn.run(app, host="0.0.0.0", port=PORT)

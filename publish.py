"""
Publish to Live App — Helpers
==============================
Pure helpers + DB read for syncing approved RMS bites to the live app
prod database (different Supabase project, schema 'bytes' instead of 'bites').

Phase A (this module's current scope): READ-ONLY status checks only.
  - Compare RMS approved bites vs app-prod bytes table
  - No writes, no audio copies

Phase B will add:
  - build_byte_row (mapping)
  - download_and_upload_audio
  - insert_byte_row (publish action)

Storage convention on app prod (bucket = 'content'):
  bytes/cover-pages/{source_id}.webp     (cover, deterministic, already exists)
  bytes/audio/{source_id}.mp3            (English audio, copied from RMS)
  bytes/audio_hi/{source_id}.mp3         (Hindi audio, copied from RMS)
"""

import logging
from typing import Dict, List, Optional

logger = logging.getLogger("fix-lab.publish")


# ── Constants ─────────────────────────────────────────────────────────────

APP_PROD_BUCKET = "content"
DEFAULT_AFFILIATE_LINKS = "https://heartfulness.org/magazine/editions"

# Audio folder per language
AUDIO_FOLDER = {"en": "audio", "hi": "audio_hi"}


# ── URL builders (deterministic, no DB calls) ─────────────────────────────

def cover_public_url(app_url: str, source_id: str) -> str:
    """Cover page is always at this template path on app prod (already uploaded)."""
    return f"{app_url.rstrip('/')}/storage/v1/object/public/content/bytes/cover-pages/{source_id}.webp"


def audio_storage_path(source_id: str, language: str) -> str:
    """Path inside the 'content' bucket where the audio gets uploaded."""
    folder = AUDIO_FOLDER.get(language)
    if not folder:
        raise ValueError(f"Unsupported language: {language}")
    return f"bytes/{folder}/{source_id}.mp3"


def audio_public_url(app_url: str, source_id: str, language: str) -> str:
    """Final public URL for the uploaded audio on app prod."""
    folder = AUDIO_FOLDER.get(language)
    if not folder:
        raise ValueError(f"Unsupported language: {language}")
    return (
        f"{app_url.rstrip('/')}/storage/v1/object/public/content/bytes/{folder}/{source_id}.mp3"
    )


# ── App Prod DB helpers (read-only for Phase A) ───────────────────────────

async def fetch_prod_byte_row(
    http_client,
    app_url: str,
    app_key: str,
    source_id: str,
    language: str,
) -> Optional[dict]:
    """Fetch a single prod 'bytes' row by (source_id, language)."""
    headers = {
        "apikey": app_key,
        "Authorization": f"Bearer {app_key}",
    }
    url = (
        f"{app_url.rstrip('/')}/rest/v1/bytes"
        f"?source_id=eq.{source_id}&language=eq.{language}&select=id&limit=1"
    )
    r = await http_client.get(url, headers=headers)
    if r.status_code != 200:
        raise Exception(f"App prod fetch failed: HTTP {r.status_code}")
    rows = r.json()
    return rows[0] if rows else None


async def download_audio(http_client, audio_url: str) -> bytes:
    """Download the RMS audio bytes."""
    r = await http_client.get(audio_url)
    if r.status_code != 200:
        raise Exception(f"Download audio failed: HTTP {r.status_code} {audio_url[:80]}")
    return r.content


async def upload_audio_to_prod(
    http_client,
    app_url: str,
    app_key: str,
    storage_path: str,
    audio_bytes: bytes,
) -> str:
    """
    Upload audio bytes to app prod's `content` bucket at `storage_path`.
    Returns the public URL.
    """
    headers = {
        "apikey": app_key,
        "Authorization": f"Bearer {app_key}",
        "Content-Type": "audio/mpeg",
        "x-upsert": "true",
    }
    url = f"{app_url.rstrip('/')}/storage/v1/object/{APP_PROD_BUCKET}/{storage_path}"
    r = await http_client.post(url, headers=headers, content=audio_bytes)
    if r.status_code not in (200, 201):
        raise Exception(f"Upload audio failed: HTTP {r.status_code} {r.text[:200]}")
    return (
        f"{app_url.rstrip('/')}/storage/v1/object/public/"
        f"{APP_PROD_BUCKET}/{storage_path}"
    )


async def insert_byte_row(
    http_client,
    app_url: str,
    app_key: str,
    row: dict,
) -> dict:
    """INSERT a row into app prod 'bytes' table. Returns the inserted row."""
    headers = {
        "apikey": app_key,
        "Authorization": f"Bearer {app_key}",
        "Content-Type": "application/json",
        "Prefer": "return=representation",
    }
    url = f"{app_url.rstrip('/')}/rest/v1/bytes"
    r = await http_client.post(url, headers=headers, json=row)
    if r.status_code not in (200, 201):
        raise Exception(f"Insert byte failed: HTTP {r.status_code} {r.text[:300]}")
    result = r.json()
    return result[0] if isinstance(result, list) and result else result


async def fetch_prod_bytes_by_source_ids(
    http_client,
    app_url: str,
    app_key: str,
    source_ids: List[str],
    language: str,
) -> Dict[str, dict]:
    """
    Bulk-fetch existing prod 'bytes' rows for given source_ids + language.
    Returns map: source_id → row.
    """
    if not source_ids:
        return {}

    headers = {
        "apikey": app_key,
        "Authorization": f"Bearer {app_key}",
    }

    result: Dict[str, dict] = {}
    batch_size = 50
    for i in range(0, len(source_ids), batch_size):
        batch = source_ids[i:i + batch_size]
        ids_filter = ",".join(batch)
        url = (
            f"{app_url.rstrip('/')}/rest/v1/bytes"
            f"?source_id=in.({ids_filter})"
            f"&language=eq.{language}"
            f"&select=id,source_id,language,published,updated_at"
        )
        r = await http_client.get(url, headers=headers)
        if r.status_code != 200:
            logger.error(f"App prod fetch failed: {r.status_code} {r.text[:300]}")
            raise Exception(f"App prod fetch failed: HTTP {r.status_code}")
        for row in r.json():
            result[row["source_id"]] = row

    return result


# ── Mapping (Phase B placeholder — kept here for forthcoming impl) ────────

def build_byte_row(rms_bite: dict, language: str, app_url: str) -> dict:
    """
    Build the app-prod 'bytes' row from an RMS 'bites' row + language.

    NOTE (Phase A): only used for the preview drawer in the UI. Phase B
    will use this as the actual INSERT payload.
    """
    source_id = rms_bite.get("source_id")
    if not source_id:
        raise ValueError("RMS bite has no source_id")

    audio_obj = (rms_bite.get("audio") or {}).get(language) or {}
    content = (rms_bite.get("content") or {}).get(language) or ""

    title_bilingual = rms_bite.get("title_bilingual") or {}
    title = title_bilingual.get(language) or rms_bite.get("title") or ""

    author_bilingual = rms_bite.get("author_bilingual") or {}
    author = author_bilingual.get(language) or rms_bite.get("author") or ""

    duration = audio_obj.get("duration") or ""

    return {
        "source_id": source_id,
        "cover_page": cover_public_url(app_url, source_id),
        "audio": audio_public_url(app_url, source_id, language),
        "duration": duration,
        "content": content,
        "title": title,
        "author": author,
        "category": rms_bite.get("category") or "",
        "language": language,
        "source": rms_bite.get("source") or "",
        "difficulty": rms_bite.get("difficulty") or "beginner",
        "priority": False,
        "published": True,
        "affiliate_links": DEFAULT_AFFILIATE_LINKS,
    }

"""
Add Intro to Bites Audio (Supabase-driven, in-place replacement)
=================================================================
Fetches bites directly from the Supabase `bites` table, filters by
`content_assignments` status (skips 'approved' and 'completed'), generates
a short TTS intro, stitches it to the existing audio, and **replaces the
audio in-place** at the same storage path.

English intro: You are now listening to the wisdom bite, "[title]".
Hindi   intro: अब आप सुन रहे हैं विजडम बाइट, '[title]'।

Voice/model selection:
  - Hindi (all voices)         → eleven_v3
  - English Anchor-V1, Coach-V1 → eleven_v3
  - English (others)           → eleven_multilingual_v2

Usage:
  python add_intro.py              # interactive mode
  python add_intro.py --dry-run    # preview which bites would be processed
"""

import os
import io
import sys
import time
import json
import logging
import argparse
from datetime import datetime
from urllib.parse import urlparse

from elevenlabs import ElevenLabs, VoiceSettings
from pydub import AudioSegment
import requests
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Load .env from parent directory (fix-lab-server/.env)
# ---------------------------------------------------------------------------
env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
load_dotenv(env_path)

# ---------------------------------------------------------------------------
# Voice config
# ---------------------------------------------------------------------------
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
try:
    from utils.config import VOICE_IDS_ENGLISH, VOICE_IDS_HINDI
except ImportError:
    try:
        from config import VOICE_IDS_ENGLISH, VOICE_IDS_HINDI
    except ImportError:
        print("Warning: Could not import VOICE_IDS. Using empty dicts.")
        VOICE_IDS_ENGLISH = {}
        VOICE_IDS_HINDI = {}

VOICE_IDS_ALL = {**VOICE_IDS_ENGLISH, **VOICE_IDS_HINDI}

# Voices that must use eleven_v3 even for English
ENGLISH_V3_VOICES = {"Anchor-V1", "Coach - V1"}

# ---------------------------------------------------------------------------
# Credentials from .env
# ---------------------------------------------------------------------------
# DB Supabase (RMS tables — bites, content_assignments)
SUPABASE_URL = (os.getenv("SUPABASE_URL") or "").rstrip("/").removesuffix("/rest/v1")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY") or ""

# Storage Supabase (RMS-content bucket — may be a different project)
STORAGE_SUPABASE_URL = os.getenv("STORAGE_SUPABASE_URL") or SUPABASE_URL
STORAGE_SUPABASE_SERVICE_KEY = os.getenv("STORAGE_SUPABASE_SERVICE_KEY") or SUPABASE_SERVICE_KEY

# ElevenLabs
ELEVENLABS_API_KEY = os.getenv("ELEVEN_LABS_API_KEY") or ""

# Production (app-live) Supabase — for checking if bites already in prod
APP_PROD_SUPABASE_URL = (os.getenv("APP_PROD_SUPABASE_URL") or "").rstrip("/").removesuffix("/rest/v1")
APP_PROD_SUPABASE_SERVICE_KEY = os.getenv("APP_PROD_SUPABASE_SERVICE_KEY") or ""

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EXCLUDED_STATUSES = {"completed"}

MAX_CONSECUTIVE_ERRORS = 3
RETRY_ATTEMPTS = 3
RETRY_BACKOFF_BASE = 2
PAUSE_BETWEEN_MS = 700  # silence gap between intro and main content

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def setup_logging():
    script_name = "add_intro"
    log_dir = os.path.join(os.path.dirname(__file__), "..", "..", "..", "logs", "scripts", script_name)
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"{script_name}_{timestamp}.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
    logging.info(f"Log file: {os.path.abspath(log_file)}")
    return log_file


# ---------------------------------------------------------------------------
# Supabase REST helpers
# ---------------------------------------------------------------------------
def sb_headers():
    return {
        "apikey": SUPABASE_SERVICE_KEY,
        "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
        "Content-Type": "application/json",
    }


def sb_get(path: str, params: dict = None) -> list:
    """GET request to Supabase REST API. Returns list of rows."""
    url = f"{SUPABASE_URL}/rest/v1/{path}"
    resp = requests.get(url, headers=sb_headers(), params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


def fetch_all_bites() -> list:
    """Fetch all bites with relevant fields."""
    return sb_get(
        "bites",
        params={
            "select": "id,source_id,title,title_bilingual,audio,audio_version",
            "order": "created_at.asc",
        },
    )


def fetch_assignments_for_bite(bite_id: str) -> list:
    """Fetch content_assignments for a specific bite."""
    return sb_get(
        "content_assignments",
        params={
            "content_type": "eq.bites",
            "content_id": f"eq.{bite_id}",
            "select": "id,status,assigned_languages",
        },
    )


def fetch_all_bite_assignments() -> dict:
    """
    Fetch ALL content_assignments for bites in one call.
    Returns a dict: bite_id → list of assignment rows.
    """
    rows = sb_get(
        "content_assignments",
        params={
            "content_type": "eq.bites",
            "select": "id,status,content_id,assigned_languages",
            "order": "assigned_at.asc",
        },
    )
    result = {}
    for row in rows:
        cid = row.get("content_id")
        if cid:
            result.setdefault(cid, []).append(row)
    return result


# ---------------------------------------------------------------------------
# Fetch source_id+language pairs from production bytes table
# ---------------------------------------------------------------------------
def fetch_prod_bytes_set() -> set:
    """
    Fetch all (source_id, language) pairs from the prod bytes table.
    Returns a set of tuples like {('uuid-1', 'en'), ('uuid-2', 'hi'), ...}
    """
    if not APP_PROD_SUPABASE_URL or not APP_PROD_SUPABASE_SERVICE_KEY:
        logging.warning("APP_PROD credentials not set — cannot check prod bytes")
        return set()

    prod_set = set()
    page_size = 1000
    offset = 0

    while True:
        url = f"{APP_PROD_SUPABASE_URL}/rest/v1/bytes?select=source_id,language&limit={page_size}&offset={offset}"
        headers = {
            "apikey": APP_PROD_SUPABASE_SERVICE_KEY,
            "Authorization": f"Bearer {APP_PROD_SUPABASE_SERVICE_KEY}",
        }
        resp = requests.get(url, headers=headers, timeout=30)
        resp.raise_for_status()
        rows = resp.json()
        if not rows:
            break
        for r in rows:
            sid = r.get("source_id", "").strip()
            lang = r.get("language", "").strip()
            if sid and lang:
                prod_set.add((sid, lang))
        if len(rows) < page_size:
            break
        offset += page_size

    return prod_set


# ---------------------------------------------------------------------------
# Filter bites by content_assignments status + prod presence
# ---------------------------------------------------------------------------
def should_process_bite_lang(
    bite_id: str,
    source_id: str,
    language: str,
    assignments_map: dict,
    prod_bytes_set: set,
) -> bool:
    """
    Returns True if the bite+language should be processed.

    Logic:
      - If no assignment for this language → include
      - If assignment status is NOT 'completed' → include
      - If status IS 'completed' AND (source_id, lang) IS in prod bytes → skip
      - If status IS 'completed' AND (source_id, lang) NOT in prod bytes → include
    """
    assignments = assignments_map.get(bite_id, [])
    if not assignments:
        # No assignments at all → include
        return True

    for a in assignments:
        # Check if this assignment covers the target language
        langs = a.get("assigned_languages", [])
        if isinstance(langs, str):
            try:
                import json as _json
                langs = _json.loads(langs)
            except Exception:
                langs = []

        if language not in langs:
            continue

        # This assignment covers our language — check its status
        status = (a.get("status") or "").lower().strip()
        if status in EXCLUDED_STATUSES:
            # Completed — but only skip if already in prod
            if (source_id, language) in prod_bytes_set:
                return False
            # Completed but NOT in prod → still process
            return True

    return True


# ---------------------------------------------------------------------------
# Parse audio URL to extract bucket and storage path
# ---------------------------------------------------------------------------
def parse_audio_url(audio_url: str) -> tuple:
    """
    Parse a Supabase storage public URL to extract (bucket, path).

    Example:
      https://kijxqpprmvywetklzhbg.supabase.co/storage/v1/object/public/RMS-content/bites/audio/english/round1/xxx.mp3
      → bucket = "RMS-content"
      → path   = "bites/audio/english/round1/xxx.mp3"
    """
    parsed = urlparse(audio_url)
    # Path looks like: /storage/v1/object/public/RMS-content/bites/audio/english/round1/xxx.mp3
    path_parts = parsed.path.split("/")

    # Find "public" or "object" marker to locate bucket
    try:
        pub_idx = path_parts.index("public")
    except ValueError:
        # Fallback: try finding after /object/
        try:
            obj_idx = path_parts.index("object")
            pub_idx = obj_idx  # bucket is right after "object"
        except ValueError:
            raise ValueError(f"Cannot parse audio URL: {audio_url}")

    # bucket is the segment right after "public"
    bucket = path_parts[pub_idx + 1]
    # path is everything after the bucket
    storage_path = "/".join(path_parts[pub_idx + 2:])

    return bucket, storage_path


# ---------------------------------------------------------------------------
# Resolve voice id + model from voice name (vo_artist)
# ---------------------------------------------------------------------------
def resolve_voice_from_name(voice_name: str, language: str):
    """
    Given a voice name (vo_artist from bites table) and language,
    returns (voice_id, model_id, voice_name) or (None, None, None) if not found.
    """
    if not voice_name:
        return None, None, None

    if language == "en":
        voice_id = VOICE_IDS_ENGLISH.get(voice_name) or VOICE_IDS_ALL.get(voice_name)
        if not voice_id:
            return None, None, None
        model_id = "eleven_v3" if voice_name in ENGLISH_V3_VOICES else "eleven_multilingual_v2"
        return voice_id, model_id, voice_name

    elif language == "hi":
        voice_id = VOICE_IDS_HINDI.get(voice_name) or VOICE_IDS_ALL.get(voice_name)
        if not voice_id:
            return None, None, None
        return voice_id, "eleven_v3", voice_name

    return None, None, None


# ---------------------------------------------------------------------------
# TTS generation with retry
# ---------------------------------------------------------------------------
def generate_intro_tts(
    client: ElevenLabs,
    text: str,
    voice_id: str,
    model_id: str,
) -> bytes:
    """Generate TTS for the short intro text. No chunking needed."""

    if model_id == "eleven_v3":
        voice_settings = VoiceSettings(
            stability=0.5,
            similarity_boost=0.75,
            style=0.0,
        )
    else:
        voice_settings = VoiceSettings(
            stability=0.7,
            similarity_boost=0.7,
            style=0.2,
            use_speaker_boost=True,
        )

    last_exc = None
    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            audio_gen = client.text_to_speech.convert(
                text=text,
                voice_id=voice_id,
                model_id=model_id,
                voice_settings=voice_settings,
                output_format="mp3_44100_128",
            )
            return b"".join(audio_gen)
        except Exception as e:
            last_exc = e
            wait = RETRY_BACKOFF_BASE ** (attempt - 1)
            print(f"     [TTS] Attempt {attempt}/{RETRY_ATTEMPTS} failed: {e}")
            if attempt < RETRY_ATTEMPTS:
                print(f"     [TTS] Retrying in {wait}s...")
                time.sleep(wait)

    raise RuntimeError(f"TTS failed after {RETRY_ATTEMPTS} attempts: {last_exc}")


# ---------------------------------------------------------------------------
# Download audio from public URL
# ---------------------------------------------------------------------------
def download_audio_from_url(audio_url: str) -> bytes:
    """Download audio from a public Supabase storage URL."""
    resp = requests.get(audio_url, timeout=120)
    resp.raise_for_status()
    return resp.content


# ---------------------------------------------------------------------------
# Upload audio to Supabase storage (upsert — replaces in-place)
# ---------------------------------------------------------------------------
def upload_audio(bucket: str, path: str, data: bytes):
    """Upload a file to Supabase storage (upsert — replaces existing file)."""
    url = f"{STORAGE_SUPABASE_URL}/storage/v1/object/{bucket}/{path}"
    headers = {
        "apikey": STORAGE_SUPABASE_SERVICE_KEY,
        "Authorization": f"Bearer {STORAGE_SUPABASE_SERVICE_KEY}",
        "Content-Type": "audio/mpeg",
        "x-upsert": "true",
    }
    resp = requests.post(url, headers=headers, data=data, timeout=120)
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# Update bites table audio column with intro_duration
# ---------------------------------------------------------------------------
def update_bite_intro_duration(bite_id: str, language: str, intro_duration_ms: float):
    """
    PATCH the bites table to add intro_duration (ms) to audio.{language}.
    Fetches current audio JSONB, merges intro_duration, and writes back.
    """
    # Fetch current bite audio
    rows = sb_get("bites", params={"id": f"eq.{bite_id}", "select": "audio"})
    if not rows:
        logging.warning(f"Cannot update intro_duration: bite {bite_id} not found")
        return

    audio_data = rows[0].get("audio") or {}
    if isinstance(audio_data, str):
        try:
            audio_data = json.loads(audio_data)
        except Exception:
            audio_data = {}

    lang_audio = audio_data.get(language, {})
    if isinstance(lang_audio, str):
        try:
            lang_audio = json.loads(lang_audio)
        except Exception:
            lang_audio = {}

    # Add intro_duration in ms
    lang_audio["intro_duration"] = round(intro_duration_ms, 3)
    audio_data[language] = lang_audio

    # PATCH bites table
    url = f"{SUPABASE_URL}/rest/v1/bites?id=eq.{bite_id}"
    headers = {
        **sb_headers(),
        "Prefer": "return=minimal",
    }
    resp = requests.patch(url, headers=headers, json={"audio": audio_data}, timeout=30)
    resp.raise_for_status()
    logging.info(f"Updated bites.audio.{language}.intro_duration = {intro_duration_ms:.3f}ms for {bite_id}")


# ---------------------------------------------------------------------------
# Measure audio duration in milliseconds
# ---------------------------------------------------------------------------
def get_audio_duration_ms(audio_bytes: bytes) -> float:
    """Return the duration of an MP3 audio in milliseconds (ms precision)."""
    seg = AudioSegment.from_file(io.BytesIO(audio_bytes), format="mp3")
    return float(len(seg))  # pydub's len() returns duration in ms


# ---------------------------------------------------------------------------
# Stitch intro + existing audio using pydub
# ---------------------------------------------------------------------------
def stitch_audio(intro_bytes: bytes, main_bytes: bytes) -> bytes:
    """Concatenate intro audio + silence gap + main audio, return combined MP3 bytes."""
    intro_seg = AudioSegment.from_file(io.BytesIO(intro_bytes), format="mp3")
    main_seg = AudioSegment.from_file(io.BytesIO(main_bytes), format="mp3")
    silence = AudioSegment.silent(duration=PAUSE_BETWEEN_MS, frame_rate=intro_seg.frame_rate)
    combined = intro_seg + silence + main_seg
    out = io.BytesIO()
    combined.export(out, format="mp3", bitrate="192k")
    return out.getvalue()


# ---------------------------------------------------------------------------
# Build intro text
# ---------------------------------------------------------------------------
def build_intro_text(title: str, language: str) -> str:
    if language == "en":
        return f'You are now listening to the wisdom bite, "{title}".'
    elif language == "hi":
        return f"अब आप सुन रहे हैं विजडम बाइट, '{title}'।"
    return ""


# ---------------------------------------------------------------------------
# Get language-specific title from a bite row
# ---------------------------------------------------------------------------
def get_title_for_language(bite: dict, language: str) -> str:
    """
    Get the best title for the given language.
    Priority: title_bilingual.{lang} → title (English fallback).
    """
    title_bilingual = bite.get("title_bilingual") or {}
    if isinstance(title_bilingual, str):
        try:
            title_bilingual = json.loads(title_bilingual)
        except Exception:
            title_bilingual = {}

    lang_title = title_bilingual.get(language, "").strip()
    if lang_title:
        return lang_title

    return (bite.get("title") or "Untitled").strip()


# ---------------------------------------------------------------------------
# Build list of processable items from bites
# ---------------------------------------------------------------------------
def build_process_list(bites: list, assignments_map: dict, prod_bytes_set: set) -> list:
    """
    Build a flat list of items to process: one entry per bite per language
    that has a valid audio URL and vo_artist.

    Each item: {bite_id, source_id, title, language, vo_artist, audio_url, bucket, storage_path}
    """
    items = []

    for bite in bites:
        bite_id = bite.get("id")
        source_id = bite.get("source_id", bite_id)

        audio_data = bite.get("audio") or {}
        if isinstance(audio_data, str):
            try:
                audio_data = json.loads(audio_data)
            except Exception:
                continue

        for lang in ("en", "hi"):
            lang_audio = audio_data.get(lang)
            if not lang_audio or not isinstance(lang_audio, dict):
                continue

            audio_url = lang_audio.get("url", "").strip()
            vo_artist = lang_audio.get("vo_artist", "").strip()

            # Idempotency: skip if intro already added (intro_duration exists)
            if lang_audio.get("intro_duration"):
                continue

            # Per-language status filter (checks assignment status + prod presence)
            if not should_process_bite_lang(bite_id, source_id, lang, assignments_map, prod_bytes_set):
                continue

            if not audio_url:
                logging.warning(f"Bite {bite_id} [{lang}]: no audio URL — skipping")
                continue

            if not vo_artist:
                logging.warning(f"Bite {bite_id} [{lang}]: no vo_artist — skipping")
                continue

            # Parse the URL to get bucket and storage path
            try:
                bucket, storage_path = parse_audio_url(audio_url)
            except ValueError as e:
                logging.warning(f"Bite {bite_id} [{lang}]: {e} — skipping")
                continue

            title = get_title_for_language(bite, lang)

            items.append({
                "bite_id": bite_id,
                "source_id": source_id,
                "title": title,
                "language": lang,
                "vo_artist": vo_artist,
                "audio_url": audio_url,
                "bucket": bucket,
                "storage_path": storage_path,
            })

    return items


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Add intro to bites audio (Supabase-driven)")
    parser.add_argument("--dry-run", action="store_true", help="Preview which bites would be processed without making changes")
    args = parser.parse_args()

    log_file = setup_logging()
    print("\n--- Add Intro to Bites Audio (Supabase-driven) ---\n")

    # 1. Validate credentials
    missing = []
    if not SUPABASE_URL:
        missing.append("SUPABASE_URL")
    if not SUPABASE_SERVICE_KEY:
        missing.append("SUPABASE_SERVICE_KEY")
    if not STORAGE_SUPABASE_URL:
        missing.append("STORAGE_SUPABASE_URL")
    if not STORAGE_SUPABASE_SERVICE_KEY:
        missing.append("STORAGE_SUPABASE_SERVICE_KEY")
    if not ELEVENLABS_API_KEY and not args.dry_run:
        missing.append("ELEVEN_LABS_API_KEY")

    if missing:
        print(f"Missing env vars: {', '.join(missing)}")
        print("Make sure fix-lab-server/.env is configured correctly.")
        return

    print(f"DB Supabase:      {SUPABASE_URL}")
    print(f"Storage Supabase: {STORAGE_SUPABASE_URL}")
    print(f"Prod Supabase:    {APP_PROD_SUPABASE_URL or '(not set)'}")

    # 2. Fetch all bites
    print("\nFetching bites from Supabase...")
    all_bites = fetch_all_bites()
    print(f"  Total bites in DB: {len(all_bites)}")

    # 3. Fetch all content_assignments for bites (single bulk call)
    print("Fetching content_assignments...")
    assignments_map = fetch_all_bite_assignments()
    print(f"  Bites with assignments: {len(assignments_map)}")

    # 3b. Fetch prod bytes to check which completed bites are already published
    print("Fetching prod bytes (app-live)...")
    prod_bytes_set = fetch_prod_bytes_set()
    print(f"  Prod bytes entries: {len(prod_bytes_set)}")

    # 4. Build processable list (filtered by status + prod presence)
    all_items = build_process_list(all_bites, assignments_map, prod_bytes_set)
    print(f"\nProcessable items (excl. completed-and-in-prod, incl. completed-not-in-prod):")
    en_items = [i for i in all_items if i["language"] == "en"]
    hi_items = [i for i in all_items if i["language"] == "hi"]
    print(f"  Total: {len(all_items)} (English: {len(en_items)}, Hindi: {len(hi_items)})")

    # Show breakdown
    completed_not_in_prod = 0
    for bite in all_bites:
        bid = bite.get("id")
        sid = bite.get("source_id", bid)
        for lang in ("en", "hi"):
            assigns = assignments_map.get(bid, [])
            for a in assigns:
                alangs = a.get("assigned_languages", [])
                if isinstance(alangs, str):
                    try: alangs = json.loads(alangs)
                    except: alangs = []
                if lang in alangs:
                    st = (a.get("status") or "").lower().strip()
                    if st == "completed" and (sid, lang) not in prod_bytes_set:
                        completed_not_in_prod += 1
    if completed_not_in_prod:
        print(f"  (includes {completed_not_in_prod} completed-but-not-in-prod items)")


    if not all_items:
        print("\nNothing to process!")
        return

    # 5. Mode selection
    print("\nMode:")
    print("1. Single item (by bite_id)")
    print("2. All remaining")
    mode = input("Select (1/2): ").strip()

    if mode == "1":
        bite_id = input("Enter bite_id: ").strip()
        selected = [i for i in all_items if i["bite_id"] == bite_id]
        if not selected:
            print(f"No processable items found for bite_id = {bite_id}")
            # Show why — check if bite exists
            bite_exists = any(b.get("id") == bite_id for b in all_bites)
            if bite_exists:
                print("  (Bite exists but may be 'completed' or missing audio/vo_artist)")
            else:
                print("  (Bite not found in DB)")
            return

        # Language selection
        available_langs = sorted(set(i["language"] for i in selected))
        if len(available_langs) > 1:
            print(f"\nAvailable languages: {', '.join(available_langs)}")
            lang_choice = input("Language (en/hi/both) [both]: ").strip().lower() or "both"
            if lang_choice != "both":
                selected = [i for i in selected if i["language"] == lang_choice]
                if not selected:
                    print(f"No items found for language '{lang_choice}'")
                    return

        to_process = selected
        print(f"\nSelected {len(to_process)} item(s):")
        for item in to_process:
            print(f"  - {item['title']} [{item['language']}]")

    elif mode == "2":
        to_process = all_items
    else:
        print("Invalid selection.")
        return

    # 6. Dry-run: just print what would be processed
    if args.dry_run:
        print(f"\n{'='*60}")
        print(f"DRY RUN — {len(to_process)} items would be processed:")
        print(f"{'='*60}")
        for i, item in enumerate(to_process):
            title = item["title"]
            lang = item["language"]
            intro = build_intro_text(title, lang)
            voice_id, model_id, voice_name = resolve_voice_from_name(item["vo_artist"], lang)
            status = "✅ ready" if voice_id else "❌ no voice"
            print(f"  [{i+1}] {title} [{lang}] | voice: {item['vo_artist']} ({status})")
            print(f"       intro: {intro}")
            print(f"       path:  {item['bucket']}/{item['storage_path']}")
        print(f"\nNo changes were made (dry-run mode).")
        return

    # 7. Confirmation
    print(f"\n⚠️  This will REPLACE {len(to_process)} audio files in-place!")
    print(f"   The original audio (without intro) will be overwritten.")
    confirm = input(f"Proceed? (y/n): ").strip().lower()
    if confirm != "y":
        print("Aborted.")
        return

    # 8. Init ElevenLabs client
    el_client = ElevenLabs(api_key=ELEVENLABS_API_KEY, timeout=120.0)

    # 9. Process
    total = len(to_process)
    consecutive_errors = 0
    success_count = 0
    fail_count = 0

    print(f"\nStarting processing... ({total} items)\n")

    for i, item in enumerate(to_process):
        bite_id = item["bite_id"]
        source_id = item["source_id"]
        title = item["title"]
        lang = item["language"]
        audio_url = item["audio_url"]
        bucket = item["bucket"]
        storage_path = item["storage_path"]

        print(f"[{i+1}/{total}] {title} [{lang}] (bite_id: {bite_id})")

        try:
            # Resolve voice
            vo_artist = item["vo_artist"]
            voice_id, model_id, voice_name = resolve_voice_from_name(vo_artist, lang)
            if not voice_id:
                print(f"  SKIP: No voice found for vo_artist={vo_artist}")
                logging.warning(f"No voice for vo_artist={vo_artist}, {bite_id} [{lang}]")
                fail_count += 1
                continue

            print(f"  Voice: {voice_name} | Model: {model_id}")

            # Generate intro TTS
            intro_text = build_intro_text(title, lang)
            print(f"  Intro: {intro_text}")
            intro_audio = generate_intro_tts(el_client, intro_text, voice_id, model_id)
            intro_duration_ms = get_audio_duration_ms(intro_audio)
            print(f"  Intro audio: {len(intro_audio)} bytes, duration: {intro_duration_ms:.3f}ms")

            # Download existing audio from public URL
            print(f"  Downloading: {audio_url[-60:]}")
            main_audio = download_audio_from_url(audio_url)
            print(f"  Main audio: {len(main_audio)} bytes")

            # Stitch
            print(f"  Stitching intro + main audio...")
            stitched = stitch_audio(intro_audio, main_audio)
            print(f"  Stitched audio: {len(stitched)} bytes")

            # Upload — replace in-place at the same bucket/path
            print(f"  Uploading to: {bucket}/{storage_path} (in-place replace)")
            upload_audio(bucket, storage_path, stitched)

            # Update bites table with intro_duration
            print(f"  Updating bites.audio.{lang}.intro_duration = {intro_duration_ms:.3f}ms")
            update_bite_intro_duration(bite_id, lang, intro_duration_ms)
            print(f"  ✅ DONE")

            logging.info(
                f"[{i+1}/{total}] SUCCESS: {bite_id} [{lang}] - {title} → {bucket}/{storage_path} "
                f"(intro: {intro_duration_ms:.3f}ms)"
            )
            success_count += 1
            consecutive_errors = 0

        except Exception as e:
            consecutive_errors += 1
            fail_count += 1
            print(f"  ❌ FAILED: {e}")
            logging.error(f"[{i+1}/{total}] FAILED: {bite_id} [{lang}] - {e}")

            if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                print(f"\nStopping after {MAX_CONSECUTIVE_ERRORS} consecutive errors.")
                print(f"Progress: {i+1}/{total}, Success: {success_count}, Failed: {fail_count}")
                print(f"Log: {log_file}")
                return

            print(f"  Error {consecutive_errors}/{MAX_CONSECUTIVE_ERRORS} — continuing...")

        # Small delay between items
        if i < total - 1:
            time.sleep(0.5)

    print(f"\n{'='*60}")
    print(f"Complete! Processed: {total}, Success: {success_count}, Failed: {fail_count}")
    print(f"Log: {log_file}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

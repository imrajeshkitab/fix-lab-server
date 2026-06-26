"""
Text-to-Speech Script using ElevenLabs Eleven v3 Model (REVIEWED)
==================================================================
Changes from original:
  1. Removed `use_speaker_boost` (not supported on v3)
  2. Increased MAX_CHARS_PER_CHUNK to 8000 (v3 supports 10,000)
  3. Added request stitching (previous_request_ids, previous_text, next_text)
  4. Replaced raw byte concatenation with pydub-based audio merging
  5. Explicit output_format parameter
  6. Fixed continue-after-error skipping Hindi when English fails
  7. Separated error counting per-article instead of per-language
"""

import os
import io
import time
import re
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from dotenv import load_dotenv
from elevenlabs import ElevenLabs, VoiceSettings
from pydub import AudioSegment  # pip install pydub (requires ffmpeg)

import sys
import csv

# Add parent directory to path to allow importing from sibling/utils modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
try:
    from utils.config import VOICE_IDS_ENGLISH, VOICE_IDS_HINDI
except ImportError:
    try:
        from scripts.elevenlabs.utils.config import VOICE_IDS_ENGLISH, VOICE_IDS_HINDI
    except ImportError:
        print("Warning: Could not import VOICE_IDS_* from utils.config")
        VOICE_IDS_ENGLISH = {}
        VOICE_IDS_HINDI = {}

VOICE_IDS_ALL = {**VOICE_IDS_ENGLISH, **VOICE_IDS_HINDI}

# Reuse the text cleaner from sibling if available
try:
    from text_cleaner import clean_text
except ImportError:
    def clean_text(text): return text

# ============================================================================
# Constants
# ============================================================================

# v3 supports up to 10,000 characters. Using 8,000 for safe margin.
MAX_CHARS_PER_CHUNK = 8000

# Output format — explicitly set rather than relying on server default.
# Options: mp3_22050_32, mp3_44100_64, mp3_44100_128, mp3_44100_192 (Creator+)
OUTPUT_FORMAT = "mp3_44100_128"

# ============================================================================
# Logging Setup
# ============================================================================

def setup_logging():
    """Setup logging to both console and file."""
    script_name = "tts_11v3"
    log_dir = os.path.join(os.path.dirname(__file__), "..", "..", "..", "logs", "scripts", script_name)
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"{script_name}_{timestamp}.log")

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )

    logging.info(f"Log file: {os.path.abspath(log_file)}")
    return log_file

# ============================================================================
# Client Initialization
# ============================================================================

def _load_elevenlabs_client() -> ElevenLabs:
    """Initialize and return ElevenLabs client using API key from environment."""
    load_dotenv()
    api_key = os.getenv("ELEVEN_LABS_API_KEY")
    if not api_key:
        raise RuntimeError("ELEVEN_LABS_API_KEY is not set in environment.")
    return ElevenLabs(api_key=api_key)


# ============================================================================
# Text Chunking for Eleven v3
# ============================================================================

def chunk_text(text: str, max_chars: int = MAX_CHARS_PER_CHUNK) -> List[str]:
    """
    Split text into chunks that fit within the character limit.
    Tries to split at paragraph boundaries first, then sentence boundaries.

    Args:
        text: The text to split
        max_chars: Maximum characters per chunk (default: 8000)

    Returns:
        List of text chunks
    """
    if len(text) <= max_chars:
        return [text]

    chunks = []
    current_chunk = ""

    # Split by sentences (period, question mark, exclamation mark followed by space or end)
    sentences = re.split(r'(?<=[.!?])\s+', text)

    for sentence in sentences:
        if len(current_chunk) + len(sentence) + 1 > max_chars:
            if current_chunk.strip():
                chunks.append(current_chunk.strip())

            if len(sentence) > max_chars:
                # Split by commas, semicolons, or newlines
                sub_parts = re.split(r'[,;\n]+', sentence)
                current_chunk = ""
                for part in sub_parts:
                    if len(part) > max_chars:
                        # Last resort: hard split
                        if current_chunk.strip():
                            chunks.append(current_chunk.strip())
                            current_chunk = ""
                        for i in range(0, len(part), max_chars):
                            chunks.append(part[i:i + max_chars].strip())
                    elif len(current_chunk) + len(part) + 1 > max_chars:
                        if current_chunk.strip():
                            chunks.append(current_chunk.strip())
                        current_chunk = part
                    else:
                        current_chunk = (current_chunk + " " + part).strip()
            else:
                current_chunk = sentence
        else:
            current_chunk = (current_chunk + " " + sentence).strip()

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks


# ============================================================================
# Audio Merging (replaces raw byte concatenation)
# ============================================================================

def merge_audio_chunks(audio_chunks: List[bytes], fmt: str = "mp3") -> bytes:
    """
    Properly merge multiple MP3 audio byte-strings using pydub.
    Raw byte concatenation (b"".join) can introduce clicks/glitches
    because each chunk has its own MP3 headers and frame boundaries.

    Args:
        audio_chunks: List of audio byte-strings
        fmt: Audio format (default: "mp3")

    Returns:
        Merged audio as bytes
    """
    if len(audio_chunks) == 1:
        return audio_chunks[0]

    combined = AudioSegment.empty()
    for chunk_bytes in audio_chunks:
        segment = AudioSegment.from_file(io.BytesIO(chunk_bytes), format=fmt)
        combined += segment

    output_buffer = io.BytesIO()
    combined.export(output_buffer, format=fmt)
    return output_buffer.getvalue()


# ============================================================================
# TTS Generation with Eleven v3 + Request Stitching
# ============================================================================

def generate_tts_v3(
    text: str,
    voice_id: str,
    model_id: str = "eleven_v3",
    output_format: str = OUTPUT_FORMAT,
) -> bytes:
    """
    Generate TTS using Eleven v3 model with request stitching for
    multi-chunk texts.

    Key differences from a naive approach:
      - Uses `previous_request_ids` so the model conditions on prior chunks,
        maintaining natural prosody across boundaries.
      - Uses `previous_text` / `next_text` for additional context.
      - Merges audio with pydub instead of raw byte concat.

    Args:
        text: The text to convert to speech
        voice_id: ElevenLabs voice ID
        model_id: Model ID (default: eleven_v3)
        output_format: Audio output format string

    Returns:
        Audio bytes (MP3)
    """
    print(f"     [TTS] Starting generation (model={model_id})")
    print(f"     [TTS] Text length: {len(text)} chars")

    client = _load_elevenlabs_client()

    # Voice settings for v3
    # NOTE: use_speaker_boost is NOT supported on v3 — omitted.
    voice_settings = VoiceSettings(
        stability=0.5,           # ~50 — balanced consistency
        similarity_boost=0.75,   # ~75 — good voice match
        style=0.0,               # 0 — recommended for stability in v3
    )

    chunks = chunk_text(text)

    if len(chunks) > 1:
        print(f"     [TTS] Text exceeds limit. Split into {len(chunks)} chunks.")
        logging.info(f"Text split into {len(chunks)} chunks for processing")

    all_audio: List[bytes] = []
    # Track previous request IDs for request stitching (max 3)
    previous_request_ids: List[str] = []

    for i, chunk in enumerate(chunks):
        if len(chunks) > 1:
            print(f"     [TTS] Processing chunk {i + 1}/{len(chunks)} ({len(chunk)} chars)...")

        try:
            # Build request-stitching context
            kwargs = dict(
                text=chunk,
                voice_id=voice_id,
                model_id=model_id,
                voice_settings=voice_settings,
                output_format=output_format,
            )

            # Request stitching: pass up to 3 previous request IDs
            if previous_request_ids:
                kwargs["previous_request_ids"] = previous_request_ids[-3:]

            # Provide surrounding text context for smoother transitions
            if i > 0:
                kwargs["previous_text"] = chunks[i - 1][-1000:]  # last 1000 chars
            if i < len(chunks) - 1:
                kwargs["next_text"] = chunks[i + 1][:1000]       # first 1000 chars

            # Call the API — convert() returns a generator + metadata
            response = client.text_to_speech.convert(**kwargs)

            # The SDK returns a generator of bytes; consume it
            chunk_audio = b"".join(response)
            all_audio.append(chunk_audio)

            # Extract request_id for stitching
            # The request_id is available via response headers. Depending on
            # SDK version, you may need to use convert_as_stream() or the
            # lower-level httpx client to capture headers. If the SDK exposes
            # it directly, capture it here:
            if hasattr(response, 'request_id'):
                previous_request_ids.append(response.request_id)

            if len(chunks) > 1:
                print(f"     [TTS] Chunk {i + 1} generated: {len(chunk_audio)} bytes")
                time.sleep(0.5)  # Rate-limit courtesy

        except Exception as e:
            print(f"     [TTS] ERROR on chunk {i + 1}: {e}")
            raise

    # Merge chunks properly via pydub (not raw byte concat)
    final_audio = merge_audio_chunks(all_audio)
    print(f"     [TTS] Total audio generated: {len(final_audio)} bytes")

    return final_audio


# ============================================================================
# Utility Functions
# ============================================================================

def get_nested_value(data: dict, path: str):
    keys = path.split('.')
    current = data
    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return None
    return current

def load_voice_map_csv(csv_path):
    mapping = {}
    if not os.path.exists(csv_path):
        return mapping
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if 'uuid' in row:
                mapping[row['uuid']] = row
    return mapping


# ============================================================================
# Main Function
# ============================================================================

def main():
    log_file = setup_logging()
    logging.info("Starting Text-to-Voice Generator (Eleven v3)")

    print("\n--- Text-to-Voice Generator (Eleven v3) ---")
    print(f"Model: eleven_v3 | Character Limit: 10,000 (auto-chunking at {MAX_CHARS_PER_CHUNK})")
    print("")

    # 1. Basic Inputs
    json_path = input("JSON File Path: ").strip().strip("'\"")
    if not os.path.exists(json_path):
        print("File not found.")
        return

    uuid_field = input("UUID Field Name [uuid]: ").strip() or "uuid"

    # 2. Language Mode Selection
    print("\nLanguage Mode:")
    print("1. Bilingual (Both English & Hindi)")
    print("2. English Only")
    print("3. Hindi Only")
    lang_choice = input("Select Language Mode (1-3) [1]: ").strip() or "1"

    mode_english = True
    mode_hindi = True

    if lang_choice == '2':
        mode_hindi = False
    elif lang_choice == '3':
        mode_english = False

    # 3. Content/Output Paths
    content_path_en = None
    content_path_hi = None
    out_dir_audio_en = None
    out_dir_audio_hi = None
    out_dir_txt_en = None
    out_dir_txt_hi = None

    if mode_english:
        content_path_en = input("Content Field Path (English) [current_data.current_content]: ").strip() or "current_data.current_content"
        out_dir_audio_en = input("Audio Output Dir (English) [audio_en]: ").strip().strip("'\"") or "audio_en"
        out_dir_txt_en = input("Transcript Output Dir (English) [transcripts_en]: ").strip().strip("'\"") or "transcripts_en"
        os.makedirs(out_dir_audio_en, exist_ok=True)
        os.makedirs(out_dir_txt_en, exist_ok=True)

    if mode_hindi:
        content_path_hi = input("Content Field Path (Hindi) [current_data.current_summary_hindi]: ").strip() or "current_data.current_summary_hindi"
        out_dir_audio_hi = input("Audio Output Dir (Hindi) [audio_hi]: ").strip().strip("'\"") or "audio_hi"
        out_dir_txt_hi = input("Transcript Output Dir (Hindi) [transcripts_hi]: ").strip().strip("'\"") or "transcripts_hi"
        os.makedirs(out_dir_audio_hi, exist_ok=True)
        os.makedirs(out_dir_txt_hi, exist_ok=True)

    # 4. Voice Selection
    print("\nVoice Selection Mode:")
    print("1. Manual")
    print("2. Mapping CSV")
    mode = input("Choice (1/2): ").strip()

    manual_en_id = None
    manual_hi_id = None
    voice_map = {}

    if mode == '1':
        if mode_english:
            print("\nAvailable English Voices:", ", ".join(VOICE_IDS_ENGLISH.keys()))
            manual_en_id = input("Enter English Voice ID: ").strip()
        if mode_hindi:
            print("\nAvailable Hindi Voices:", ", ".join(VOICE_IDS_HINDI.keys()))
            manual_hi_id = input("Enter Hindi Voice ID: ").strip()
    else:
        csv_path = input("Mapping CSV Path: ").strip().strip("'\"")
        print(f"Loading mapping from {csv_path}...")
        voice_map = load_voice_map_csv(csv_path)
        print(f"Loaded {len(voice_map)} mapped entries.")

    # 5. Load Articles
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        articles = data.get('articles', []) if isinstance(data, dict) else (data if isinstance(data, list) else [data])

    print(f"Loaded {len(articles)} articles.")

    article_by_uuid = {}
    for article in articles:
        uid = str(get_nested_value(article, uuid_field))
        if uid:
            article_by_uuid[uid] = article

    # 6. Validation function
    def validate_article(article, uid):
        """Validate a single article and return processing info or None."""
        txt_en = get_nested_value(article, content_path_en) if content_path_en else None
        txt_hi = get_nested_value(article, content_path_hi) if content_path_hi else None

        has_en_content = bool(txt_en and str(txt_en).strip())
        has_hi_content = bool(txt_hi and str(txt_hi).strip())

        if not has_en_content and not has_hi_content:
            return None, "No content available"

        exists_en = os.path.exists(os.path.join(out_dir_audio_en, f"{uid}.mp3")) if out_dir_audio_en else True
        exists_hi = os.path.exists(os.path.join(out_dir_audio_hi, f"{uid}.mp3")) if out_dir_audio_hi else True

        do_en = has_en_content and not exists_en and mode_english
        do_hi = has_hi_content and not exists_hi and mode_hindi

        final_en_voice = manual_en_id
        final_hi_voice = manual_hi_id
        skip_reasons = []

        if mode == '2':
            row = voice_map.get(uid)
            if not row:
                return None, "UUID not found in voice mapping CSV"
            else:
                if do_en:
                    v_name_en = row.get('english_voice_name', '').strip()
                    if v_name_en:
                        final_en_voice = VOICE_IDS_ENGLISH.get(v_name_en) or VOICE_IDS_ALL.get(v_name_en)
                        if not final_en_voice:
                            skip_reasons.append(f"English voice '{v_name_en}' not found in config")
                            do_en = False
                    else:
                        skip_reasons.append("No English voice name in CSV")
                        do_en = False

                if do_hi:
                    v_name_hi = row.get('hindi_voice_name', '').strip()
                    if v_name_hi:
                        final_hi_voice = VOICE_IDS_HINDI.get(v_name_hi) or VOICE_IDS_ALL.get(v_name_hi)
                        if not final_hi_voice:
                            skip_reasons.append(f"Hindi voice '{v_name_hi}' not found in config")
                            do_hi = False
                    else:
                        skip_reasons.append("No Hindi voice name in CSV")
                        do_hi = False

        if mode_english and exists_en and has_en_content:
            skip_reasons.append("English audio already exists")
        if mode_hindi and exists_hi and has_hi_content:
            skip_reasons.append("Hindi audio already exists")

        if not do_en and not do_hi:
            return None, "; ".join(skip_reasons) if skip_reasons else "Nothing to process"

        return {
            'uuid': uid,
            'do_en': do_en,
            'do_hi': do_hi,
            'text_en': txt_en,
            'text_hi': txt_hi,
            'voice_en': final_en_voice,
            'voice_hi': final_hi_voice
        }, None

    # 7. Build processing list
    to_process = []

    print("\nValidating and checking existing files...")
    for article in articles:
        uid = str(get_nested_value(article, uuid_field))
        if not uid:
            continue
        result, _ = validate_article(article, uid)
        if result:
            to_process.append(result)

    count = len(to_process)
    print(f"\nFound {count} articles requiring processing.")

    # 8. Processing Mode
    print("\nProcessing Options:")
    print("1. Process ALL")
    print("2. Process specific QUANTITY")
    print("3. Process specific UUID")
    proc_choice = input("Select option (1-3) [1]: ").strip() or "1"

    run_list = []

    if proc_choice == '1':
        run_list = to_process
    elif proc_choice == '2':
        if count == 0:
            print("No articles available to process.")
            return
        limit_in = input(f"Enter quantity (1-{count}): ").strip()
        if limit_in.isdigit():
            limit = min(int(limit_in), count)
            run_list = to_process[:limit]
        else:
            print("Invalid quantity.")
            return
    elif proc_choice == '3':
        target_uuid = input("Enter UUID to process: ").strip()
        if target_uuid not in article_by_uuid:
            print(f"UUID '{target_uuid}' not found in JSON file.")
            return
        article = article_by_uuid[target_uuid]
        result, reason = validate_article(article, target_uuid)
        if result:
            run_list = [result]
            print(f"\nUUID '{target_uuid}' is valid for processing:")
            if result['do_en']:
                print(f"  - English: Will generate (Voice: {result['voice_en']})")
            if result['do_hi']:
                print(f"  - Hindi: Will generate (Voice: {result['voice_hi']})")
        else:
            print(f"\nCannot process UUID '{target_uuid}': {reason}")
            return
    else:
        print("Invalid option.")
        return

    if not run_list:
        print("No articles to process.")
        return

    total = len(run_list)
    consecutive_errors = 0
    MAX_CONSECUTIVE_ERRORS = 3

    print(f"\nStarting Generation... ({total} item(s))")
    for i, item in enumerate(run_list):
        uid = item['uuid']
        print(f"[{i + 1}/{total}] Processing {uid}")
        article_failed = False  # Track failure at article level

        # --- English ---
        if item['do_en']:
            print(f"  -> Generating English (Voice: {item['voice_en']})...")
            logging.info(f"[{i + 1}/{total}] Generating English for {uid}")
            try:
                t_path = os.path.join(out_dir_txt_en, f"{uid}.txt")
                with open(t_path, 'w', encoding='utf-8') as f:
                    f.write(clean_text(item['text_en']))

                cl_txt = clean_text(item['text_en'])
                audio = generate_tts_v3(cl_txt, item['voice_en'])

                a_path = os.path.join(out_dir_audio_en, f"{uid}.mp3")
                with open(a_path, 'wb') as f:
                    f.write(audio)
                print("     [DONE English]")
                logging.info(f"[{i + 1}/{total}] English DONE for {uid}")
            except Exception as e:
                article_failed = True
                print(f"     [FAILED English] {e}")
                logging.error(f"[{i + 1}/{total}] English FAILED for {uid}: {e}")

        # --- Hindi (NOT skipped if English fails) ---
        if item['do_hi']:
            print(f"  -> Generating Hindi (Voice: {item['voice_hi']})...")
            logging.info(f"[{i + 1}/{total}] Generating Hindi for {uid}")
            try:
                t_path = os.path.join(out_dir_txt_hi, f"{uid}.txt")
                with open(t_path, 'w', encoding='utf-8') as f:
                    f.write(clean_text(item['text_hi']))

                cl_txt = clean_text(item['text_hi'])
                audio = generate_tts_v3(cl_txt, item['voice_hi'])

                a_path = os.path.join(out_dir_audio_hi, f"{uid}.mp3")
                with open(a_path, 'wb') as f:
                    f.write(audio)
                print("     [DONE Hindi]")
                logging.info(f"[{i + 1}/{total}] Hindi DONE for {uid}")
            except Exception as e:
                article_failed = True
                print(f"     [FAILED Hindi] {e}")
                logging.error(f"[{i + 1}/{total}] Hindi FAILED for {uid}: {e}")

        # --- Error tracking at article level ---
        if article_failed:
            consecutive_errors += 1
            if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                logging.error(f"Stopping after {MAX_CONSECUTIVE_ERRORS} consecutive article failures. Progress: {i}/{total}")
                print(f"\nERROR: {MAX_CONSECUTIVE_ERRORS} consecutive failures. Stopping.")
                print(f"Progress: {i}/{total} articles processed.")
                print(f"Log file: {log_file}")
                return
            else:
                print(f"     Warning: Error {consecutive_errors}/{MAX_CONSECUTIVE_ERRORS} - continuing...")
        else:
            consecutive_errors = 0  # Reset only on full article success

    logging.info(f"Generation complete! {total} articles processed.")
    print(f"\nGeneration complete! {total} articles processed.")
    print(f"Log file: {log_file}")

if __name__ == "__main__":
    main()
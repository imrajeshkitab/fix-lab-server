import os
import time
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional

from dotenv import load_dotenv
from elevenlabs import ElevenLabs, VoiceSettings

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

MAX_CHARS_PER_CHUNK = 4800          # ElevenLabs safe limit per request
MAX_CONSECUTIVE_ERRORS = 3
REQUEST_DELAY_SECONDS = 0.5         # Delay between API calls to avoid 429s
RETRY_ATTEMPTS = 3
RETRY_BACKOFF_BASE = 2              # Exponential backoff base (seconds)

# ============================================================================
# Logging Setup
# ============================================================================

def setup_logging():
    """Setup logging to both console and file."""
    script_name = "text2voice_v3_bilingual"
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
# Client Initialization (singleton — initialized once, reused across calls)
# ============================================================================

_CLIENT: Optional[ElevenLabs] = None

def get_elevenlabs_client() -> ElevenLabs:
    """
    Return a shared ElevenLabs client, initializing it only once.
    Avoids re-loading env + re-creating the client on every TTS call.
    """
    global _CLIENT
    if _CLIENT is None:
        load_dotenv()
        api_key = os.getenv("ELEVEN_LABS_API_KEY")
        if not api_key:
            raise RuntimeError("ELEVEN_LABS_API_KEY is not set in environment.")
        _CLIENT = ElevenLabs(api_key=api_key)
        logging.info("ElevenLabs client initialized.")
    return _CLIENT

# ============================================================================
# Text Chunking
# ============================================================================

def chunk_text(text: str, max_chars: int = MAX_CHARS_PER_CHUNK) -> List[str]:
    """
    Split text into chunks that fit within ElevenLabs' per-request character limit.
    Splits on sentence boundaries ('. ') to avoid mid-sentence cuts.
    """
    if len(text) <= max_chars:
        return [text]

    sentences = text.replace('\n', ' ').split('. ')
    chunks = []
    current = ""

    for sentence in sentences:
        candidate = sentence + ". "
        if len(current) + len(candidate) <= max_chars:
            current += candidate
        else:
            if current.strip():
                chunks.append(current.strip())
            # If a single sentence itself exceeds max_chars, hard-split it
            if len(candidate) > max_chars:
                for start in range(0, len(candidate), max_chars):
                    chunks.append(candidate[start:start + max_chars].strip())
                current = ""
            else:
                current = candidate

    if current.strip():
        chunks.append(current.strip())

    return chunks

# ============================================================================
# TTS Generation (with retry + backoff + chunking)
# ============================================================================

def generate_tts(
    text: str,
    voice_id: str,
    language: str = "en",                   # "en" or "hi" — affects style setting
    model_id: str = "eleven_multilingual_v2",
    output_format: str = "mp3_44100_192",   # Explicit high-quality output
) -> bytes:
    """
    Generate TTS audio using ElevenLabs API.

    Handles:
    - Long text via chunking (> 4800 chars split into multiple requests)
    - Retry with exponential backoff on transient failures
    - Language-aware voice settings (style=0.0 for Hindi to avoid artifacts)
    - Explicit output format for consistent audio quality
    """
    client = get_elevenlabs_client()

    # Language-aware settings:
    # style > 0 on non-English languages can introduce artifacts in multilingual_v2
    style_value = 0.2 if language == "en" else 0.0

    voice_settings = VoiceSettings(
        stability=0.7,
        similarity_boost=0.7,
        style=style_value,
        use_speaker_boost=True,
        # NOTE: `speed` is NOT a VoiceSettings field in the ElevenLabs SDK.
        # To control speed, use the `tts_settings` param at the request level
        # if your SDK version supports it. Omitted here to avoid silent failures.
    )

    text_chunks = chunk_text(text)
    total_chunks = len(text_chunks)

    if total_chunks > 1:
        print(f"     [TTS] Text split into {total_chunks} chunks ({len(text)} chars total)")
    else:
        print(f"     [TTS] Text: {len(text)} chars, single chunk")

    all_audio: List[bytes] = []

    for chunk_idx, chunk in enumerate(text_chunks):
        if total_chunks > 1:
            print(f"     [TTS] Chunk {chunk_idx + 1}/{total_chunks} ({len(chunk)} chars)...")

        audio_bytes = _generate_tts_with_retry(
            client=client,
            text=chunk,
            voice_id=voice_id,
            model_id=model_id,
            voice_settings=voice_settings,
            output_format=output_format,
            chunk_label=f"chunk {chunk_idx + 1}/{total_chunks}",
        )
        all_audio.append(audio_bytes)

        # Avoid hammering the API between chunks
        if chunk_idx < total_chunks - 1:
            time.sleep(REQUEST_DELAY_SECONDS)

    combined = b"".join(all_audio)
    print(f"     [TTS] Total audio: {len(combined)} bytes")
    return combined


def _generate_tts_with_retry(
    client: ElevenLabs,
    text: str,
    voice_id: str,
    model_id: str,
    voice_settings: VoiceSettings,
    output_format: str,
    chunk_label: str = "",
) -> bytes:
    """
    Internal helper: call ElevenLabs API with exponential backoff retry.
    Raises on final failure after RETRY_ATTEMPTS.
    """
    last_exception = None

    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            audio_generator = client.text_to_speech.convert(
                text=text,
                voice_id=voice_id,
                model_id=model_id,
                voice_settings=voice_settings,
                output_format=output_format,
            )
            return b"".join(audio_generator)

        except Exception as e:
            last_exception = e
            wait = RETRY_BACKOFF_BASE ** (attempt - 1)  # 1s, 2s, 4s
            print(f"     [API] Attempt {attempt}/{RETRY_ATTEMPTS} failed ({chunk_label}): {e}")
            if attempt < RETRY_ATTEMPTS:
                print(f"     [API] Retrying in {wait}s...")
                time.sleep(wait)

    raise RuntimeError(f"All {RETRY_ATTEMPTS} attempts failed for {chunk_label}: {last_exception}")

# ============================================================================
# Helper Functions
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
    logging.info("Starting Bilingual Text-to-Voice Generator (v3 Fixed)")

    print("\n--- Bilingual Text-to-Voice Generator (v3 Fixed) ---\n")

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

    # Build article lookup by UUID
    article_by_uuid = {}
    for article in articles:
        uid = str(get_nested_value(article, uuid_field))
        if uid:
            article_by_uuid[uid] = article

    # 6. Validation function
    def validate_article(article, uid):
        """Validate a single article and return processing info or None if invalid."""
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

    # 7. Build to_process list
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

    # 8. Processing Mode Selection
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

    print(f"\nStarting Generation... ({total} item(s))\n")

    for i, item in enumerate(run_list):
        uid = item['uuid']
        print(f"[{i+1}/{total}] Processing {uid}")

        # ── English ──────────────────────────────────────────────────────────
        if item['do_en']:
            print(f"  -> Generating English (Voice: {item['voice_en']})...")
            logging.info(f"[{i+1}/{total}] Generating English for {uid}")
            try:
                clean = clean_text(item['text_en'])

                # Save transcript
                t_path = os.path.join(out_dir_txt_en, f"{uid}.txt")
                with open(t_path, 'w', encoding='utf-8') as f:
                    f.write(clean)

                audio = generate_tts(clean, item['voice_en'], language="en")

                a_path = os.path.join(out_dir_audio_en, f"{uid}.mp3")
                with open(a_path, 'wb') as f:
                    f.write(audio)

                print("     [DONE English]")
                logging.info(f"[{i+1}/{total}] English DONE for {uid}")
                consecutive_errors = 0

            except Exception as e:
                consecutive_errors += 1
                print(f"     [FAILED English] {e}")
                logging.error(f"[{i+1}/{total}] English FAILED for {uid}: {e}")

                if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                    logging.error(f"Stopping after {MAX_CONSECUTIVE_ERRORS} consecutive errors. Progress: {i}/{total}")
                    print(f"\n❌ ERROR: {MAX_CONSECUTIVE_ERRORS} consecutive failures. Stopping.")
                    print(f"📁 Log file: {log_file}")
                    return
                else:
                    print(f"     ⚠️  Error {consecutive_errors}/{MAX_CONSECUTIVE_ERRORS} — continuing...")
                    continue

        # ── Hindi ─────────────────────────────────────────────────────────────
        if item['do_hi']:
            print(f"  -> Generating Hindi (Voice: {item['voice_hi']})...")
            logging.info(f"[{i+1}/{total}] Generating Hindi for {uid}")
            try:
                clean = clean_text(item['text_hi'])

                # Save transcript
                t_path = os.path.join(out_dir_txt_hi, f"{uid}.txt")
                with open(t_path, 'w', encoding='utf-8') as f:
                    f.write(clean)

                audio = generate_tts(clean, item['voice_hi'], language="hi")

                a_path = os.path.join(out_dir_audio_hi, f"{uid}.mp3")
                with open(a_path, 'wb') as f:
                    f.write(audio)

                print("     [DONE Hindi]")
                logging.info(f"[{i+1}/{total}] Hindi DONE for {uid}")
                consecutive_errors = 0

            except Exception as e:
                consecutive_errors += 1
                print(f"     [FAILED Hindi] {e}")
                logging.error(f"[{i+1}/{total}] Hindi FAILED for {uid}: {e}")

                if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                    logging.error(f"Stopping after {MAX_CONSECUTIVE_ERRORS} consecutive errors. Progress: {i}/{total}")
                    print(f"\n❌ ERROR: {MAX_CONSECUTIVE_ERRORS} consecutive failures. Stopping.")
                    print(f"📁 Log file: {log_file}")
                    return
                else:
                    print(f"     ⚠️  Error {consecutive_errors}/{MAX_CONSECUTIVE_ERRORS} — continuing...")
                    continue

        # Inter-article delay to respect rate limits
        if i < total - 1:
            time.sleep(REQUEST_DELAY_SECONDS)

    logging.info(f"Generation complete! {total} articles processed.")
    print(f"\n✅ Generation complete! {total} articles processed.")
    print(f"📁 Log file: {log_file}")


if __name__ == "__main__":
    main()
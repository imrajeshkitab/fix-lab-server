"""
Audio Pipeline — Splicing & Helpers
=====================================
Pure helpers (no DB / no network) for stitching regenerated paragraphs into
the original audio, plus segment naming conventions.

Splicing uses pydub (which uses ffmpeg under the hood — already a dep via
the TTS scripts). We apply small fade in/out at splice boundaries to mask
any micro-glitches at the cut points.

Functions:
  segment_filename()  — construct storage filename for a segment
  splice_audio()      — replace paragraph time ranges with new audio
  audio_duration_sec()— get audio duration in seconds
  audio_duration_str()— "MM:SS" formatted duration
"""

import logging
from io import BytesIO
from typing import List, Optional

logger = logging.getLogger("fix-lab.pipeline")

# ── Storage Naming ────────────────────────────────────────────────────────

SEGMENT_BUCKET = "RMS-content"
SEGMENT_PATH_PREFIX = "bites/audio_segments_triage"


def segment_filename(triage_id: str, paragraph_index: int) -> str:
    """
    Storage filename for a segment.

    paragraph_index >= 0  → '{triage_id}_p{N}.mp3'
    paragraph_index == -1 → '{triage_id}_full.mp3' (decision=full sentinel)
    """
    if paragraph_index < 0:
        return f"{triage_id}_full.mp3"
    return f"{triage_id}_p{paragraph_index}.mp3"


def segment_storage_path(triage_id: str, paragraph_index: int) -> str:
    """Full storage path inside the bucket."""
    return f"{SEGMENT_PATH_PREFIX}/{segment_filename(triage_id, paragraph_index)}"


# ── Audio Duration ────────────────────────────────────────────────────────

def audio_duration_sec(audio_bytes: bytes) -> Optional[float]:
    """Get MP3 duration in seconds via mutagen. Returns None on failure."""
    try:
        from mutagen.mp3 import MP3
        mp3 = MP3(BytesIO(audio_bytes))
        return float(mp3.info.length)
    except Exception as e:
        logger.warning(f"audio_duration_sec failed: {e}")
        return None


def audio_duration_str(audio_bytes: bytes) -> Optional[str]:
    """Get MP3 duration as 'MM:SS' string. Returns None on failure."""
    sec = audio_duration_sec(audio_bytes)
    if sec is None:
        return None
    minutes = int(sec // 60)
    seconds = int(sec % 60)
    return f"{minutes:02d}:{seconds:02d}"


# ── Splicing ──────────────────────────────────────────────────────────────

def splice_audio(
    original_bytes: bytes,
    paragraph_timings: List[dict],
    new_segments: List[dict],
    fade_ms: int = 40,
    bitrate: str = "128k",
) -> bytes:
    """
    Replace specified paragraph time ranges in the original audio with
    new TTS-generated segment audio.

    Strategy:
      1. Sort paragraphs to be replaced by their t_start (timeline order).
      2. Walk forward through the original audio:
         - Append untouched audio up to t_start of next replacement
         - Append new segment audio (with fade in/out for clean splice)
         - Skip over [t_start, t_end] of the original
      3. Append remaining audio after the last replaced paragraph.

    Args:
        original_bytes: Raw MP3 bytes of the original audio.
        paragraph_timings: List of {paragraph_index, t_start, t_end} from triage.
        new_segments: List of {paragraph_index, audio_bytes} — the new audio
                      for each paragraph being replaced.
        fade_ms: Fade in/out duration at splice points (milliseconds). 30-50ms
                 is enough to mask click artifacts without being audible.
        bitrate: Output MP3 bitrate.

    Returns:
        Spliced audio as MP3 bytes.

    Raises:
        Exception: If pydub/ffmpeg unavailable or splice produces no audio.
    """
    from pydub import AudioSegment

    if not new_segments:
        # Nothing to replace — return original unchanged
        return original_bytes

    original = AudioSegment.from_file(BytesIO(original_bytes), format="mp3")
    total_ms = len(original)

    # Map paragraph_index → new audio bytes (skip sentinel -1)
    new_by_idx = {
        s["paragraph_index"]: s["audio_bytes"]
        for s in new_segments
        if s.get("paragraph_index", -1) >= 0
    }
    if not new_by_idx:
        return original_bytes

    # Filter timings to only those we're actually replacing, sort by t_start
    replacements = [
        t for t in (paragraph_timings or [])
        if t.get("paragraph_index") in new_by_idx
    ]
    replacements.sort(key=lambda t: float(t.get("t_start", 0)))

    if not replacements:
        logger.warning(
            "splice_audio: have new segments but no matching paragraph_timings — "
            "returning original unchanged"
        )
        return original_bytes

    result = AudioSegment.empty()
    cursor_ms = 0

    for timing in replacements:
        t_start_ms = max(0, int(float(timing["t_start"]) * 1000))
        t_end_ms = max(t_start_ms, int(float(timing["t_end"]) * 1000))
        # Clamp to audio length
        t_start_ms = min(t_start_ms, total_ms)
        t_end_ms = min(t_end_ms, total_ms)

        # Append original audio up to this segment's start
        if t_start_ms > cursor_ms:
            head = original[cursor_ms:t_start_ms]
            # Tiny fade-out at the cut so the splice is clean
            head = head.fade_out(min(fade_ms, len(head)))
            result += head

        # Append new TTS segment with fade in/out
        new_bytes = new_by_idx[timing["paragraph_index"]]
        new_seg = AudioSegment.from_file(BytesIO(new_bytes), format="mp3")
        new_seg = new_seg.fade_in(min(fade_ms, len(new_seg))).fade_out(min(fade_ms, len(new_seg)))
        result += new_seg

        cursor_ms = t_end_ms

    # Append tail (everything after the last replaced segment)
    if cursor_ms < total_ms:
        tail = original[cursor_ms:total_ms]
        tail = tail.fade_in(min(fade_ms, len(tail)))
        result += tail

    if len(result) == 0:
        raise Exception("splice_audio produced empty result")

    out = BytesIO()
    result.export(out, format="mp3", bitrate=bitrate)
    spliced = out.getvalue()

    logger.info(
        f"splice_audio: original={total_ms}ms, replaced={len(replacements)} segments, "
        f"final={len(result)}ms, size={len(spliced)} bytes"
    )

    return spliced

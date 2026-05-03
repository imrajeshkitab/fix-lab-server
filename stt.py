"""
STT Module — Local Whisper for Paragraph Alignment
====================================================
Uses faster-whisper to transcribe audio and align paragraphs to timestamps.

This module is a PRE-PROCESSING step only:
  - Whisper transcript is used to determine paragraph start/end times
  - Transcript is NOT sent to Gemini (no decision-making value)
  - Feedback timestamps are mapped to paragraphs using simple range checks

Functions:
  transcribe_audio()          — Run Whisper, get timestamped segments
  align_paragraphs()          — Match segments to source paragraphs
  map_feedback_to_paragraphs() — Map feedback items to paragraphs by timestamp
"""

import json
import logging
import os
import tempfile
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger("fix-lab.stt")


# ── Whisper Transcription ─────────────────────────────────────────────────

def transcribe_audio(
    audio_bytes: bytes,
    language: str,
) -> List[dict]:
    """
    Transcribe audio using faster-whisper.

    Args:
        audio_bytes: Raw audio file bytes (MP3)
        language: "en" or "hi"

    Returns:
        List of segments: [{"text": str, "start": float, "end": float}]
    """
    from faster_whisper import WhisperModel

    model_size = os.getenv("WHISPER_MODEL", "base")
    device = os.getenv("WHISPER_DEVICE", "cpu")
    compute_type = os.getenv("WHISPER_COMPUTE_TYPE", "int8")

    logger.info(
        f"STT: Loading Whisper model={model_size}, device={device}, "
        f"compute_type={compute_type}, language={language}"
    )

    model = WhisperModel(model_size, device=device, compute_type=compute_type)

    # Write audio to temp file (faster-whisper needs a file path)
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        segments_gen, info = model.transcribe(
            tmp_path,
            language=language,
            word_timestamps=False,  # Segment-level is sufficient for paragraph alignment
            vad_filter=True,        # Filter out silence
        )

        segments = []
        for seg in segments_gen:
            segments.append({
                "text": seg.text.strip(),
                "start": round(seg.start, 2),
                "end": round(seg.end, 2),
            })

        logger.info(
            f"STT: Transcribed {len(segments)} segments, "
            f"detected_language={info.language}, "
            f"language_probability={info.language_probability:.2f}, "
            f"duration={info.duration:.1f}s"
        )

        return segments

    finally:
        os.unlink(tmp_path)


# ── Paragraph Alignment ──────────────────────────────────────────────────

def _normalize_text(text: str) -> str:
    """Normalize text for comparison: lowercase, strip extra whitespace."""
    return " ".join(text.lower().split())


def _text_similarity(a: str, b: str) -> float:
    """Calculate text similarity ratio between two strings."""
    return SequenceMatcher(None, _normalize_text(a), _normalize_text(b)).ratio()


def align_paragraphs(
    segments: List[dict],
    source_paragraphs: List[str],
) -> List[dict]:
    """
    Align Whisper transcript segments to source paragraphs.

    Strategy: Walk through segments sequentially, accumulating text until
    we find the best match for the current paragraph, then move to the next.
    This works because both the audio and source text follow the same order.

    Args:
        segments: Whisper output [{"text", "start", "end"}]
        source_paragraphs: Source text split into paragraphs

    Returns:
        [{"paragraph_index": int, "t_start": float, "t_end": float}]
    """
    if not segments or not source_paragraphs:
        logger.warning("STT: No segments or paragraphs to align")
        return []

    timings = []
    seg_idx = 0
    total_segs = len(segments)

    for para_idx, para_text in enumerate(source_paragraphs):
        if seg_idx >= total_segs:
            # No more segments — remaining paragraphs get no timing
            logger.warning(
                f"STT: Ran out of segments at paragraph {para_idx}/{len(source_paragraphs)}"
            )
            break

        # Start of this paragraph = start of current segment
        t_start = segments[seg_idx]["start"]
        accumulated_text = ""
        best_similarity = 0.0
        best_end_idx = seg_idx

        # Accumulate segments until we find the best match for this paragraph
        for i in range(seg_idx, total_segs):
            accumulated_text += " " + segments[i]["text"]
            accumulated_text = accumulated_text.strip()

            sim = _text_similarity(accumulated_text, para_text)

            if sim >= best_similarity:
                best_similarity = sim
                best_end_idx = i

            # If similarity starts dropping significantly, we've passed the paragraph
            if sim < best_similarity - 0.15 and best_similarity > 0.3:
                break

            # If we have a very strong match, stop looking
            if sim > 0.85:
                best_end_idx = i
                break

        t_end = segments[best_end_idx]["end"]

        timings.append({
            "paragraph_index": para_idx,
            "t_start": round(t_start, 2),
            "t_end": round(t_end, 2),
        })

        logger.debug(
            f"STT: P{para_idx} aligned to {t_start:.1f}s-{t_end:.1f}s "
            f"(similarity={best_similarity:.2f})"
        )

        # Move past the segments we just consumed
        seg_idx = best_end_idx + 1

    logger.info(
        f"STT: Aligned {len(timings)}/{len(source_paragraphs)} paragraphs"
    )

    return timings


# ── Feedback Mapping ─────────────────────────────────────────────────────

def map_feedback_to_paragraphs(
    feedback_items: List[dict],
    paragraph_timings: List[dict],
) -> Tuple[List[dict], List[dict], Set[int]]:
    """
    Map feedback items to paragraphs using timestamp range checks.

    Feedback with a timestamp is matched to the paragraph whose time range
    contains that timestamp. Feedback without timestamps goes to unmapped.

    Args:
        feedback_items: Reviewer feedback [{type, content, timestamp, language, ...}]
        paragraph_timings: [{paragraph_index, t_start, t_end}]

    Returns:
        (mapped_feedback, unmapped_feedback, affected_paragraph_indices)

        mapped_feedback: items with "paragraph_index" attached
        unmapped_feedback: items without timestamps (general feedback)
        affected_paragraph_indices: set of paragraph indices that have issues
    """
    mapped = []
    unmapped = []
    affected = set()

    for item in feedback_items:
        ts = item.get("timestamp")

        if ts is not None:
            try:
                ts = float(ts)
            except (ValueError, TypeError):
                # Invalid timestamp — treat as unmapped
                unmapped.append(item)
                continue

            # Find which paragraph this timestamp falls in
            matched_para = None
            for timing in paragraph_timings:
                if timing["t_start"] <= ts <= timing["t_end"]:
                    matched_para = timing["paragraph_index"]
                    break

            if matched_para is None:
                # Timestamp is outside all paragraph ranges — find closest
                if paragraph_timings:
                    closest = min(
                        paragraph_timings,
                        key=lambda t: min(abs(t["t_start"] - ts), abs(t["t_end"] - ts))
                    )
                    matched_para = closest["paragraph_index"]
                    logger.debug(
                        f"STT: Feedback @{ts:.1f}s outside ranges, "
                        f"mapped to closest P{matched_para}"
                    )

            if matched_para is not None:
                mapped_item = {**item, "paragraph_index": matched_para}
                mapped.append(mapped_item)
                affected.add(matched_para)
            else:
                unmapped.append(item)
        else:
            # No timestamp — general feedback
            unmapped.append(item)

    logger.info(
        f"STT: Mapped {len(mapped)} feedback items to paragraphs, "
        f"{len(unmapped)} unmapped (general), "
        f"{len(affected)} affected paragraphs: {sorted(affected)}"
    )

    return mapped, unmapped, affected

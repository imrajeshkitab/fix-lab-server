"""
STT Module — Local Whisper for Paragraph Alignment
====================================================
Uses faster-whisper to transcribe audio and align paragraphs to timestamps.

This module is a PRE-PROCESSING step only:
  - Whisper transcript is used to determine paragraph start/end times
  - Transcript is NOT sent to Gemini (no decision-making value)
  - Feedback timestamps are mapped to paragraphs using simple range checks

Functions:
  transcribe_audio()          — Run Whisper, get segment-level timestamps (EN path)
  align_paragraphs()          — Char-similarity alignment (EN path, works fine)
  transcribe_audio_words()    — Run Whisper with word_timestamps=True (HI path)
  align_paragraphs_hi()       — Word-anchor alignment for Hindi
  map_feedback_to_paragraphs() — Map feedback items to paragraphs by timestamp

Language dispatch:
  - EN → transcribe_audio + align_paragraphs (original logic, unchanged)
  - HI → transcribe_audio_words + align_paragraphs_hi (word-anchor approach)
"""

import json
import logging
import os
import re
import tempfile
import unicodedata
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


# ════════════════════════════════════════════════════════════════════════════
# HINDI PATH — Word-anchor alignment
# ════════════════════════════════════════════════════════════════════════════
# The English `align_paragraphs` uses SequenceMatcher.ratio() which performs
# poorly on Devanagari (Hindi script). For Hindi we use a different strategy:
#
#   1. Run Whisper with word_timestamps=True
#   2. For each source paragraph, pick 2-3 distinctive (non-stopword) anchor
#      words from its beginning
#   3. Find those anchor words IN ORDER in the Whisper word stream
#   4. The first matched word's `start` time = paragraph's t_start
#   5. Paragraphs without a found anchor → fill via char-proportional
#      interpolation between known anchors
#   6. t_end of each para = t_start of next (or audio_duration for last)
#
# This is robust to Whisper transcription errors because we only need ONE
# anchor word to land — the rest of the paragraph doesn't need to match.
# ════════════════════════════════════════════════════════════════════════════

# Hindi stopwords to skip when picking anchors (these repeat across paragraphs
# and would cause false matches).
HI_STOPWORDS = {
    "है", "हैं", "था", "थी", "थे", "और", "के", "का", "की", "में", "से",
    "कि", "यह", "वह", "वे", "इस", "उस", "को", "ने", "पर", "एक", "हो",
    "जो", "जा", "ही", "भी", "तो", "ya", "हम", "तुम", "आप", "मैं", "मेरा",
    "तेरा", "अपना", "बहुत", "कुछ", "नहीं", "हाँ", "ना", "रहा", "रही", "रहे",
}

# English stopwords (for safety, in case anchor picker is reused)
EN_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "of", "to", "in", "on",
    "at", "and", "or", "but", "so", "that", "this", "it", "be", "have",
    "has", "had", "for", "with", "as", "by", "we", "i", "you", "he", "she",
    "they", "his", "her", "their", "our", "my", "your",
}


def _normalize_word(word: str) -> str:
    """Strip punctuation/whitespace and Unicode-normalize for word matching."""
    if not word:
        return ""
    # Strip punctuation including Devanagari danda + common ASCII punctuation
    w = word.strip("।.,!?;:'\"()[]{}—–-…॥ ")
    w = w.lower()
    w = unicodedata.normalize("NFC", w)
    return w


def _tokenize(text: str) -> List[str]:
    """Tokenize text into normalized words. Handles Devanagari + Latin scripts."""
    # Split on whitespace + punctuation (works for both scripts)
    raw = re.split(r"[\s।.,!?;:'\"()\[\]{}—–\-…॥]+", text)
    return [_normalize_word(w) for w in raw if _normalize_word(w)]


def _pick_anchor_words(
    para_text: str,
    stopwords: Set[str],
    n: int = 3,
    min_len: int = 2,
) -> List[str]:
    """
    Pick the first N distinctive (non-stopword, non-trivial) words from a
    paragraph for use as alignment anchors.
    """
    tokens = _tokenize(para_text)
    anchors = []
    for tok in tokens:
        if tok in stopwords:
            continue
        if len(tok) < min_len:
            continue
        anchors.append(tok)
        if len(anchors) >= n:
            break
    # Fallback: if everything was a stopword, just take first non-empty tokens
    if not anchors:
        anchors = [t for t in tokens if t][:n]
    return anchors


def transcribe_audio_words(
    audio_bytes: bytes,
    language: str,
) -> Tuple[List[dict], float]:
    """
    Transcribe audio with WORD-level timestamps.

    Returns:
        (words, duration_seconds)
        words: list of {"word": str, "start": float, "end": float}
    """
    from faster_whisper import WhisperModel

    model_size = os.getenv("WHISPER_MODEL", "base")
    device = os.getenv("WHISPER_DEVICE", "cpu")
    compute_type = os.getenv("WHISPER_COMPUTE_TYPE", "int8")

    logger.info(
        f"STT (HI): Loading Whisper model={model_size}, device={device}, "
        f"compute_type={compute_type}, language={language} (word-level)"
    )

    model = WhisperModel(model_size, device=device, compute_type=compute_type)

    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        segments_gen, info = model.transcribe(
            tmp_path,
            language=language,
            word_timestamps=True,   # ← key difference: per-word times
            vad_filter=True,
        )

        words: List[dict] = []
        for seg in segments_gen:
            if not seg.words:
                continue
            for w in seg.words:
                words.append({
                    "word": w.word.strip() if w.word else "",
                    "start": round(w.start, 3) if w.start is not None else 0.0,
                    "end": round(w.end, 3) if w.end is not None else 0.0,
                })

        logger.info(
            f"STT (HI): Got {len(words)} word-tokens, "
            f"detected_language={info.language}, "
            f"language_probability={info.language_probability:.2f}, "
            f"duration={info.duration:.1f}s"
        )

        return words, float(info.duration or 0.0)

    finally:
        os.unlink(tmp_path)


def _find_anchor_index(
    anchors: List[str],
    words: List[dict],
    start_idx: int,
    search_window: int = 400,
) -> Optional[int]:
    """
    Find the start index of an ORDERED occurrence of `anchors` in `words`,
    starting search from `start_idx`. Returns the index where the first
    anchor word matches, or None if no match within the window.

    Uses fuzzy equality (edit distance ≤ 1 for words ≥ 4 chars, exact otherwise).
    """
    if not anchors or start_idx >= len(words):
        return None

    end_search = min(len(words), start_idx + search_window)

    def _fuzzy_eq(a: str, b: str) -> bool:
        if a == b:
            return True
        if len(a) < 4 or len(b) < 4:
            return False
        # Allow 1-char diff for longer words (handles minor Whisper errors)
        if abs(len(a) - len(b)) > 1:
            return False
        return SequenceMatcher(None, a, b).ratio() >= 0.85

    for i in range(start_idx, end_search):
        candidate = _normalize_word(words[i].get("word", ""))
        if not _fuzzy_eq(candidate, anchors[0]):
            continue
        # First anchor matched — verify subsequent anchors appear in order
        # within a small look-ahead window (allow up to 8 intervening words)
        cursor = i + 1
        all_matched = True
        for next_anchor in anchors[1:]:
            found_at = None
            for j in range(cursor, min(cursor + 8, len(words))):
                if _fuzzy_eq(_normalize_word(words[j].get("word", "")), next_anchor):
                    found_at = j
                    break
            if found_at is None:
                all_matched = False
                break
            cursor = found_at + 1
        if all_matched:
            return i

    return None


def align_paragraphs_hi(
    words: List[dict],
    source_paragraphs: List[str],
    audio_duration: float,
) -> List[dict]:
    """
    Word-anchor based paragraph alignment for Hindi.

    Strategy:
      1. For each paragraph, pick 2-3 distinctive anchor words
      2. Search the Whisper word stream (forward-only cursor) for those
         anchors in order
      3. If found → t_start = first anchor word's start time
      4. If not found → leave None for now, fill later by interpolation
      5. Char-proportional interpolation fills missing t_starts
      6. t_end of each para = t_start of next para (or audio_duration)

    Args:
        words: Whisper word-level timestamps from transcribe_audio_words()
        source_paragraphs: List of paragraph strings
        audio_duration: Total audio duration in seconds (from Whisper info)

    Returns:
        [{"paragraph_index": int, "t_start": float, "t_end": float}]
    """
    if not source_paragraphs:
        return []

    if audio_duration <= 0:
        logger.warning("STT (HI): No audio duration, cannot align")
        return []

    # ── 1. Anchor each paragraph ──
    word_cursor = 0
    para_starts: List[Optional[float]] = []  # one per paragraph
    matched_count = 0

    for p_idx, para in enumerate(source_paragraphs):
        anchor_tokens = _pick_anchor_words(para, HI_STOPWORDS, n=3)
        if not anchor_tokens:
            para_starts.append(None)
            continue

        match_idx = _find_anchor_index(anchor_tokens, words, word_cursor)

        if match_idx is not None:
            t_start = words[match_idx].get("start", 0.0)
            para_starts.append(round(float(t_start), 2))
            # Advance cursor past this anchor (rough estimate)
            word_cursor = match_idx + max(5, len(anchor_tokens))
            matched_count += 1
            logger.debug(
                f"STT (HI): P{p_idx} anchored at word#{match_idx} "
                f"({t_start:.2f}s) via {anchor_tokens}"
            )
        else:
            para_starts.append(None)
            logger.debug(
                f"STT (HI): P{p_idx} NO anchor found for {anchor_tokens}"
            )

    # ── 2. Char-proportional fallback for unmatched paragraphs ──
    char_counts = [len(p) for p in source_paragraphs]
    total_chars = sum(char_counts) or 1
    cumulative_chars = [0]
    for c in char_counts:
        cumulative_chars.append(cumulative_chars[-1] + c)
    # Proportional t_start estimate per paragraph
    prop_starts = [
        (cumulative_chars[i] / total_chars) * audio_duration
        for i in range(len(source_paragraphs))
    ]

    # Fill None entries with proportional estimates, but clamp to be ≥ previous
    # known t_start (preserves monotonicity)
    for i in range(len(para_starts)):
        if para_starts[i] is None:
            est = prop_starts[i]
            # Ensure monotonic: at least equal to previous t_start
            if i > 0 and para_starts[i - 1] is not None:
                est = max(est, para_starts[i - 1] + 0.1)
            para_starts[i] = round(est, 2)

    # Defensive: enforce monotonic non-decreasing
    for i in range(1, len(para_starts)):
        if para_starts[i] < para_starts[i - 1]:
            para_starts[i] = para_starts[i - 1] + 0.1

    # ── 3. Build timings: t_end = next para's t_start (or audio_duration) ──
    timings = []
    for i, t_start in enumerate(para_starts):
        if i + 1 < len(para_starts):
            t_end = para_starts[i + 1]
        else:
            t_end = round(audio_duration, 2)
        timings.append({
            "paragraph_index": i,
            "t_start": t_start,
            "t_end": t_end,
        })

    logger.info(
        f"STT (HI): Aligned {len(timings)} paragraphs "
        f"({matched_count} anchored, {len(timings) - matched_count} interpolated)"
    )

    return timings

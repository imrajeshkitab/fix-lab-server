"""
VO Triage Agent — Phase 1 (Text-Only)
=======================================
Gemini text-only decision making for VO quality issues.

Architecture:
  1. Whisper STT (in stt.py) handles paragraph alignment — no audio sent to Gemini
  2. Code logic (in stt.py) maps timestamped feedback to paragraphs
  3. This module sends ONLY text to Gemini for decision making:
     - Affected paragraphs with inlined feedback
     - General feedback in a separate section
     - Gemini decides: partial / full / skip / escalate

Design rationale:
  - No audio bytes to Gemini → dramatically cheaper (no audio tokens)
  - Only affected paragraphs sent → minimal token usage
  - Full feedback thread for context → holistic decisions
  - Whisper transcript NOT sent → no decision-making value

Usage:
  result, usage = await run_triage_decision(
      paragraphs, affected_indices, mapped_feedback, unmapped_feedback, "en"
  )
"""

import json
import logging
import os
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger("fix-lab.triage")


# ── Text Formatting ───────────────────────────────────────────────────────

def md_to_plain_text(text: str) -> str:
    """
    Convert markdown-formatted text to plain text.

    Uses the `markdown` library to parse all MD syntax correctly,
    then strips HTML tags to get clean plain text.
    Also handles JSON encoding artifacts (\\n, \\t, etc.).
    """
    import html
    import re
    import markdown

    # Handle JSON-encoded strings (e.g., \\n → \n, \\" → ")
    if text.startswith('"') and text.endswith('"'):
        try:
            text = json.loads(text)
        except (json.JSONDecodeError, ValueError):
            pass
    # Replace literal \n and \t escapes from JSON encoding
    text = text.replace('\\n', '\n').replace('\\t', '\t').replace('\\"', '"')

    # Convert markdown → HTML → plain text
    html_content = markdown.markdown(text)
    plain = re.sub(r'<[^>]+>', '', html_content)
    plain = html.unescape(plain)

    # Clean up extra whitespace
    plain = re.sub(r'\n{3,}', '\n\n', plain)
    return plain.strip()


# ── Paragraph Splitting ───────────────────────────────────────────────────

def split_into_paragraphs(text: str) -> List[str]:
    """
    Split text into paragraphs, then strip markdown from each one.

    Order matters: split FIRST (to preserve paragraph boundaries),
    THEN strip markdown per paragraph (since markdown→HTML conversion
    would collapse paragraph breaks).
    """
    # Handle JSON encoding artifacts before splitting
    if text.startswith('"') and text.endswith('"'):
        try:
            text = json.loads(text)
        except (json.JSONDecodeError, ValueError):
            pass
    text = text.replace('\\n', '\n').replace('\\t', '\t').replace('\\"', '"')

    text = text.replace('\r\n', '\n').replace('\r', '\n')

    # Split on double-newline first (preserves paragraph structure)
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    # Fallback: single newline
    if len(paragraphs) <= 1 and '\n' in text:
        paragraphs = [p.strip() for p in text.split('\n') if p.strip()]

    # Strip markdown from each paragraph individually
    paragraphs = [md_to_plain_text(p) for p in paragraphs]

    # Filter out empty paragraphs and pure formatting artifacts (e.g., "***")
    paragraphs = [p for p in paragraphs if p and len(p) > 3]

    return paragraphs


# ── Prompt Construction ───────────────────────────────────────────────────

TRIAGE_SYSTEM_PROMPT = """You are an AI quality assurance agent for Kitab, a content platform that produces voice-over narrations from text scripts using TTS.

Your role: Analyze reviewer feedback about a voice-over audio file and determine the optimal fix strategy.

CRITICAL RULES:
1. The source TEXT is always correct — issues are ONLY in the audio rendering
2. Do NOT suggest text changes — only audio regeneration strategies
3. Human reviewer feedback is the source of truth for what's wrong
4. Be conservative: when unsure, recommend "escalate"
5. Categories of VO issues: pronunciation, missing_words, pacing, intonation, audio_glitch, wrong_emphasis, background_noise
6. Non-VO issues (text_issue, not_applicable) should NOT trigger regeneration
7. Consider the FULL feedback thread when deciding — look for patterns across comments"""


def build_triage_prompt(
    paragraphs: List[str],
    affected_indices: Set[int],
    mapped_feedback: List[dict],
    unmapped_feedback: List[dict],
    language: str,
    title: str = "",
) -> str:
    """
    Build the text-only prompt for Gemini triage decision.

    Only affected paragraphs (with their feedback inlined) are included.
    General feedback goes in a separate section.
    """

    total_paras = len(paragraphs)
    num_affected = len(affected_indices)

    # ── Header ──
    header = f"BITE: \"{title}\" | Language: {language.upper()} | {total_paras} paragraphs total | {num_affected} with reported issues"

    # ── Affected paragraphs with inlined feedback ──
    affected_section_parts = []

    # Group mapped feedback by paragraph
    feedback_by_para: Dict[int, List[dict]] = {}
    for fb in mapped_feedback:
        p_idx = fb["paragraph_index"]
        feedback_by_para.setdefault(p_idx, []).append(fb)

    for p_idx in sorted(affected_indices):
        if p_idx >= len(paragraphs):
            continue

        para_text = paragraphs[p_idx]
        # Truncate very long paragraphs
        display = para_text if len(para_text) <= 500 else para_text[:500] + "..."

        para_block = f"[P{p_idx}] {display}"

        # Inline the feedback for this paragraph
        para_feedbacks = feedback_by_para.get(p_idx, [])
        for fb in para_feedbacks:
            ts = fb.get("timestamp")
            ts_str = f"@{float(ts):.1f}s " if ts is not None else ""
            content = fb.get("content") or fb.get("text") or "(no text)"
            para_block += f"\n  → {ts_str}\"{content}\""

        affected_section_parts.append(para_block)

    affected_section = "\n\n".join(affected_section_parts) if affected_section_parts else "(No paragraphs with timestamped feedback)"

    # ── General feedback (unmapped) ──
    if unmapped_feedback:
        general_lines = []
        for fb in unmapped_feedback:
            content = fb.get("content") or fb.get("text") or "(no text)"
            general_lines.append(f"  → \"{content}\"")
        general_section = "\n".join(general_lines)
    else:
        general_section = "  (None)"

    return f"""{header}

══════════════════════════════════════════
AFFECTED PARAGRAPHS (with reviewer feedback):
══════════════════════════════════════════

{affected_section}

══════════════════════════════════════════
GENERAL FEEDBACK (not tied to a specific location):
══════════════════════════════════════════

{general_section}

══════════════════════════════════════════
TASK
══════════════════════════════════════════

Analyze the feedback above and return a single JSON object:

{{
    "feedback_classification": [
        {{
            "feedback_index": 0,
            "is_actionable": true,
            "category": "<one of: pronunciation, missing_words, pacing, intonation, audio_glitch, wrong_emphasis, background_noise, text_issue, not_applicable, other>",
            "mapped_paragraph": 2,
            "summary": "One-line description of the issue"
        }}
    ],
    "decision": "<one of: partial, full, skip, escalate>",
    "confidence": 0.85,
    "reasoning": "2-3 sentence explanation of your decision",
    "segments_to_regen": [
        {{
            "paragraph_index": 2,
            "issue": "Description of what is wrong with this paragraph's audio",
            "tts_hints": {{
                "pronunciation": "specific guidance if applicable (e.g., 'Ayodhya → ah-YODH-ya')",
                "emphasis": "words that need emphasis",
                "rate": "slower|normal|faster"
            }}
        }}
    ]
}}

DECISION GUIDE:
- "skip"     → No actionable VO issues. Feedback is about text content, not audio quality.
- "full"     → Issues are widespread (affecting many paragraphs) or systemic (e.g., voice tone is consistently bad). Cheaper to regenerate everything.
- "partial"  → Only specific paragraphs need regeneration. Include them in segments_to_regen.
- "escalate" → Issues too complex for automated fix (e.g., wrong voice, needs human intervention).

For "partial": include ONLY affected paragraphs in segments_to_regen.
For "full": include ALL {total_paras} paragraphs in segments_to_regen (with issue descriptions).
For "skip" or "escalate": segments_to_regen should be an empty array.

IMPORTANT: Consider the overall pattern of feedback. If similar issues (e.g., "voice is draggy") appear across many paragraphs plus a general comment confirming it, that's likely a "full" regeneration case, not "partial"."""


# ── Triage Decision Call ──────────────────────────────────────────────────

async def run_triage_decision(
    paragraphs: List[str],
    affected_indices: Set[int],
    mapped_feedback: List[dict],
    unmapped_feedback: List[dict],
    language: str,
    title: str = "",
    model: str = None,
) -> Tuple[dict, dict]:
    """
    Run triage decision via a text-only Gemini call.

    No audio is sent to Gemini. The model receives only:
    - Affected paragraphs with inlined feedback
    - General feedback in a separate section
    - Instructions to decide: partial / full / skip / escalate

    Args:
        paragraphs: All source text paragraphs
        affected_indices: Set of paragraph indices with mapped feedback
        mapped_feedback: Feedback items with paragraph_index attached
        unmapped_feedback: General feedback items (no timestamp)
        language: "en" or "hi"
        title: Bite title for context
        model: Optional Gemini model name override

    Returns:
        (triage_result_dict, usage_dict)

    Raises:
        RuntimeError: If GEMINI_API_KEY is not set
        json.JSONDecodeError: If model returns invalid JSON
        Exception: On API errors
    """
    from google import genai
    from google.genai import types

    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY or GOOGLE_API_KEY not set in environment. "
            "Triage requires a Google AI API key."
        )

    client = genai.Client(api_key=api_key)
    model_name = model or os.getenv("GEMINI_TRIAGE_MODEL", "gemini-2.0-flash")

    prompt = build_triage_prompt(
        paragraphs, affected_indices, mapped_feedback,
        unmapped_feedback, language, title,
    )

    total_feedback = len(mapped_feedback) + len(unmapped_feedback)
    logger.info(
        f"Running triage decision: {len(paragraphs)} total paragraphs, "
        f"{len(affected_indices)} affected, "
        f"{total_feedback} feedback items, model={model_name}"
    )

    response = await client.aio.models.generate_content(
        model=model_name,
        contents=[
            types.Content(
                parts=[types.Part.from_text(text=prompt)],
                role="user",
            )
        ],
        config=types.GenerateContentConfig(
            system_instruction=TRIAGE_SYSTEM_PROMPT,
            temperature=0.15,  # Low temp for precise, deterministic output
            response_mime_type="application/json",
        ),
    )

    # Parse structured response
    result = json.loads(response.text)

    # ── Post-process: enrich segments with paragraph text ──
    for seg in result.get("segments_to_regen", []):
        idx = seg.get("paragraph_index", -1)
        if 0 <= idx < len(paragraphs):
            seg["paragraph_text"] = paragraphs[idx]

    # ── Extract token usage for cost tracking ──
    usage = {
        "input_tokens": 0,
        "output_tokens": 0,
        "model": model_name,
    }
    if hasattr(response, "usage_metadata") and response.usage_metadata:
        um = response.usage_metadata
        usage["input_tokens"] = getattr(um, "prompt_token_count", 0) or 0
        usage["output_tokens"] = getattr(um, "candidates_token_count", 0) or 0

    logger.info(
        f"Triage decision: decision={result.get('decision')}, "
        f"confidence={result.get('confidence')}, "
        f"segments={len(result.get('segments_to_regen', []))}, "
        f"tokens={usage['input_tokens']}in+{usage['output_tokens']}out"
    )

    return result, usage

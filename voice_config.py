"""
Voice configuration for Fix Lab TTS generation.
Maps vo_artist names to ElevenLabs voice IDs.
Ported from TTS-scripts/config.py

NOTE: Update these IDs to match your ElevenLabs account.

      - To assign a real voice, replace the placeholder string with the
        actual ElevenLabs voice ID for that artist.
      - FALLBACK_VOICE is only used when you explicitly assign it below.
      - If an artist is NOT listed here at all, get_voice_id() returns None
        and the job item will be SKIPPED (not silently regenerated with wrong voice).
"""

# Pre-made fallback voice (Adam) — only used when explicitly assigned in the maps below.
FALLBACK_VOICE = "pNInz6obpgDQGcFmaJgB"

VOICE_IDS_ENGLISH = {
    "Anchor-V1": FALLBACK_VOICE,    # TODO: replace with real voice ID
    "Coach - V1": FALLBACK_VOICE,   # TODO: replace with real voice ID
    "Maya J": FALLBACK_VOICE,       # TODO: replace with real voice ID
    "Monika Sogam": FALLBACK_VOICE, # TODO: replace with real voice ID
    "Rudra": FALLBACK_VOICE,        # TODO: replace with real voice ID
}

VOICE_IDS_HINDI = {
    "Anchor-V1": FALLBACK_VOICE,    # TODO: replace with real voice ID
    "Coach - V1": FALLBACK_VOICE,   # TODO: replace with real voice ID
    "Maya J": FALLBACK_VOICE,       # TODO: replace with real voice ID
    "Monika Sogam": FALLBACK_VOICE, # TODO: replace with real voice ID
    "Rudra": FALLBACK_VOICE,        # TODO: replace with real voice ID
    "Sandeep_das_v1": FALLBACK_VOICE, # TODO: replace with real voice ID
}


def get_voice_id(vo_artist: str, language: str) -> str | None:
    """
    Look up the ElevenLabs voice ID for a given vo_artist and language.

    Returns:
        str  — voice ID if the artist is found in the map.
        None — if the artist is completely unknown (not listed).
                The caller MUST handle None by skipping the item,
                rather than silently using a fallback voice.
    """
    voice_map = VOICE_IDS_HINDI if language == "hi" else VOICE_IDS_ENGLISH

    # Exact match
    if vo_artist and vo_artist in voice_map:
        return voice_map[vo_artist]

    # Case-insensitive fallback match
    for name, vid in voice_map.items():
        if name.lower() == (vo_artist or "").lower():
            return vid

    # Artist not configured — caller should skip the item
    return None

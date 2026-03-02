"""
Voice configuration for Fix Lab TTS generation.
Maps vo_artist names to ElevenLabs voice IDs.
Ported from TTS-scripts/config.py

NOTE: Update these IDs to match your ElevenLabs account.
      If a voice is not found, the fallback voice is used.
"""

# Universal fallback voice ID (Adam — default pre-made voice, works on free tier)
FALLBACK_VOICE = "pNInz6obpgDQGcFmaJgB"

VOICE_IDS_ENGLISH = {
    "Anchor-V1": FALLBACK_VOICE,
    "Coach - V1": FALLBACK_VOICE,
    "Maya J": FALLBACK_VOICE,
    "Monika Sogam": FALLBACK_VOICE,
    "Rudra": FALLBACK_VOICE,
}

VOICE_IDS_HINDI = {
    "Anchor-V1": FALLBACK_VOICE,
    "Coach - V1": FALLBACK_VOICE,
    "Maya J": FALLBACK_VOICE,
    "Monika Sogam": FALLBACK_VOICE,
    "Rudra": FALLBACK_VOICE,
    "Sandeep_das_v1": FALLBACK_VOICE,
}


def get_voice_id(vo_artist: str, language: str) -> str:
    """Look up voice ID for a given vo_artist and language.
    Falls back to FALLBACK_VOICE if not found."""
    voice_map = VOICE_IDS_HINDI if language == "hi" else VOICE_IDS_ENGLISH

    if vo_artist and vo_artist in voice_map:
        return voice_map[vo_artist]

    # Try case-insensitive match
    for name, vid in voice_map.items():
        if name.lower() == (vo_artist or "").lower():
            return vid

    return FALLBACK_VOICE

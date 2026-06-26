"""
TTS Engine Wrapper
==================
Handles importing the TTS scripts (which have dots in their filenames)
and provides a clean interface for audio generation.
"""

import os
import sys
import importlib.util
import logging

logger = logging.getLogger("fix-lab.tts")

# Path to TTS scripts
TTS_DIR = os.path.join(os.path.dirname(__file__), "TTS-scripts")

# ── Import modules with dots in filenames ───────────────────────────────────

def _import_from_path(module_name: str, file_path: str):
    """Import a Python module from an arbitrary file path."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


# Lazy-loaded module references
_tts_v2 = None
_tts_v3 = None


def _get_tts_v2():
    global _tts_v2
    if _tts_v2 is None:
        path = os.path.join(TTS_DIR, "tts_11v2.1.py")
        logger.info(f"Loading TTS v2 module from {path}")
        _tts_v2 = _import_from_path("tts_11v2_1", path)
    return _tts_v2


def _get_tts_v3():
    global _tts_v3
    if _tts_v3 is None:
        path = os.path.join(TTS_DIR, "tts_11v3.1.py")
        logger.info(f"Loading TTS v3 module from {path}")
        _tts_v3 = _import_from_path("tts_11v3_1", path)
    return _tts_v3


# ── Public API ──────────────────────────────────────────────────────────────

def generate_audio(text: str, voice_id: str, language: str) -> bytes:
    """
    Generate TTS audio using the appropriate ElevenLabs model.

    EN → eleven_multilingual_v2 (tts_11v2.1.py)
    HI → eleven_v3 (tts_11v3.1.py)

    Args:
        text: The text to convert to speech
        voice_id: ElevenLabs voice ID
        language: "en" or "hi"

    Returns:
        Audio bytes (MP3)
    """
    # Ensure API key is set
    api_key = os.getenv("ELEVEN_LABS_API_KEY")
    if not api_key:
        raise RuntimeError("ELEVEN_LABS_API_KEY not set")
    os.environ["ELEVEN_LABS_API_KEY"] = api_key

    if language == "hi":
        logger.info(f"Generating Hindi audio ({len(text)} chars) with eleven_v3")
        tts = _get_tts_v3()
        return tts.generate_tts_v3(text=text, voice_id=voice_id)
    else:
        logger.info(f"Generating English audio ({len(text)} chars) with eleven_multilingual_v2")
        tts = _get_tts_v2()
        return tts.generate_tts(text=text, voice_id=voice_id, language="en", output_format="mp3_44100_128")

"""GUI configuration.

Reads settings from the environment first, then from ``gui/.env`` if present,
with sensible local defaults. The only thing the GUI needs to know about the
backend is its HTTP base URL.
"""

from __future__ import annotations

import os
from pathlib import Path

GUIDIR = Path(__file__).resolve().parent


def _load_dotenv(path: Path) -> None:
    """Tiny .env loader (KEY=VALUE lines) — avoids an extra dependency."""
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


_load_dotenv(GUIDIR / ".env")

API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
# Qwen3-VL:8b can take several minutes per answer on CPU; do not time out early.
API_TIMEOUT_SECONDS = float(os.getenv("API_TIMEOUT_SECONDS", "1200"))

# Trim a finished answer that exceeds this many characters (defensive only).
MAX_ANSWER_CHARS = int(os.getenv("MAX_ANSWER_CHARS", "20000"))

# Local voice input/output. The transcript still goes through the exact same
# RAG /chat endpoint as typed input; these values only control audio I/O.
VOICE_ENABLED = os.getenv("VOICE_ENABLED", "true").lower() in {"1", "true", "yes", "on"}
VOICE_STT_MODEL = os.getenv("VOICE_STT_MODEL", "small")
VOICE_STT_DEVICE = os.getenv("VOICE_STT_DEVICE", "cpu")
VOICE_STT_COMPUTE_TYPE = os.getenv("VOICE_STT_COMPUTE_TYPE", "int8")
VOICE_SAMPLE_RATE = int(os.getenv("VOICE_SAMPLE_RATE", "16000"))
VOICE_SILENCE_SECONDS = float(os.getenv("VOICE_SILENCE_SECONDS", "1.2"))
VOICE_SILENCE_THRESHOLD = float(os.getenv("VOICE_SILENCE_THRESHOLD", "350"))
VOICE_MIN_RECORD_SECONDS = float(os.getenv("VOICE_MIN_RECORD_SECONDS", "0.5"))
VOICE_MAX_RECORD_SECONDS = float(os.getenv("VOICE_MAX_RECORD_SECONDS", "20"))
VOICE_TTS_ARABIC_VOICE = os.getenv("VOICE_TTS_ARABIC_VOICE", "ar-EG-SalmaNeural")
VOICE_TTS_ENGLISH_VOICE = os.getenv("VOICE_TTS_ENGLISH_VOICE", "en-US-AvaMultilingualNeural")

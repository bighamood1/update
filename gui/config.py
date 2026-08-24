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
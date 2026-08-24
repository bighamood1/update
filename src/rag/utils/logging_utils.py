"""Structured logging configuration.

Logs are written both to the console and to rotating files under
``logs/``. Nothing sensitive is ever logged.
"""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from ..config import PROJECT_ROOT, get_config

_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(log_level: str | None = None) -> None:
    """Configure root logger with console + rotating file handlers.

    Safe to call multiple times; reconfiguration is idempotent.
    """
    cfg = get_config()
    level = (log_level or cfg["log_level"] or "INFO").upper()
    logs_dir: Path = cfg["logs_dir"] or (PROJECT_ROOT / "logs")
    logs_dir.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    root.setLevel(level)

    # Avoid stacking duplicate handlers when called repeatedly.
    for handler in list(root.handlers):
        root.removeHandler(handler)

    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter(_FORMAT, _DATE_FORMAT))

    file_handler = RotatingFileHandler(
        logs_dir / "rag.log",
        maxBytes=5_000_000,
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setFormatter(logging.Formatter(_FORMAT, _DATE_FORMAT))

    root.addHandler(console)
    root.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    """Return a logger for the given module name (after setup_logging)."""
    return logging.getLogger(name)

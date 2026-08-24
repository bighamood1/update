"""Language / text-direction utilities (Arabic RTL, English LTR, mixed)."""

from __future__ import annotations

import re

from PySide6.QtCore import Qt

_ARABIC_RE = re.compile(
    r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]"
)
_HEBREW_RE = re.compile(r"[\u0590-\u05FF]")
_LETTER_RE = re.compile(r"[^\W\d_]", re.UNICODE)
_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)

_RTL_RATIO = 0.25
_RTL_ABS_THRESHOLD = 3


def _clean_text(text: str) -> str:
    text = _URL_RE.sub(" ", text or "")
    return re.sub(r"[\s\d\W_]+", " ", text, flags=re.UNICODE)


def direction_for(text: str | None) -> Qt.LayoutDirection:
    """Return RTL when Arabic characters dominate the letters of ``text``."""
    text = text or ""
    if not text.strip():
        return Qt.LayoutDirection.LeftToRight
    cleaned = _clean_text(text)
    letters = _LETTER_RE.findall(cleaned)
    if not letters:
        return Qt.LayoutDirection.LeftToRight
    arabic = len(_ARABIC_RE.findall(cleaned))
    hebrew = len(_HEBREW_RE.findall(cleaned))
    rtl_count = arabic + hebrew
    if rtl_count >= _RTL_ABS_THRESHOLD and rtl_count >= len(letters) * _RTL_RATIO:
        return Qt.LayoutDirection.RightToLeft
    return Qt.LayoutDirection.LeftToRight


def is_rtl(text: str | None) -> bool:
    return direction_for(text) == Qt.LayoutDirection.RightToLeft


def direction_name(text: str | None) -> str:
    return "rtl" if is_rtl(text) else "ltr"
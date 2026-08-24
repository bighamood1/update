"""Final user-facing answer cleanup.

This module is the last boundary before an answer leaves the RAG pipeline.
It does not invent or rewrite facts; it removes accidental implementation
artifacts (reasoning preambles, source labels, raw context headers, bare URLs)
and normalizes whitespace/Markdown so every client receives the same clean
answer string.
"""

from __future__ import annotations

import re

from .validation import normalize_markdown_structure, strip_reasoning_artifacts

_SOURCE_LABEL_RE = re.compile(
    r"(?im)^\s*(?:source|evidence item)\s*\d+\s*[:\-]?\s*"
)
_INLINE_SOURCE_RE = re.compile(
    r"(?i)\[?\s*(?:source|evidence item)\s*\d+\s*\]?"
)
_SOURCE_PHRASE_RE = re.compile(
    r"(?i)\b(?:according to|based on|from)\s+(?:the\s+)?"
    r"(?:retrieved\s+)?(?:context|sources?(?:\s+\d+)?|"
    r"evidence(?:\s+item\s+\d+)?)[:,]?\s*"
)
_RAW_CONTEXT_HEADER_RE = re.compile(
    r"(?im)^\s*(?:question|context|end of context|rules|instructions|task|language)\s*:\s*.*$"
)
_URL_RE = re.compile(r"https?://[^\s)\]}>'\"]+", re.IGNORECASE)


def format_final_answer(answer: str, *, remove_urls: bool = True) -> tuple[str, list[str]]:
    """Return ``(cleaned_answer, issues)`` for a final response.

    The cleanup is intentionally conservative. It removes only artifacts that
    should never appear in a normal user-facing answer; factual text, Arabic
    right-to-left content, tables and lists are preserved.
    """
    issues: list[str] = []
    text, stripped_reasoning = strip_reasoning_artifacts(answer or "")
    if stripped_reasoning:
        issues.append("reasoning_artifact_stripped")

    before = text
    text = _RAW_CONTEXT_HEADER_RE.sub("", text)
    text = _SOURCE_PHRASE_RE.sub("", text)
    text = _SOURCE_LABEL_RE.sub("", text)
    text = _INLINE_SOURCE_RE.sub("", text)
    if text != before:
        issues.append("source_or_context_label_stripped")

    if remove_urls and _URL_RE.search(text):
        text = _URL_RE.sub("", text)
        issues.append("url_removed_from_answer")

    text = _dedupe_repeated_units(text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if not text:
        return "", issues
    return normalize_markdown_structure(text).strip(), issues


def _dedupe_repeated_units(text: str) -> str:
    """Drop repeated lines and repeated plain sentences while preserving order."""
    kept_lines: list[str] = []
    seen: set[str] = set()
    for line in (text or "").splitlines():
        key = re.sub(r"\s+", " ", line).strip().lower()
        if key and key in seen:
            continue
        if key:
            seen.add(key)
        kept_lines.append(_dedupe_sentences_in_line(line))
    return "\n".join(kept_lines)


def _dedupe_sentences_in_line(line: str) -> str:
    if line.lstrip().startswith(("-", "*", "|", "#")):
        return line
    parts = re.split(r"(?<=[.!؟])\s+", line)
    if len(parts) < 2:
        return line
    seen: set[str] = set()
    kept: list[str] = []
    for part in parts:
        key = re.sub(r"\W+", " ", part, flags=re.UNICODE).strip().lower()
        if key and key in seen:
            continue
        if key:
            seen.add(key)
        kept.append(part)
    return " ".join(kept)

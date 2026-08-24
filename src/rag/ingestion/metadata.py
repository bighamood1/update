"""Deterministic metadata enrichment for NMU documents.

The scraped dataset does not carry a ``faculty`` field, yet most faculty pages
follow the URL shape ``/faculties/<id>/<slug>``. This module derives a
canonical ``faculty`` key (English, lowercase-hyphenated) and numeric
``faculty_id`` from the URL (falling back to the page title), so retrieval can
optionally filter on faculty metadata WITHOUT touching ``documents.jsonl``.

The dataset is never modified; enrichment is applied in-memory at index-build
time and stored only in the Chroma chunk metadata.
"""

from __future__ import annotations

import re
import unicodedata

# Canonical faculty key per numeric /faculties/<id>/ URL id (observed index).
_FACULTY_ID_TO_KEY = {
    "1": "business",
    "2": "law",
    "3": "engineering",
    "4": "computer-science-and-engineering",
    "5": "textile-science-and-engineering",
    "6": "science",
    "7": "medicine",
    "8": "dentistry",
    "9": "pharmacy",
    "10": "social-and-human-sciences",
    "12": "applied-health-sciences",
    "13": "nursing",
    "14": "graduate-studies",
    "16": "mass-media-and-communication",
    "21": "physical-therapy",
}

_FACULTIES_URL_RE = re.compile(r"/faculties/(\d+)(?:/|$)", re.IGNORECASE)
_TITLE_EN_RE = re.compile(r"faculty\s+of\s+([^|]+)", re.IGNORECASE)


def normalize_key(label: str) -> str:
    """Turn a faculty label into a canonical lowercase-hyphenated key."""
    if not label:
        return ""
    text = unicodedata.normalize("NFKC", label).strip()
    text = re.sub(r"[^A-Za-z0-9\u0600-\u06FF]+", "-", text)
    return text.strip("-").lower()


def derive_faculty(url: str | None, title: str | None) -> tuple[str | None, str | None]:
    """Return ``(faculty_key, faculty_id)`` derived from URL / title."""
    url = url or ""
    title = title or ""

    m = _FACULTIES_URL_RE.search(url)
    if m:
        fid = m.group(1)
        key = _FACULTY_ID_TO_KEY.get(fid)
        if key:
            return key, fid
        # Unknown id: fall through to slug / title below (id still useful).
        slug = url[m.end():].split("/")[0] if url[m.end():] else ""
        slug = slug.split("?")[0]
        if slug and slug != fid:
            key = normalize_key(slug)
            if key and len(key) >= 3:
                return key, fid
        # Known id but no key in map -> reuse id-derived English slug if any.

    tm = _TITLE_EN_RE.search(title)
    if tm:
        key = normalize_key(tm.group(1))
        if key:
            return key, m.group(1) if m else None

    # Arabic title "كلية X | ..."
    am = re.match(r"\s*كلية\s+([^|]+)", title)
    if am:
        key = normalize_key(am.group(1))
        if key:
            return key, m.group(1) if m else None

    return None, (m.group(1) if m else None)


def enrich(doc) -> "object":
    """Set ``faculty`` / ``faculty_id`` on a NormalizedDocument (in place)."""
    key, fid = derive_faculty(doc.url, doc.title)
    if key and not doc.faculty:
        doc.faculty = key
    if fid and not doc.faculty_id:
        doc.faculty_id = fid
    return doc
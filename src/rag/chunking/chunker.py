"""Intelligent, content-type-aware chunking.

Strategy
--------
1. Split the document into logical sections first, preferring semantic
   boundaries: lines that look like headings, blank-line-separated
   paragraphs, FAQ question/answer pairs, list items, and sentence ends.
2. Each logical section becomes a chunk if it fits within ``CHUNK_SIZE``.
3. Sections larger than ``CHUNK_SIZE`` are split with sentence-aware
   boundary detection and a configurable overlap, so meaning is preserved
   instead of blindly cutting mid-sentence.
4. Different content types use different section-splitting heuristics.

Chunk IDs are stable: ``sha256(document_id + "::" + index)``, so re-running
the indexer never re-inserts duplicates. Chunks carry section provenance
(``section_id``, ``section_index``) and document relationships
(``parent_document_id``, ``chunk_index``, ``chunk_count``) so a complete
semantic section or parent document can be recovered later.
"""

from __future__ import annotations

import hashlib
import re

from ..config import get_config
from ..schemas.documents import DocumentChunk, NormalizedDocument
from ..utils.logging_utils import get_logger

logger = get_logger(__name__)

# --------------------------------------------------------------------------
# Boundary detection helpers
# --------------------------------------------------------------------------

# Heading-like lines: short lines that look like titles/section headers.
_HEADING_RE = re.compile(
    r"^\s*(?:\d+[.)]\s+|\u2022\s+|[A-Z][A-Za-z0-9 /&'-]{2,60}|[\u0600-\u06FF][^:]{2,80})$",
    re.UNICODE,
)

_SENTENCE_END_RE = re.compile(r"(?<=[.!?\u060C\u061F\u2026])[\s]+")

_FAQ_QUESTION_MARKERS = ("what", "how", "why", "when", "where", "who", "هل", "ما", "كيف", "لماذا", "متى", "أين", "من")

_NEWS_DATE_RE = re.compile(r"^\s*\d{1,2}\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}", re.IGNORECASE)

_MAX_HEADING_LEN = 120


def _is_heading(line: str) -> bool:
    line = line.strip()
    if not line or len(line) > _MAX_HEADING_LEN:
        return False
    if _NEWS_DATE_RE.match(line):
        return False
    # English / Arabic title-ish lines.
    return bool(_HEADING_RE.match(line))


def _split_sentences(text: str) -> list[str]:
    """Split text on sentence boundaries (works for Arabic + English)."""
    parts = _SENTENCE_END_RE.split(text)
    return [p.strip() for p in parts if p and p.strip()]


def _split_paragraphs(text: str) -> list[str]:
    """Split into paragraphs on blank lines / newlines, keeping headings."""
    raw = re.split(r"\n\s*\n|\r\n\s*\r\n", text)
    parts: list[str] = []
    for block in raw:
        block = block.strip()
        if not block:
            continue
        # Preserve multi-line logical blocks (e.g. label: value) as one.
        parts.append(block)
    return parts


def _is_faq_question(line: str) -> bool:
    low = line.strip().lower()
    if not low:
        return False
    if low.rstrip("?؟").endswith(("؟", "?")):
        return True
    return any(low.startswith(m) for m in _FAQ_QUESTION_MARKERS)


def _split_faq(text: str) -> list[str]:
    """Split FAQ text into question+answer units."""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    chunks: list[str] = []
    current: list[str] = []
    for line in lines:
        if _is_faq_question(line) and current:
            chunks.append("\n".join(current))
            current = []
        current.append(line)
    if current:
        chunks.append("\n".join(current))
    return [c for c in chunks if c]


def _split_by_headings_and_paragraphs(text: str) -> list[str]:
    """Generic section splitter: headings mark new sections; otherwise paragraphs.

    Consecutive heading-like lines (e.g. a navigation menu of faculty names)
    are grouped into a single section so short menu entries are not dropped.
    When a heading run immediately follows a line ending in ``:`` (a list
    introducer such as "The university has the following faculties:"), the run
    is kept in the same section so the intro and its list stay one unit.
    """
    lines = text.splitlines()
    sections: list[str] = []
    current: list[str] = []
    i = 0
    n = len(lines)
    while i < n:
        stripped = lines[i].strip()
        if not stripped:
            i += 1
            continue
        if _is_heading(stripped):
            # Consume a run of consecutive heading-like lines as one section.
            run = [stripped]
            j = i + 1
            while j < n:
                nxt = lines[j].strip()
                if nxt and _is_heading(nxt):
                    run.append(nxt)
                    j += 1
                elif not nxt:
                    j += 1
                else:
                    break
            # A run right after a ":" introducer belongs to that section.
            last_current = next(
                (ln for ln in reversed(current) if ln.strip()), ""
            ).strip()
            if last_current.endswith((":", "؟:")):
                current.extend(run)
                i = j
                continue
            if current:
                sections.append("\n".join(current))
                current = []
            current = list(run)
            i = j
        else:
            current.append(stripped)
            i += 1
    if current:
        sections.append("\n".join(current))
    return [s for s in sections if s and s.strip()]


# --------------------------------------------------------------------------
# Chunker
# --------------------------------------------------------------------------


class Chunker:
    """Split NormalizedDocuments into traceable, stable-id chunks."""

    def __init__(self, chunk_size: int | None = None, chunk_overlap: int | None = None) -> None:
        cfg = get_config()
        self.chunk_size = chunk_size or cfg["chunk_size"]
        self.chunk_overlap = chunk_overlap if chunk_overlap is not None else cfg["chunk_overlap"]
        self.min_chunk_chars = cfg["min_chunk_chars"]

    # -- public API -------------------------------------------------------

    def chunk_document(self, doc: NormalizedDocument) -> list[DocumentChunk]:
        """Return a list of chunks for one document."""
        text = doc.text.strip()
        if not text:
            return []

        sections = self._split_sections(doc.content_type or "", text)

        # Assign stable section identifiers before building chunks.
        sections = [(self._section_id(doc, i, sec), i, sec) for i, sec in enumerate(sections)]

        chunks: list[DocumentChunk] = []
        for section_id, section_index, section in sections:
            for piece in self._ensure_max_size(section):
                if len(piece) < self.min_chunk_chars:
                    continue
                chunk = self._build_chunk(
                    doc, piece, len(chunks), section_id, section_index
                )
                chunks.append(chunk)

        # Tag chunk ordering / totals now that the full list is known.
        total = len(chunks)
        for index, chunk in enumerate(chunks):
            chunk.chunk_index = index
            chunk.chunk_count = total
        return chunks

    # -- content-type-aware section splitting ------------------------------

    def _split_sections(self, content_type: str, text: str) -> list[str]:
        ct = content_type.strip().lower()
        if ct == "faq":
            return _split_faq(text)
        if ct in ("news", "event", "scholarship", "admission", "tuition",
                  "regulation", "policy", "guide", "faculty", "program",
                  "facility", "career", "tender", "about", "home", "president",
                  "administration", "contact"):
            return _split_by_headings_and_paragraphs(text)
        # Default: paragraph-based.
        return _split_paragraphs(text)

    # -- size enforcement with semantic boundaries --------------------------

    def _ensure_max_size(self, section: str) -> list[str]:
        if len(section) <= self.chunk_size:
            return [section]

        # Prefer sentence boundaries.
        sentences = _split_sentences(section)
        if len(sentences) > 1:
            return self._pack(sentences)

        # Fall back to character split.
        return self._char_split(section)

    def _pack(self, sentences: list[str]) -> list[str]:
        """Pack sentences into chunks <= chunk_size with overlap."""
        chunks: list[str] = []
        current: list[str] = []
        current_len = 0
        overlap_sentences: list[str] = []

        def flush() -> None:
            nonlocal current, current_len, overlap_sentences
            if current:
                chunks.append(" ".join(current))
            # Keep last few sentences for overlap.
            total = sum(len(s) for s in current)
            keep: list[str] = []
            budget = 0
            for s in reversed(current):
                if budget + len(s) > self.chunk_overlap or len(keep) >= 2:
                    break
                keep.insert(0, s)
                budget += len(s)
            overlap_sentences = keep
            current = []
            current_len = 0

        for sentence in sentences:
            sep_len = 1 if current else 0
            if current_len + sep_len + len(sentence) > self.chunk_size and current:
                flush()
                current = list(overlap_sentences)
                current_len = sum(len(s) for s in current) + len(current) - 1 if current else 0
            current.append(sentence)
            current_len += len(sentence) + 1
        flush()

        return [c for c in chunks if c.strip()]

    def _char_split(self, text: str) -> list[str]:
        """Character-based split (last resort), preserving overlap."""
        size, overlap = self.chunk_size, self.chunk_overlap
        chunks: list[str] = []
        start = 0
        n = len(text)
        while start < n:
            end = min(start + size, n)
            # Try not to split mid-word.
            if end < n:
                last_space = text.rfind(" ", start, end)
                if last_space > start + size // 2:
                    end = last_space
            piece = text[start:end].strip()
            if piece:
                chunks.append(piece)
            if end >= n:
                break
            start = max(end - overlap, start + 1)
        return chunks

    # -- chunk building -----------------------------------------------------

    @staticmethod
    def _section_id(doc: NormalizedDocument, index: int, section_text: str) -> str:
        """Stable, readable section id: first heading line + hash of content."""
        first_line = next(
            (ln.strip() for ln in section_text.splitlines() if ln.strip()),
            section_text[:40],
        )
        slug = re.sub(r"[^\w\u0600-\u06FF]+", "-", first_line).strip("-").lower()[:48]
        digest = hashlib.sha256(
            f"{doc.id}::{index}::{section_text[:120]}".encode("utf-8")
        ).hexdigest()[:10]
        return f"{slug or 'section'}-{digest}"

    def _build_chunk(
        self,
        doc: NormalizedDocument,
        text: str,
        index: int,
        section_id: str,
        section_index: int,
    ) -> DocumentChunk:
        chunk_id = self._stable_chunk_id(doc.id, index)
        return DocumentChunk(
            chunk_id=chunk_id,
            document_id=doc.id,
            parent_document_id=doc.id,
            section_id=section_id,
            section_index=section_index,
            text=text,
            title=doc.title,
            source_url=doc.url,
            language=doc.language,
            content_type=doc.content_type,
            section=doc.section,
            faculty=doc.faculty,
            faculty_id=doc.faculty_id,
            program=doc.program,
            department=doc.department,
            academic_year=doc.academic_year,
            published_at=doc.published_at,
            document_hash=doc.content_hash,
        )

    @staticmethod
    def _stable_chunk_id(document_id: str, index: int) -> str:
        raw = f"{document_id}::{index}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

"""Document and chunk schemas.

These are the typed, internal representations used everywhere in the RAG
pipeline. Raw records from ``documents.jsonl`` are first normalized into
:class:`NormalizedDocument` objects (unknown fields default to ``None``),
then split into :class:`DocumentChunk` objects that carry full provenance.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class RawDocument(BaseModel):
    """A single raw JSONL record exactly as produced by PHASE 1.

    Only fields the pipeline relies on are typed; everything else is
    preserved in ``extra`` so no metadata is ever silently dropped.
    """

    id: str
    text: str | None = None
    title: str | None = None
    language: str | None = None
    content_type: str | None = None
    url: str | None = None
    source_domain: str | None = None
    section: str | None = None
    faculty: str | None = None
    faculty_id: str | None = None
    program: str | None = None
    department: str | None = None
    academic_year: str | None = None
    published_at: str | None = None
    scraped_at: str | None = None
    content_hash: str | None = None
    metadata: dict[str, Any] | None = Field(default_factory=dict)
    extra: dict[str, Any] = Field(default_factory=dict)


class NormalizedDocument(BaseModel):
    """A validated, text-bearing document ready for chunking."""

    id: str
    text: str
    title: str | None = None
    language: str | None = None
    content_type: str | None = None
    url: str | None = None
    source_domain: str | None = None
    section: str | None = None
    faculty: str | None = None
    faculty_id: str | None = None
    program: str | None = None
    department: str | None = None
    academic_year: str | None = None
    published_at: str | None = None
    scraped_at: str | None = None
    content_hash: str | None = None


class DocumentChunk(BaseModel):
    """A chunk of a document with complete source traceability."""

    chunk_id: str
    document_id: str
    parent_document_id: str | None = None
    section_id: str | None = None
    section_index: int | None = None
    chunk_index: int | None = None
    chunk_count: int | None = None
    text: str
    title: str | None = None
    source_url: str | None = None
    language: str | None = None
    content_type: str | None = None
    section: str | None = None
    faculty: str | None = None
    faculty_id: str | None = None
    program: str | None = None
    department: str | None = None
    academic_year: str | None = None
    published_at: str | None = None
    document_hash: str | None = None

    @property
    def metadata(self) -> dict[str, Any]:
        """Compact metadata dict suitable for vector-store persistence."""
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "parent_document_id": self.parent_document_id,
            "section_id": self.section_id,
            "section_index": self.section_index,
            "chunk_index": self.chunk_index,
            "chunk_count": self.chunk_count,
            "title": self.title,
            "source_url": self.source_url,
            "language": self.language,
            "content_type": self.content_type,
            "section": self.section,
            "faculty": self.faculty,
            "faculty_id": self.faculty_id,
            "program": self.program,
            "department": self.department,
            "academic_year": self.academic_year,
            "published_at": self.published_at,
            "document_hash": self.document_hash,
        }


class RetrievedChunk(BaseModel):
    """A chunk returned by the retriever with its similarity score."""

    chunk_id: str
    document_id: str
    text: str
    score: float
    dense_score: float | None = None
    bm25_score: float | None = None
    rerank_score: float | None = None
    title: str | None = None
    source_url: str | None = None
    language: str | None = None
    content_type: str | None = None
    section: str | None = None
    faculty: str | None = None
    faculty_id: str | None = None
    program: str | None = None
    department: str | None = None
    academic_year: str | None = None
    published_at: str | None = None
    document_hash: str | None = None
    section_id: str | None = None
    section_index: int | None = None
    chunk_index: int | None = None
    chunk_count: int | None = None

    def to_context_block(self, index: int = 1) -> str:
        """Format this chunk as a context block for the LLM prompt."""
        title = self.title or "(untitled)"
        url = self.source_url or "(no url)"
        language = self.language or "unknown"
        content_type = self.content_type or "unknown"
        lines = [
            f"[Source {index}]",
            f"Title: {title}",
            f"URL: {url}",
            f"Type: {content_type}",
            f"Language: {language}",
        ]
        if self.faculty:
            lines.append(f"Faculty: {self.faculty}")
        if self.program:
            lines.append(f"Program: {self.program}")
        if self.section:
            lines.append(f"Section: {self.section}")
        if self.published_at:
            lines.append(f"Published: {self.published_at}")
        if self.chunk_index is not None and self.chunk_count:
            lines.append(f"Part: {self.chunk_index + 1}/{self.chunk_count}")
        lines.append(f"Content: {self.text}")
        return "\n".join(lines)

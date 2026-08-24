"""Context assembly for the RAG pipeline.

Transforms retrieved chunks into a compact, source-anchored context string
that is sent to the LLM together with the user question.
"""

from __future__ import annotations

from typing import Any

from ..config import get_config
from ..schemas.documents import RetrievedChunk


class ContextBuilder:
    """Transform retrieved chunks into a compact, relevant context string."""

    def __init__(
        self,
        max_chunks: int | None = None,
        max_chars: int | None = None,
    ) -> None:
        cfg = get_config()
        self.max_chunks = max_chunks if max_chunks is not None else cfg["final_context_chunks"]
        self.max_chars = max_chars if max_chars is not None else cfg["context_max_chars"]

    def build(
        self,
        chunks: list[RetrievedChunk],
        max_chunks: int | None = None,
        max_chars: int | None = None,
    ) -> str:
        """Assemble a context string; per-call limits override the instance defaults."""
        max_chunks = self.max_chunks if max_chunks is None else max_chunks
        max_chars = self.max_chars if max_chars is None else max_chars
        chunks = self._remove_redundant(chunks)
        # Preserve retriever order. Earlier versions regrouped all chunks by
        # source URL, which could move lower-priority chunks from the first URL
        # ahead of coverage-critical evidence from another URL (e.g. a full
        # Arabic tuition source). The retriever already applies source
        # diversity; context assembly must not undo that ranking.
        grouped = chunks[: max_chunks]

        blocks: list[str] = []
        total = 0
        for i, chunk in enumerate(grouped, start=1):
            block = self._evidence_block(chunk, i)
            if total + len(block) > max_chars:
                break
            blocks.append(block)
            total += len(block)
        return "\n\n".join(blocks) if blocks else ""

    @staticmethod
    def _evidence_block(chunk: RetrievedChunk, index: int) -> str:
        """Format a clean grounding block for generation.

        URLs, retrieval scores and "Source N" labels stay out of the prompt.
        The application carries sources as structured metadata separately, so
        the LLM only sees factual evidence plus minimal provenance cues needed
        to interpret it.
        """
        title = chunk.title or "(untitled)"
        language = chunk.language or "unknown"
        content_type = chunk.content_type or "unknown"
        lines = [
            f"[Evidence item {index}]",
            f"Title: {title}",
            f"Type: {content_type}",
            f"Language: {language}",
        ]
        if chunk.faculty:
            lines.append(f"Faculty: {chunk.faculty}")
        if chunk.program:
            lines.append(f"Program: {chunk.program}")
        if chunk.section:
            lines.append(f"Section: {chunk.section}")
        if chunk.published_at:
            lines.append(f"Published: {chunk.published_at}")
        if chunk.chunk_index is not None and chunk.chunk_count:
            lines.append(f"Part: {chunk.chunk_index + 1}/{chunk.chunk_count}")
        lines.append(f"Content: {chunk.text}")
        return "\n".join(lines)

    @staticmethod
    def _remove_redundant(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        """Drop chunks that duplicate an earlier chunk's content verbatim.

        Redundancy is scoped to the same source page: two pages may
        legitimately carry identical sentences, while within one page
        duplicated paragraphs are noise.
        """
        seen: set[tuple[str, str]] = set()
        kept: list[RetrievedChunk] = []
        for c in chunks:
            norm = " ".join(c.text.split())[:300]
            key = (c.source_url or "no-url", norm)
            if key in seen:
                continue
            seen.add(key)
            kept.append(c)
        return kept

    def sources(self, chunks: list[RetrievedChunk]) -> list[dict[str, Any]]:
        seen: set[tuple[str, str]] = set()
        sources: list[dict[str, Any]] = []
        for c in chunks:
            key = (c.source_url or "", c.title or "")
            if key in seen:
                continue
            seen.add(key)
            sources.append(
                {
                    "chunk_id": c.chunk_id,
                    "document_id": c.document_id,
                    "title": c.title,
                    "url": c.source_url,
                    "language": c.language,
                    "content_type": c.content_type,
                    "section": c.section,
                    "faculty": c.faculty,
                    "program": c.program,
                    "academic_year": c.academic_year,
                    "published_at": c.published_at,
                    "score": c.score,
                }
            )
        return sources

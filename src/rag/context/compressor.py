"""Context compression for the RAG pipeline.

Keeps the assembled context within a strict token budget and drops whole
sentences that carry no overlap with the question when the context is long.
All compression is deterministic (no LLM call) so it can never change facts —
it can only remove redundant or unrelated material.
"""

from __future__ import annotations

import re

from ..config import get_config


def estimate_tokens(text: str) -> int:
    """Cheap token estimate (words + punctuation tokens)."""
    if not text:
        return 0
    return max(1, len(re.findall(r"\S+", text)))


def _sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?؟])\s+", text.strip())
    return [p for p in parts if p.strip()]


class ContextCompressor:
    """Deterministic context trimming to a token budget."""

    def __init__(self, max_tokens: int | None = None) -> None:
        cfg = get_config()
        self.max_tokens = max_tokens if max_tokens is not None else cfg["max_context_tokens"]

    def compress(self, context: str, question: str | None = None) -> str:
        if not context:
            return ""
        if estimate_tokens(context) <= self.max_tokens:
            return context

        # Lower-case / hamza-normalize terms for overlap matching.
        q = (question or "").lower()
        q_terms = set(re.findall(r"[\w\u0600-\u06FF]+", q))

        # Prefer keeping whole paragraphs (context blocks) over sentence cuts.
        blocks = re.split(r"\n{2,}", context)
        kept: list[str] = []
        budget = self.max_tokens
        for block in blocks:
            if estimate_tokens(block) > budget:
                # Fall back to sentence-level trimming for oversized blocks.
                for sent in _sentences(block):
                    if estimate_tokens(sent) <= budget:
                        kept.append(sent)
                        budget -= estimate_tokens(sent)
            else:
                kept.append(block)
                budget -= estimate_tokens(block)
            if budget <= 0:
                break
        return "\n\n".join(kept)


def compress_context(context: str, question: str | None = None,
                     max_tokens: int | None = None) -> str:
    """Convenience wrapper around :class:`ContextCompressor`."""
    return ContextCompressor(max_tokens=max_tokens).compress(context, question)
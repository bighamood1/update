"""Lightweight retrieval memory (learned retrieval patterns).

After a successful generation, the pipeline remembers the question's best
sources + strategy (in the runtime SQLite store). When a future question
normalizes to the same form, those historical source URLs are added to the
candidate pool as SOFT hints.

Safety (Phase 14): historical patterns are NEVER the only source — they are
seeded into the pool and still go through reranking + diversity + final
top-k, and full fallback retrieval always remains available.
"""

from __future__ import annotations

import json

from ..cache.store import RuntimeStore, get_runtime_store
from ..utils.logging_utils import get_logger

logger = get_logger(__name__)


class RetrievalMemory:
    """Read/write access to historical retrieval patterns (best-effort)."""

    def __init__(self, store: RuntimeStore | None = None) -> None:
        self.store = store or get_runtime_store()

    def hint(
        self,
        kb_version: str,
        normalized_question: str,
        *,
        understanding=None,
    ) -> list[str]:
        """Return trusted source URLs remembered for this question pattern."""
        if not normalized_question:
            return []
        urls: list[str] = []
        seen = set()
        if understanding is not None:
            for row in self.strategy_hints(kb_version, normalized_question, understanding):
                if float(row.get("confidence") or 0.0) <= 0:
                    continue
                try:
                    sources = json.loads(row.get("source_urls_json") or "[]")
                except (ValueError, TypeError):
                    sources = []
                for url in sources:
                    if url and url not in seen:
                        seen.add(url)
                        urls.append(url)
        row = self.store.get_memory_hint(kb_version, normalized_question)
        if not row:
            return urls
        try:
            sources = json.loads(row.get("sources_json") or "[]")
        except (ValueError, TypeError):
            sources = []
        for s in sources:
            url = (s.get("url") or "").strip()
            if url and url not in seen:
                seen.add(url)
                urls.append(url)
        return urls

    def strategy_hints(self, kb_version: str, normalized_question: str, understanding) -> list[dict]:
        try:
            return self.store.get_strategy_hints(
                kb_version=kb_version,
                normalized_question=normalized_question,
                intent=understanding.intent,
                language=understanding.language,
                category=understanding.category,
                faculty=understanding.faculty,
                topic=getattr(understanding, "topic", None),
                subtopic=getattr(understanding, "subtopic", None),
            )
        except Exception:  # noqa: BLE001
            logger.exception("Strategy feedback lookup failed; ignoring")
            return []

    def should_diversify(self, kb_version: str, normalized_question: str, understanding) -> bool:
        rows = self.strategy_hints(kb_version, normalized_question, understanding)
        return any(float(r.get("confidence") or 0.0) < 0 for r in rows)

    def remember(
        self,
        *,
        kb_version: str,
        normalized_question: str,
        intent: str,
        category: str,
        faculty: str | None,
        sources: list[dict],
        strategy: str,
    ) -> None:
        try:
            self.store.upsert_retrieval_memory(
                kb_version=kb_version,
                normalized_question=normalized_question,
                intent=intent,
                category=category,
                faculty=faculty,
                sources=sources,
                strategy=strategy,
            )
        except Exception:  # noqa: BLE001 - memory must never break RAG
            logger.exception("Retrieval memory update failed; continuing")

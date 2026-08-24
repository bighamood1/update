"""Feedback analytics + learning-data aggregation.

Pure analytics over the runtime store. NO model weights are ever modified:
the output feeds retrieval optimization, query expansion, semantic caching,
and a future LoRA / DPO / reranker training pipeline.
"""

from __future__ import annotations

from ..cache.store import RuntimeStore, get_runtime_store
from ..utils.logging_utils import get_logger

logger = get_logger(__name__)

# Rating -> numeric quality (0..1) used for cache / memory quality gates.
_RATING_SCORE = {"useful": 1.0, "somewhat": 0.3, "medium": 0.3, "not_useful": 0.0}


def quality_score_for_rating(rating: str) -> float:
    return _RATING_SCORE.get((rating or "").lower(), 0.5)


def initial_quality_score(answer: str, is_multi_intent: bool = False) -> float:
    """Deterministic initial quality for a NEW cache entry (no feedback yet).

    Starts neutral (0.5); a complete, non-empty answer bumps the score; a
    likely-incomplete multi-intent answer is kept neutral. Feedback later
    overwrites this with real user ratings.
    """
    score = 0.5
    text = (answer or "").strip()
    if not text:
        return 0.1
    if len(text) >= 120:
        score += 0.15
    if len(text) >= 300:
        score += 0.10
    if is_multi_intent and len(text) < 200:
        score = 0.4
    return max(0.1, min(1.0, round(score, 2)))


class FeedbackAnalytics:
    """Read-only analytics over the runtime store (best-effort)."""

    def __init__(self, store: RuntimeStore | None = None) -> None:
        self.store = store or get_runtime_store()

    def top_faqs(self, limit: int = 10) -> list[dict]:
        return self.store.top_faqs(limit)

    def failed_questions(self, limit: int = 20) -> list[dict]:
        return self.store.failed_questions(limit)

    def stats(self) -> dict:
        return self.store.stats()

    def training_dataset(self) -> list[dict]:
        """Phase 28: clean rows for future training (never trains anything)."""
        return self.store.export_training_rows()

"""Feedback submission + validation (best-effort, never breaks chat).

Accepted ratings: ``useful`` | ``somewhat`` | ``not_useful``.
The legacy GUI value ``medium`` is accepted and normalized to ``somewhat``.
Optional reasons are validated per rating (Phase 12). Unknown values are
rejected; storage failures are logged and never propagated.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..cache.store import RuntimeStore, get_runtime_store
from ..utils.logging_utils import get_logger

logger = get_logger(__name__)

VALID_RATINGS = ("useful", "somewhat", "not_useful")
RATING_ALIASES = {"medium": "somewhat"}

# Rating -> allowed optional reasons.
REASONS_BY_RATING: dict[str, tuple[str, ...]] = {
    "not_useful": (
        "incorrect_answer",
        "incomplete_answer",
        "misunderstood_question",
        "irrelevant_sources",
        "outdated_information",
        "other",
    ),
    "somewhat": ("partially_correct", "incomplete", "unclear"),
    "useful": (),
}


@dataclass
class FeedbackRecord:
    question_id: str
    rating: str
    reason: str | None = None
    feedback_id: str | None = None
    ok: bool = True
    message: str = ""


class FeedbackStore:
    """Validates and persists user feedback via the runtime SQLite store."""

    def __init__(self, store: RuntimeStore | None = None) -> None:
        self.store = store or get_runtime_store()

    def submit(
        self, question_id: str, rating: str, reason: str | None = None
    ) -> FeedbackRecord:
        question_id = (question_id or "").strip()
        rating = (rating or "").strip().lower()
        rating = RATING_ALIASES.get(rating, rating)
        reason = (reason or "").strip().lower() or None

        if not question_id:
            return FeedbackRecord(question_id="", rating=rating, ok=False,
                                  message="question_id is required")
        if rating not in VALID_RATINGS:
            return FeedbackRecord(question_id=question_id, rating=rating, ok=False,
                                  message=f"rating must be one of {', '.join(VALID_RATINGS)}")
        if reason is not None:
            allowed = REASONS_BY_RATING.get(rating, ())
            if reason not in allowed:
                return FeedbackRecord(question_id=question_id, rating=rating,
                                      reason=reason, ok=False,
                                      message=f"reason '{reason}' is not valid for rating '{rating}'")

        try:
            feedback_id = self.store.add_feedback(question_id, rating, reason)
        except Exception:  # noqa: BLE001 - feedback must never break chat
            logger.exception("Feedback storage failed for question_id=%s", question_id)
            return FeedbackRecord(question_id=question_id, rating=rating,
                                  reason=reason, ok=False,
                                  message="feedback could not be stored")
        if feedback_id is None:
            return FeedbackRecord(question_id=question_id, rating=rating,
                                  reason=reason, ok=False,
                                  message="unknown question_id")
        logger.info("Feedback recorded: %s = %s (reason=%s)", question_id, rating, reason or "-")
        return FeedbackRecord(question_id=question_id, rating=rating,
                              reason=reason, feedback_id=feedback_id, ok=True,
                              message="ok")

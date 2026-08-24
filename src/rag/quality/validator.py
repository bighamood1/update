"""Deterministic answer-quality control (Phase 16).

Reuses the existing hard grounding validation
(``rag.generation.validation.validate_answer``) and adds lightweight SOFT
checks: verbosity, multi-intent coverage and overall completeness. Everything
is deterministic — no second LLM call. The resulting 0..1 score feeds cache
quality initialization and analytics; it never silently rewrites the answer.
"""

from __future__ import annotations

from typing import Any

from ..config import get_config
from ..generation.validation import validate_answer
from ..query.understanding import QueryUnderstanding
from ..schemas.documents import RetrievedChunk

# Intents that legitimately need long / exhaustive answers.
_LONG_INTENTS = {"LIST", "FACULTY", "PROGRAM", "COMPARISON", "REGULATION",
                 "ADMINISTRATION", "SCHOLARSHIP", "TRANSFER"}


def evaluate_answer(
    answer: str,
    understanding: QueryUnderstanding | None,
    sources: list[dict] | None,
    retrieved: list[RetrievedChunk] | None = None,
) -> dict:
    """Return ``{ok, score, issues, cleaned}`` for a generated answer.

    ``ok`` reflects the HARD grounding gate (empty / fabricated URLs / emptied
    after cleanup). ``score`` is the soft 0..1 quality estimate used for cache
    initialization.
    """
    language = (understanding.language if understanding else "en")
    hard = validate_answer(
        answer,
        sources or [],
        retrieved,
        question_language="en" if language != "ar" else "ar",
    )
    text = hard["cleaned"]
    issues: list[str] = list(hard["issues"])
    score = 0.5

    if not text:
        return {"ok": False, "score": 0.0, "issues": issues, "cleaned": ""}

    intent = (understanding.intent if understanding else "FACT")
    max_chars = int(get_config().get("answer_max_chars", 4000) or 4000)

    # Verbosity (soft): only flagged, never fails on its own.
    if len(text) > max_chars and intent not in _LONG_INTENTS:
        issues.append("verbose")
        score -= 0.1

    # Completeness.
    if len(text) >= 120:
        score += 0.15
    if len(text) >= 300:
        score += 0.10

    # Multi-intent coverage (soft): a very short answer to a multi-part
    # question is suspect.
    if understanding is not None and understanding.is_multi_intent and len(text) < 200:
        issues.append("incomplete_multi")
        score = min(score, 0.45)

    # Source relevance: at least one source should exist for grounded answers.
    if sources:
        score += 0.05

    score = max(0.1, min(1.0, round(score, 2)))
    return {"ok": hard["ok"], "score": score, "issues": issues, "cleaned": text}
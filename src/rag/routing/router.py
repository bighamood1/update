"""Deterministic, rule-based query router.

Identifies intent, category, faculty, language and confidence WITHOUT any LLM.
Designed to be conservative: when confidence is below the configured
threshold, no restrictive metadata filter is applied (broad retrieval), and
multi-intent queries are flagged so retrieval never over-restricts.
"""

from __future__ import annotations

import re
import time

from ..config import get_config
from ..retrieval.query_normalizer import is_arabic, normalize_query
from .rules import CATEGORY_MAP, FACULTY_ALIASES, INTENT_KEYWORDS, _STRONG_MARKERS
from .schemas import RouteResult


class QueryRouter:
    """Route a question to intent/category/faculty/language + confidence."""

    def __init__(self) -> None:
        cfg = get_config()
        self.confidence_threshold = cfg.get("router_confidence_threshold", 0.80)
        self._aliases: list[tuple[str, str]] = [
            (key, normalize_query(alias).lower())
            for key, aliases in FACULTY_ALIASES.items()
            for alias in aliases
        ]
        self._aliases.sort(key=lambda item: len(item[1]), reverse=True)
        # Keywords are normalized once (lowercase, hamza-stripped) so they match
        # the normalized query regardless of Arabic orthographic variants.
        self._keywords: list[tuple[str, str, float]] = [
            (intent, normalize_query(kw), w)
            for intent, pairs in INTENT_KEYWORDS.items()
            for kw, w in pairs
        ]
        self._priority = {
            intent: i
            for i, intent in enumerate(_TIE_PRIORITY)
        }

    # -- public API --------------------------------------------------------

    def route(self, question: str) -> RouteResult:
        """Return a RouteResult for a question (never raises)."""
        start = time.perf_counter()
        text = (question or "").strip()
        if not text:
            result = RouteResult(confidence=0.0)
            result._routing_ms = 0.0
            return result

        normalized = normalize_query(text)
        language = "mixed" if _is_mixed(text) else ("ar" if is_arabic(text) else "en")
        if language == "mixed":
            language = "mixed"

        scores: dict[str, float] = {}
        matched_terms: list[str] = []
        for intent, keyword, weight in self._keywords:
            if keyword in normalized:
                scores[intent] = scores.get(intent, 0.0) + weight
                matched_terms.append(keyword)

        # Faculty detection (longest alias wins).
        faculty_key, faculty_id = self._detect_faculty(normalized)

        if not scores:
            intent = "FACT"
            confidence = 0.35
        else:
            # Rank by score, breaking ties by intent specificity.
            ranked = sorted(
                scores.items(),
                key=lambda kv: (-kv[1], self._priority.get(kv[0], 99)),
            )
            top_intent, top_score = ranked[0]
            runner_up = ranked[1][1] if len(ranked) > 1 else 0.0
            intent = top_intent
            # Multi-intent when two strong signals are close.
            is_multi = _is_multi(top_intent, scores)
            if faculty_key and top_intent in {"PROGRAM", "FACULTY", "LIST"}:
                strong = {
                    name for name, score in ranked
                    if score >= 0.6 * top_score
                }
                if strong <= {"PROGRAM", "FACULTY", "LIST"}:
                    is_multi = False
            confidence = _confidence(top_score, is_multi, normalized)

        # A canonical faculty alias is an unambiguous routing signal even for
        # conversational overview prompts such as "احكيلي عن كلية الطب".
        # Without this lift those prompts score only on the generic word
        # "كلية", miss the metadata filter, and can retrieve the university-
        # wide faculty directory instead of the requested faculty page.
        multi = is_multi if scores else False
        if faculty_key and not multi:
            confidence = max(confidence, 0.90)

        category, types = CATEGORY_MAP.get(intent, ("general", []))
        if faculty_key and not types:
            types = ["faculty", "program", "about", "home"]
        elif faculty_key and intent == "PERSON":
            # A dean is a person, but the authoritative evidence normally
            # lives on the faculty page rather than a generic people page.
            types = ["faculty", "program", "about", "home", "news"]

        result = RouteResult(
            intent=intent,
            category=category,
            category_types=list(types),
            faculty=faculty_key,
            faculty_id=faculty_id,
            language=language,
            confidence=round(confidence, 3),
            is_multi=multi,
            matched=sorted(set(matched_terms))[:12],
        )
        result._routing_ms = round((time.perf_counter() - start) * 1000, 3)
        return result

    # -- internals ----------------------------------------------------------

    def _detect_faculty(self, normalized: str) -> tuple[str | None, str | None]:
        normalized_low = normalized.lower()
        for key, alias in self._aliases:
            if alias in normalized_low:
                fid = _FACULTY_IDS.get(key)
                return key, fid
        return None, None


_FACULTY_IDS = {
    "business": "1",
    "law": "2",
    "engineering": "3",
    "computer-science-and-engineering": "4",
    "textile-science-and-engineering": "5",
    "science": "6",
    "medicine": "7",
    "dentistry": "8",
    "pharmacy": "9",
    "social-and-human-sciences": "10",
    "applied-health-sciences": "12",
    "nursing": "13",
    "graduate-studies": "14",
    "mass-media-and-communication": "16",
    "physical-therapy": "21",
}

# Tie-break order: more specific intents win over generic ones on equal scores
# (e.g. "برامج Faculty of X" should be PROGRAM, not FACULTY).
_TIE_PRIORITY = (
    "ADMISSION", "TUITION", "SCHOLARSHIP", "LOCATION", "CONTACT", "PRESIDENT",
    "EVENTS", "NEWS", "REGULATION", "COMPARISON", "ADMINISTRATION",
    "PROGRAM", "FACULTY", "FACILITY", "PERSON", "FAQ", "LIST",
    "FACT", "GENERAL",
)


def _is_mixed(text: str) -> bool:
    ar = sum(1 for ch in text if "\u0600" <= ch <= "\u06FF")
    en = sum(1 for ch in text if ch.isascii() and ch.isalpha())
    return bool(ar) and bool(en)


def _is_multi(intent: str, scores: dict[str, float]) -> bool:
    if not scores:
        return False
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    if len(ranked) < 2:
        return False
    return ranked[1][1] >= 0.6 * ranked[0][1]


def _confidence(score: float, multi: bool, normalized: str) -> float:
    """Map raw keyword weight to a 0..1 confidence estimate."""
    conf = 0.35 + min(0.6, score * 0.12)
    if any(m in normalized for m in _STRONG_MARKERS):
        conf += 0.08
    if multi:
        conf = min(conf, 0.72)  # multi-intent is never high-confidence
    return min(1.0, conf)

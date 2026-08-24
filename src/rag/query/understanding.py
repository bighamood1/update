"""Lightweight, deterministic query-understanding layer.

Runs BEFORE retrieval and produces a structured ``QueryUnderstanding`` with
everything the pipeline needs to route, normalize, expand and split the user
question — without ever calling the LLM. Reuses the existing rule-based router
(``rag.routing.router.QueryRouter``) and the existing normalizer
(``rag.retrieval.query_normalizer``).

This layer is intentionally cheap: routing is keyword/substring based and
entity extraction is regex-driven. The expensive LLM is reserved for
generation, never for classification.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..config import get_config
from ..retrieval.query_normalizer import is_arabic, normalize_query
from ..routing.rules import CATEGORY_MAP
from ..routing.router import QueryRouter
from ..routing.schemas import RouteResult

# Faculty key -> canonical display name / topic used for topic detection.
_FACULTY_TOPICS: dict[str, str] = {
    "business": "business",
    "law": "law",
    "engineering": "engineering",
    "computer-science-and-engineering": "computer_science",
    "textile-science-and-engineering": "textiles",
    "science": "science",
    "medicine": "medicine",
    "dentistry": "dentistry",
    "pharmacy": "pharmacy",
    "social-and-human-sciences": "social_sciences",
    "applied-health-sciences": "applied_health_sciences",
    "nursing": "nursing",
    "graduate-studies": "graduate_studies",
    "mass-media-and-communication": "media",
    "physical-therapy": "physical_therapy",
}

# Intent -> (topic, subtopic). Topics are stable slugs used for analytics,
# cache compatibility and (later) retrieval memory grouping.
_TOPIC_MAP: dict[str, tuple[str, str]] = {
    "ADMISSION": ("admission", "requirements"),
    "TUITION": ("tuition", "fees"),
    "SCHOLARSHIP": ("scholarship", "funding"),
    "REGULATION": ("regulation", "rules"),
    "TRANSFER": ("transfer", "requirements"),
    "LOCATION": ("location", "address"),
    "CONTACT": ("contact", "communication"),
    "PERSON": ("leadership", "officials"),
    "ADMINISTRATION": ("governance", "council"),
    "FACULTY": ("faculties", "overview"),
    "PROGRAM": ("programs", "overview"),
    "LIST": ("catalog", "list"),
    "COMPARISON": ("comparison", "difference"),
    "NEWS": ("news", "events"),
    "FACILITY": ("facilities", "services"),
    "FAQ": ("faq", "general"),
    "FACT": ("general", "fact"),
    "GENERAL": ("general", "general"),
}

# Explicit transfer markers (Arabic + English). Used to refine the
# REGULATION intent into TRANSFER when the question is about transferring.
_TRANSFER_MARKERS = (
    "transfer", "تحويل", "التحويل", "حول", "حويل", "نقل",
)

_NUMBER_RE = re.compile(r"\b\d[\d,.]*\b")
_YEAR_RE = re.compile(r"\b(20\d{2}|14\d{2})\b")
_DATE_WORDS = (
    "2024", "2025", "2026", "2027", "2028",
    "المتاح", "الفصل", "الترم", "semester", "term", "year",
)

# Splitter cues for multi-intent detection (very conservative).
_CONJUNCTIONS = (" وكما", "وكمان", "وما ", " وما ", " و ما ", " and also ",
                 " and what ", " and how ", " و ايه", " واي", " ولا ")


def _detect_language(text: str) -> str:
    if any("\u0600" <= ch <= "\u06FF" for ch in text) and any(
        ch.isascii() and ch.isalpha() for ch in text
    ):
        return "mixed"
    if is_arabic(text):
        return "ar"
    return "en"


def _extract_entities(text: str) -> list[str]:
    """Return a short, ordered list of salient entities (no LLM)."""
    entities: list[str] = []
    seen: set[str] = set()
    for m in _YEAR_RE.findall(text):
        if m not in seen:
            seen.add(m)
            entities.append(f"year:{m}")
    for m in _NUMBER_RE.findall(text):
        if m not in seen:
            seen.add(m)
            entities.append(f"number:{m}")
    # Faculty names are captured by the router (aliases) already; surface the
    # canonical key as an entity for the cache/memory layers.
    return entities


@dataclass
class QueryUnderstanding:
    """Structured, internal understanding of a user question.

    This metadata is used ONLY internally (routing, retrieval, caching,
    analytics). It is never exposed to the GUI.
    """

    original_question: str
    normalized_question: str
    language: str = "en"
    intent: str = "FACT"
    category: str = "general"
    faculty: str | None = None
    faculty_id: str | None = None
    topic: str = "general"
    subtopic: str = "general"
    entities: list[str] = field(default_factory=list)
    is_multi_intent: bool = False
    confidence: float = 0.0
    route: RouteResult | None = None

    def to_dict(self) -> dict:
        return {
            "original_question": self.original_question,
            "normalized_question": self.normalized_question,
            "language": self.language,
            "intent": self.intent,
            "category": self.category,
            "faculty": self.faculty,
            "faculty_id": self.faculty_id,
            "topic": self.topic,
            "subtopic": self.subtopic,
            "entities": list(self.entities),
            "is_multi_intent": self.is_multi_intent,
            "confidence": round(self.confidence, 3),
        }


def _resolve_topic(intent: str, faculty: str | None) -> tuple[str, str]:
    topic, subtopic = _TOPIC_MAP.get(intent, ("general", "general"))
    if faculty:
        subtopic = _FACULTY_TOPICS.get(faculty, subtopic)
    return topic, subtopic


def _refine_intent(
    intent: str, normalized: str, faculty: str | None = None
) -> str:
    """Refine coarse intents into more specific ones when markers exist."""
    if intent in ("REGULATION", "ADMISSION") and any(
        m in normalized for m in _TRANSFER_MARKERS
    ):
        return "TRANSFER"
    program_marker = any(
        m in normalized.lower()
        for m in (
            "تخصص", "تخصصات", "برنامج", "برامج", "قسم", "اقسام", "أقسام",
            "program", "programs", "department", "departments", "major", "majors",
        )
    )
    if program_marker and (
        intent == "FACULTY"
        or (faculty and intent in {"LIST", "LOCATION", "FACT", "GENERAL"})
    ):
        return "PROGRAM"
    return intent


def _has_conjunction(text: str) -> bool:
    low = text.lower()
    return any(c in low for c in _CONJUNCTIONS)


def understand(question: str, router: QueryRouter | None = None) -> QueryUnderstanding:
    """Build a ``QueryUnderstanding`` for a question (never raises)."""
    cfg = get_config()
    router = router or QueryRouter()
    text = (question or "").strip()

    if not text:
        return QueryUnderstanding(
            original_question=text,
            normalized_question="",
            confidence=0.0,
            route=router.route(""),
        )

    route = router.route(text)
    normalized = normalize_query(text)
    intent = _refine_intent(route.intent, normalized, route.faculty)
    if not get_config().get("query_understanding_enabled", True):
        intent = route.intent
    elif intent != route.intent:
        route.intent = intent
        route.category, category_types = CATEGORY_MAP.get(intent, ("general", []))
        route.category_types = list(category_types)
    language = _detect_language(text)
    topic, subtopic = _resolve_topic(intent, route.faculty)
    multi = bool(route.is_multi) or (
        cfg.get("multi_intent_enabled", True) and _has_conjunction(text)
    )

    return QueryUnderstanding(
        original_question=text,
        normalized_question=normalized,
        language=language,
        intent=intent,
        category=route.category,
        faculty=route.faculty,
        faculty_id=route.faculty_id,
        topic=topic,
        subtopic=subtopic,
        entities=_extract_entities(text),
        is_multi_intent=multi,
        confidence=route.confidence,
        route=route,
    )

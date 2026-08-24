"""Router result schema."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RouteResult:
    """Output of the deterministic query router.

    Fields:
        intent: coarse intent label (FACT, LIST, FACULTY, PROGRAM, ...).
        category: coarse content bucket used to build optional Chroma filters.
        category_types: concrete content_type values that back ``category``.
        faculty: canonical faculty key (e.g. "medicine") or None.
        faculty_id: numeric faculty id from the NMU URL schema or None.
        language: dominant script ("ar" | "en" | "mixed").
        confidence: 0..1 rule-based certainty.
        is_multi: True when several intents/categories match (multi-intent).
        matched: list of raw rule tokens matched (diagnostics only).
    """

    intent: str = "FACT"
    category: str = "general"
    category_types: list[str] = field(default_factory=list)
    faculty: str | None = None
    faculty_id: str | None = None
    language: str = "en"
    confidence: float = 0.0
    is_multi: bool = False
    matched: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "intent": self.intent,
            "category": self.category,
            "faculty": self.faculty,
            "faculty_id": self.faculty_id,
            "language": self.language,
            "confidence": round(self.confidence, 3),
            "is_multi": self.is_multi,
        }
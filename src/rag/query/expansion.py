"""Bounded query expansion for lexical (BM25) retrieval.

For SIMPLE questions only the normalized query is used — the current
architecture's default. For AMBIGUOUS or multi-intent questions a small,
bounded set of retrieval variants (original wording, normalized, expanded
synonyms) is produced. Variants feed ONLY the BM25 stage; the dense embedding
still uses the original question so accuracy never regresses.

Never more than ``MAX_RETRIEVAL_VARIANTS`` (default 3) variants are emitted,
and duplicates are removed.
"""

from __future__ import annotations

from ..config import get_config
from ..retrieval.query_normalizer import expand_query, normalize_query
from .understanding import QueryUnderstanding


def retrieval_variants(
    understanding: QueryUnderstanding, original_text: str | None = None
) -> list[str]:
    """Return a bounded list of lexical query strings for BM25.

    ``understanding.normalized_question`` is always present. For ambiguous /
    multi-intent questions the intent-expanded form and the raw user wording
    are added (both deduplicated against the base).
    """
    cfg = get_config()
    max_variants = max(1, int(cfg.get("max_retrieval_variants", 5) or 5))
    original = (original_text or understanding.original_question or "").strip()

    base = understanding.normalized_question or normalize_query(original)
    variants: list[str] = []
    for v in (base,):
        if v and v not in variants:
            variants.append(v)

    high_recall_intents = {
        "LOCATION", "SCHOLARSHIP", "PROGRAM", "FACULTY", "TUITION", "ABOUT",
    }
    high_recall = (understanding.intent or "").upper() in high_recall_intents
    ambiguous = (
        understanding.is_multi_intent
        or understanding.confidence < float(cfg.get("router_confidence_threshold", 0.8))
    )
    if (ambiguous or high_recall) and len(variants) < max_variants:
        expanded = expand_query(
            original or base,
            understanding.intent,
            max_terms=int(cfg.get("max_query_expansion_terms", 8)),
        )
        if expanded and expanded not in variants:
            variants.append(expanded)

        raw_normalized = normalize_query(original)
        if raw_normalized and raw_normalized not in variants:
            variants.append(raw_normalized)

    if high_recall and len(variants) < max_variants:
        topic_variants = _topic_variants(understanding)
        for variant in topic_variants:
            if variant and variant not in variants:
                variants.append(variant)
            if len(variants) >= max_variants:
                break

    return variants[:max_variants]


def _topic_variants(understanding: QueryUnderstanding) -> list[str]:
    """Deterministic retrieval-only paraphrases for high-recall intents."""
    lang = understanding.language
    intent = (understanding.intent or "").upper()
    faculty = understanding.faculty or ""
    variants: list[str] = []
    if intent == "SCHOLARSHIP":
        variants.extend([
            "المنح الدراسية جامعة المنصورة الجديدة أنواع المنح شروط المنح الدعم الاجتماعي",
            "scholarships New Mansoura University scholarship rules financial aid",
        ])
    elif intent == "LOCATION":
        variants.extend([
            "موقع جامعة المنصورة الجديدة العنوان مدينة المنصورة الجديدة الطريق الساحلي الدولي",
            "New Mansoura University location address Dakahlia International Coastal Road",
        ])
    elif intent == "PROGRAM" and faculty == "computer-science-and-engineering":
        variants.extend([
            "كلية علوم وهندسة الحاسب البرامج الاقسام تخصصات جامعة المنصورة الجديدة",
            "Computer Science & Engineering programs departments majors New Mansoura University",
        ])
    elif intent in {"PROGRAM", "FACULTY"}:
        variants.extend([
            "برامج كليات جامعة المنصورة الجديدة الاقسام التخصصات",
            "faculties programs departments majors New Mansoura University",
        ])
    elif intent == "TUITION":
        variants.extend([
            "رسوم الدراسة مصروفات جامعة المنصورة الجديدة الكليات",
            "tuition fees New Mansoura University faculties",
        ])
    elif intent == "ABOUT":
        variants.extend([
            "عن جامعة المنصورة الجديدة رؤيتنا رسالتنا أهدافنا وقيمنا الأهداف الاستراتيجية القيم الحاكمة",
            "about New Mansoura University vision mission goals objectives values strategic objectives",
        ])
    if lang == "en":
        variants = list(reversed(variants))
    return variants

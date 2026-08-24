"""Query normalization and local keyword expansion.

Normalization fixes spelling/whitespace variants (Arabic hamza/alef forms,
tatweel, diacritics, doubled spacing) WITHOUT translating anything. The
normalized text is what feeds the lexical BM25 stage.

Expansion is a static, intent-aware synonym map (English + Arabic). It adds
a handful of related terms so lexical retrieval still finds passages that
phrase the concept differently (e.g. "faculties" -> "colleges"). Expansion is
purely local — no LLM is ever called — and only affects lexical retrieval;
the dense embedding still uses the original question.
"""

from __future__ import annotations

import re

_AR_DIACRITICS = re.compile(r"[\u064B-\u0652\u0670\u0640]")
_AR_NORMALIZE = str.maketrans(
    {"أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا", "ى": "ي", "ئ": "ي", "ؤ": "و"}
)

# Name-variant aliases for New Mansoura University. Mapping them to one
# canonical form means BM25 finds the same pages regardless of how the user
# writes the name (English abbreviation / Latin spelling / Arabic تاء مربوطة
# vs هاء, hamza-spelling, etc.). Applied BEFORE the general cleanup so the
# canonical terms then pass through standard normalization.
_ALIAS_SUBS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\bnmu\b", re.IGNORECASE), "New Mansoura University"),
    (re.compile(r"\bcse\b", re.IGNORECASE), "Computer Science & Engineering"),
    (re.compile(r"newmansoura", re.IGNORECASE), "New Mansoura University"),
    (re.compile(r"new\s*mansoura\s*univ(?:ersity)?", re.IGNORECASE),
     "New Mansoura University"),
    (re.compile(r"computer\s+science\s+(?:and|&)\s+engineering", re.IGNORECASE),
     "Computer Science & Engineering"),
    (re.compile(r"كليه\s*الحاسب", re.IGNORECASE), "كلية علوم وهندسة الحاسب"),
    (re.compile(r"كلية\s*الحاسب", re.IGNORECASE), "كلية علوم وهندسة الحاسب"),
    (re.compile(r"كلية\s*الحاسبات", re.IGNORECASE), "كلية علوم وهندسة الحاسب"),
    (re.compile(r"(?<!وهندسة\s)علوم\s*الحاسب", re.IGNORECASE), "علوم وهندسة الحاسب"),
    (re.compile(r"(?<!و)هندسة\s*الحاسب", re.IGNORECASE), "علوم وهندسة الحاسب"),
    (re.compile(r"جامعة\s*المنصوره\s*الجديده", re.IGNORECASE),
     "جامعة المنصورة الجديدة"),
    (re.compile(r"جامعة\s*المنصوره", re.IGNORECASE), "جامعة المنصورة"),
    (re.compile(r"المنصوره\s*الجديده", re.IGNORECASE), "المنصورة الجديدة"),
    (re.compile(r"جامعه", re.IGNORECASE), "جامعة"),
    (re.compile(r"كليه", re.IGNORECASE), "كلية"),
    (re.compile(r"المنصوره", re.IGNORECASE), "المنصورة"),
    (re.compile(r"الجديده", re.IGNORECASE), "الجديدة"),
]


def apply_aliases(text: str) -> str:
    """Replace NMU name variants with a canonical spelling (never raises)."""
    if not text:
        return text
    for pattern, canonical in _ALIAS_SUBS:
        text = pattern.sub(canonical, text)
    return text

# Intent -> expansion keywords (both languages). Kept small and specific so
# the fused candidate pool is not flooded with noise.
_EXPANSIONS: dict[str, list[str]] = {
    "FACULTY": ["faculty", "faculties", "college", "colleges", "كلية", "كليات"],
    "PROGRAM": [
        "program", "programs", "course", "courses", "department", "departments",
        "major", "majors", "برنامج", "برامج", "مقرر", "قسم", "أقسام",
        "اقسام", "تخصص",
    ],
    "LIST": ["list", "all", "available", "قائمة", "جميع", "متاحة"],
    "LOCATION": ["location", "located", "where", "address", "campus", "موقع", "مكان", "أين", "عنوان", "حرم"],
    "ADMISSION": ["admission", "requirements", "apply", "enrollment", "قبول", "التحاق", "شروط", "تسجيل"],
    "TUITION": [
        "tuition", "fees", "fee", "how much", "cost", "price", "expenses",
        "مصروفات", "مصاريف", "رسوم", "تكاليف", "تكلفة", "كام", "بكام",
    ],
    "SCHOLARSHIP": [
        "scholarship", "scholarships", "financial aid", "financial support",
        "grant", "grants", "funding", "student support",
        "منحة", "منح", "المنح الدراسية", "الدعم الاجتماعي", "التفوق",
        "خصم", "تمويل", "الطلاب المتفوقين",
    ],
    "PERSON": ["president", "dean", "head", "رئيس", "عميد"],
    "ADMINISTRATION": ["administration", "council", "board", "إدارة", "مجلس", "هيئة"],
    "REGULATION": ["regulation", "rules", "policy", "law", "قواعد", "لوائح", "سياسة", "قانون"],
    "FAQ": ["faq", "system", "study", "how", "كيف", "نظام"],
    "NEWS": ["news", "event", "latest", "أخبار", "فعاليات"],
    "CONTACT": ["contact", "address", "phone", "email", "اتصال", "تواصل", "عنوان", "هاتف"],
    "FACILITY": ["facility", "facilities", "library", "campus", "مرافق", "مكتبة"],
}


def normalize_query(question: str) -> str:
    """Return a normalized query suitable for lexical (BM25) matching."""
    if not question:
        return ""
    text = apply_aliases(question)
    text = re.sub(r"[\u064B-\u0652\u0670\u0640]", "", text)  # diacritics + tatweel
    text = text.translate(_AR_NORMALIZE)
    # Colloquial Egyptian "ايه"/"اية" -> "ما هي" before splitting.
    text = re.sub(r"(^|\s)اي[هة](?=\s)", " ما هي ", text, flags=re.IGNORECASE)
    text = re.sub(r"بتقدم", "تقدم توفر", text, flags=re.IGNORECASE)
    # Do not rewrite bare "موجودة" as a location signal: in questions such
    # as "إيه البرامج الموجودة في كلية..." it means "available", not
    # "located". Explicit words such as فين/أين/تقع still route locations.
    text = re.sub(r"فين", "اين موقع", text, flags=re.IGNORECASE)
    # Collapse whitespace and normalize punctuation spacing.
    text = re.sub(r"[^\w\s\u0600-\u06FF؟?،,.:!']+", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def expand_query(question: str, intent: str, max_terms: int = 8) -> str:
    """Return the original query plus a small set of intent-aligned synonyms.

    The expansion is appended to the normalized query text; it only feeds the
    BM25 stage, never the embedding.
    """
    if not question:
        return question
    base = normalize_query(question)
    keywords = _EXPANSIONS.get(intent, [])
    if not keywords:
        return base
    # Only add keywords the query does not already contain (as substrings).
    low = base.lower()
    added = [k for k in keywords if k.lower() not in low][: max_terms]
    if not added:
        return base
    return f"{base} {' '.join(added)}"


def is_arabic(text: str) -> bool:
    """True if the dominant script of ``text`` is Arabic."""
    if not text:
        return False
    ar = sum(1 for ch in text if "\u0600" <= ch <= "\u06FF")
    other = sum(1 for ch in text if ch.isalpha() and not ("\u0600" <= ch <= "\u06FF"))
    return ar > 0 and ar >= other

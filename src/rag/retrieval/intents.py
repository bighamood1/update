"""Lightweight, rule-based query intent classification.

No external model is called: intent is derived from the question text using
language-aware keyword/regex rules (English + Arabic). Detected intent is
used to adjust retrieval behaviour (candidate pool size, source preference,
list expansion).

Supported intents:
    FACT, LIST, FACULTY, PROGRAM, LOCATION, PERSON, ADMINISTRATION,
    ADMISSION, REGULATION, FAQ, COMPARISON, UNKNOWN
"""

from __future__ import annotations

import re

# --------------------------------------------------------------------------
# Intent rule definitions: (regex, intent). First match wins; more specific
# rules are listed first.
# --------------------------------------------------------------------------

_COMPARISON = re.compile(
    r"\b(compare|comparison|difference|differences|vs\.?|versus|أيهما|أفضل|مقارنة|الفرق|بين)\b",
    re.IGNORECASE,
)

_LOCATION = re.compile(
    r"\b(located|location|where|address|في.?ن|أين|العنوان|موقع)\b",
    re.IGNORECASE,
)

_PERSON = re.compile(
    r"\b(who|president|dean|minister|head of|رئيس|عميد|من هو|من يكون)\b",
    re.IGNORECASE,
)

_ADMINISTRATION = re.compile(
    r"\b(board of trustees|council|administration|management|إدارة|مجلس الأمناء|مجلس الجامعة|الإدارة)\b",
    re.IGNORECASE,
)

_ADMISSION = re.compile(
    r"\b(admission|apply|application|enroll|requirements|acceptance|secondary school|"
    r"قبول|تسجيل|الالتحاق|شروط|ثانوية|تقديم)\b",
    re.IGNORECASE,
)

_REGULATION = re.compile(
    r"\b(regulation|rules|transfer|policy|law|code of ethics|قواعد|التحويل|لوائح|سياسة|قانون|الميثاق)\b",
    re.IGNORECASE,
)

_FACULTY = re.compile(
    r"\b(faculty|faculties|college|colleges|كلية|كليات)\b",
    re.IGNORECASE,
)

_PROGRAM = re.compile(
    r"\b(program|programs|course|courses|degree|degrees|برنامج|برامج|مقرر|مقررات|درجة)\b",
    re.IGNORECASE,
)

_FAQ = re.compile(
    r"\b(faq|frequently asked|how do|how can|what is the study system|"
    r"الأسئلة الشائعة|كيف|ما هو نظام|هل يمكن)\b",
    re.IGNORECASE,
)

# List markers: request the complete set of items.
_LIST = re.compile(
    r"\b(list|list all|all the|all of|how many|name the|what .*available|enumerate|"
    r"قائمة|اذكر|جميع|كل ال|عدد|ما هي كل|ما هي كليات|ما هي برامج)\b",
    re.IGNORECASE,
)

# Out-of-domain / geography / politics / generic world-knowledge cues.
_UNKNOWN = re.compile(
    r"\b(egypt's population|president of the united states|president of the usa|"
    r"capital of|discovered america|population of|weather|sports|football|"
    r"تعداد|عدد سكان|رئيس الولايات المتحدة|عاصمة|اكتشف أمريكا|طقس)\b",
    re.IGNORECASE,
)

_INTENT_ORDER = [
    (_UNKNOWN, "UNKNOWN"),
    (_COMPARISON, "COMPARISON"),
    (_LOCATION, "LOCATION"),
    (_PERSON, "PERSON"),
    (_ADMINISTRATION, "ADMINISTRATION"),
    (_ADMISSION, "ADMISSION"),
    (_REGULATION, "REGULATION"),
    (_FAQ, "FAQ"),
    (_PROGRAM, "PROGRAM"),
    (_FACULTY, "FACULTY"),
    (_LIST, "LIST"),
]


def classify_intent(question: str) -> str:
    """Return the detected intent for a question (never raises)."""
    q = (question or "").strip()
    if not q:
        return "UNKNOWN"
    for pattern, intent in _INTENT_ORDER:
        if pattern.search(q):
            return intent
    return "FACT"


def is_list_intent(intent: str) -> bool:
    """True for intents that should trigger list retrieval behaviour."""
    return intent in {"LIST", "FACULTY", "PROGRAM"}
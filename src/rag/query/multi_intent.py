"""Conservative multi-intent question splitting.

A question is only split when the evidence is strong: a conjunction directly
followed by a question word ("وما", "وكمان", "and what", ...) OR by a clear
intent keyword ("مصاريف", "قبول", "شروط", "tuition", "admission", ...), AND
every resulting part still looks like a standalone question. Otherwise the
original question is returned unchanged (single intent).

Splitting only affects RETRIEVAL (each part is searched separately and the
evidence is merged). The LLM still receives the original full question and
produces one coherent answer covering all parts.
"""

from __future__ import annotations

import re

from .understanding import QueryUnderstanding

# (regex, capture-group-start) pairs used to find safe split points.
# Arabic: a whitespace "و" followed by a question word or intent keyword.
# NOTE: the lookahead MUST use a non-capturing group, otherwise re.split()
# would emit the captured word as its own element and every split would fail
# the "looks like a question" check.
_AR_SPLIT = re.compile(r"\s+و(?=(?:ما|إيه|ايه|هل|كم|من|أين|كيف|مصاريف|رسوم|"
                       r"شروط|قبول|كليات|برامج|منحة|منح|تحويل|أقساط|تكاليف)\s)")
# English: " and " followed by a question word or intent keyword.
_EN_SPLIT = re.compile(r"\s+and(?=\s+(?:what|which|who|where|when|how|why|is|are|"
                       r"tuition|admission|requirements|faculties|programs|"
                       r"scholarship|fees|transfer)\b)")

# A part must contain at least one of these to be kept as a question.
_QUESTION_WORDS = (
    "ما", "ايه", "إيه", "من", "أين", "كيف", "متى", "كم", "هل", "اذكر", "قائمة",
    "what", "which", "who", "where", "when", "how", "why", "how many",
    "list", "name", "is", "are",
)

_MAX_PARTS = 3


def _looks_like_question(part: str) -> bool:
    p = (part or "").strip()
    if len(p) < 4:
        return False
    low = p.lower()
    if any(q in low for q in _QUESTION_WORDS):
        return True
    return bool(re.search(r"[؟?]|^كليات|^مصاريف|^رسوم|^شروط|^tuition|^admission", p.lower()))


def split_question(
    question: str, understanding: QueryUnderstanding
) -> list[str]:
    """Return a list of sub-questions (or ``[original]`` when no safe split)."""
    text = (question or "").strip()
    if not text:
        return [""]
    if not understanding.is_multi_intent:
        return [text]

    parts = _AR_SPLIT.split(text)
    if len(parts) < 2:
        parts = _EN_SPLIT.split(text)
    if len(parts) < 2:
        return [text]

    cleaned = [p.strip() for p in parts if p.strip()]
    cleaned = [c if c[-1] in ("؟", "?") else c + "؟" for c in cleaned]
    if not (2 <= len(cleaned) <= _MAX_PARTS):
        return [text]
    if not all(_looks_like_question(c) for c in cleaned):
        return [text]
    return cleaned
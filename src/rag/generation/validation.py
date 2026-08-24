"""Deterministic post-generation answer validation (no LLM).

Checks that a generated answer is usable:
- non-empty
- cites no fabricated URLs (every URL in the answer must be a retrieved one)
- does not silently contradict the retrieved evidence set

If the evidence is insufficient, callers receive a controlled refusal in the
user's language instead of a hallucinated or truncated answer.
"""

from __future__ import annotations

import re

from ..schemas.documents import RetrievedChunk

_URL_RE = re.compile(r"https?://[^\s)\]}>'\"]+", re.IGNORECASE)
_THINK_BLOCK_RE = re.compile(r"<think\b[^>]*>.*?</think>", re.IGNORECASE | re.DOTALL)
_LEADING_REASONING_RE = re.compile(
    r"^\s*(first|starting with|looking at|based on (?:the )?evidence|wait|"
    r"therefore|okay|ok|let'?s|let me|i need|i should|"
    r"we need|the user (?:is asking|asked)|the question asks|"
    r"according to the instructions|therefore, the answer should be|"
    r"أحتاج|سأقوم|دعني|يجب أن|السؤال يطلب)\b",
    re.IGNORECASE,
)
_REASONING_MARKERS = (
    "okay, i need", "ok, i need", "let me", "the question asks",
    "the user is asking", "according to the instructions", "<think", "</think>",
)
_SOURCE_LEAKAGE_RE = re.compile(
    r"(\bsource\s+\d+\b|\bevidence items?\b|"
    r"according to (?:the )?(?:retrieved )?(?:context|sources?|evidence)|"
    r"the context (?:says|states|mentions)|"
    r"retrieved context|end of context)",
    re.IGNORECASE,
)

REFUSAL_EN = (
    "I couldn't find enough information in the official NMU knowledge base "
    "to answer this reliably."
)
REFUSAL_AR = (
    "لم أتمكن من العثور على معلومات كافية في قاعدة المعرفة الرسمية "
    "لجامعة المنصورة الجديدة للإجابة بشكل موثوق."
)


def _is_arabic_dominant(text: str) -> bool:
    if not text:
        return False
    ar = sum(1 for ch in text if "\u0600" <= ch <= "\u06FF")
    other = sum(1 for ch in text if ch.isalpha() and not ("\u0600" <= ch <= "\u06FF"))
    return ar > 0 and ar >= other


def refusal_text(language: str = "en") -> str:
    return REFUSAL_AR if (language or "").lower() == "ar" else REFUSAL_EN


def normalize_markdown_structure(text: str) -> str:
    """Lightweight, content-preserving markdown cleanup (cosmetic only).

    - trims outer whitespace,
    - collapses runs of blank lines,
    - ensures headings sit on their own line separated by a blank line,
    - guarantees exactly one trailing newline.

    Nothing here rewrites words or reorders content; it only makes the LLM's
    markdown render cleanly in the GUI (headings, lists, paragraphs).
    """
    if not text:
        return ""
    cleaned = re.sub(r"[ \t]+$", "", text, flags=re.MULTILINE)  # trailing spaces
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned.strip())
    # Blank line before ATX headings that follow non-blank content.
    cleaned = re.sub(r"(?<!\n)\n(?=#{1,6}\s+\S)", "\n\n", cleaned)
    return cleaned.rstrip() + "\n"


def strip_reasoning_artifacts(text: str) -> tuple[str, bool]:
    """Remove leaked chain-of-thought preambles while preserving the answer.

    Some local reasoning models occasionally return a visible planning block
    before the final answer despite prompt instructions. This sanitizer is
    intentionally conservative: it removes XML-style think blocks and leading
    meta-reasoning lines, but it does not rewrite substantive answer content.
    """
    if not text:
        return "", False
    original = text
    cleaned = _THINK_BLOCK_RE.sub("", text)
    lines = cleaned.splitlines()
    while lines and (
        not lines[0].strip() or _LEADING_REASONING_RE.search(lines[0])
    ):
        lines.pop(0)
    cleaned = "\n".join(lines).strip()
    return cleaned, cleaned != original.strip()


def validate_answer(
    answer: str,
    sources: list[dict],
    retrieved: list[RetrievedChunk] | None = None,
    question_language: str = "en",
) -> dict:
    """Validate an answer; return ``{ok, issues, cleaned}``.

    ``cleaned`` is the answer with any fabricated URLs removed; callers should
    use it rather than the raw answer when issues are found.
    """
    issues: list[str] = []
    text, stripped_reasoning = strip_reasoning_artifacts(answer or "")
    text = text.strip()

    if not text:
        return {
            "ok": False,
            "issues": ["empty_answer", *([] if not stripped_reasoning else ["reasoning_artifact_stripped"])],
            "cleaned": "",
        }
    if stripped_reasoning:
        issues.append("reasoning_artifact_stripped")

    # 1. Fabricated-URL check: any URL must come from the retrieved evidence.
    retrieved_urls = {
        (c.source_url or "").rstrip("/").lower() for c in (retrieved or [])
    }
    for s in sources:
        if s.get("url"):
            retrieved_urls.add(str(s["url"]).rstrip("/").lower())

    urls = _URL_RE.findall(text)
    fabricated = [u for u in urls if u.rstrip("/").lower() not in retrieved_urls]
    if fabricated:
        issues.append(f"fabricated_url:{len(fabricated)}")
        for bad in fabricated:
            text = text.replace(bad, "")

    # 2. Language coherence (soft check — never fails alone).
    if question_language == "ar" and not _is_arabic_dominant(text):
        issues.append("language_mismatch")
    elif question_language == "en" and _is_arabic_dominant(text):
        issues.append("language_mismatch")

    if any(marker in text.lower() for marker in _REASONING_MARKERS):
        issues.append("reasoning_artifact_remaining")
    if _SOURCE_LEAKAGE_RE.search(text):
        issues.append("source_or_context_leakage")
    if _has_excessive_repetition(text):
        issues.append("excessive_repetition")

    text = re.sub(r"\s{2,}", " ", text).strip()
    if not text:
        issues.append("emptied_after_cleanup")
        return {"ok": False, "issues": issues, "cleaned": ""}

    # Meta-reasoning, source leakage, and a wholly wrong answer language must
    # never reach the GUI. Mixed Arabic/English terminology still passes the
    # dominance check above; only genuinely mismatched responses are rejected.
    hard_issues = [
        i for i in issues
        if i.startswith("fabricated_url")
        or i in {
            "emptied_after_cleanup",
            "language_mismatch",
            "reasoning_artifact_remaining",
            "source_or_context_leakage",
        }
    ]
    return {"ok": len(hard_issues) == 0, "issues": issues, "cleaned": text}


def completeness_issues(
    answer: str,
    retrieved: list[RetrievedChunk] | None,
    *,
    question: str,
    intent: str | None = None,
    is_multi_intent: bool = False,
) -> list[str]:
    """Return soft completeness issues for list/multi-item questions.

    This is intentionally generic: it looks for list-like evidence density and
    compares it with the answer's visible item coverage. It never names NMU
    faculties, scholarships, fees, or any other project-specific facts.
    """
    text = (answer or "").strip()
    if not text:
        return []
    if not _looks_completeness_sensitive(question, intent, is_multi_intent):
        return []

    evidence_items = _evidence_item_count(retrieved or [])
    if evidence_items < 4:
        return []
    answer_items = _answer_item_count(text)
    if answer_items >= min(evidence_items, 4):
        return []
    if answer_items >= max(3, evidence_items // 2):
        return []
    return [f"incomplete_evidence_coverage:{answer_items}/{evidence_items}"]


def _has_excessive_repetition(text: str) -> bool:
    """Conservative repetition detector for generated answers."""
    units = [
        re.sub(r"\W+", " ", part, flags=re.UNICODE).strip().lower()
        for part in re.split(r"(?<=[.!؟])\s+|\n+", text or "")
    ]
    units = [u for u in units if len(u) > 24]
    if len(units) < 3:
        return False
    counts: dict[str, int] = {}
    for unit in units:
        counts[unit] = counts.get(unit, 0) + 1
    return max(counts.values(), default=0) >= 2


def _looks_completeness_sensitive(
    question: str, intent: str | None, is_multi_intent: bool
) -> bool:
    q = (question or "").lower()
    if is_multi_intent:
        return True
    if (intent or "").upper() in {"LIST", "FACULTY", "PROGRAM", "SCHOLARSHIP", "COMPARISON"}:
        return True
    markers = (
        "all", "available", "list", "types", "categories", "departments",
        "programs", "faculties", "colleges", "fees", "كل", "جميع", "ما هي",
        "اذكر", "قائمة", "انواع", "أنواع", "اقسام", "أقسام", "برامج",
        "كليات", "رسوم",
    )
    return any(
        bool(re.search(r"(?<![\u0600-\u06FF])كل(?![\u0600-\u06FF])", q))
        if m == "كل" else m in q
        for m in markers
    )


def _evidence_item_count(chunks: list[RetrievedChunk]) -> int:
    seen: set[str] = set()
    for chunk in chunks:
        text = re.sub(r"\s*اعرف المزيد\s*", "\n", chunk.text or "")
        for line in re.split(r"[\n\r|;•]+", text):
            item = re.sub(r"^\s*(?:[-*]|\d+[.)]|[A-Za-z]\))\s*", "", line).strip()
            item = re.sub(r"\s+", " ", item)
            if not item or len(item) < 3 or len(item) > 140:
                continue
            # Long prose sentences are evidence, but not countable list items.
            if len(item.split()) > 14 and not re.search(r"[:،,]\s*\S+", item):
                continue
            key = re.sub(r"\W+", " ", item, flags=re.UNICODE).strip().lower()
            if key:
                seen.add(key)
    return len(seen)


def _answer_item_count(answer: str) -> int:
    bullet_like = [
        line for line in (answer or "").splitlines()
        if re.match(r"\s*(?:[-*]|\d+[.)])\s+\S+", line)
    ]
    if bullet_like:
        return len({
            re.sub(r"\W+", " ", line, flags=re.UNICODE).strip().lower()
            for line in bullet_like
        })
    sentences = [
        s.strip() for s in re.split(r"(?<=[.!؟])\s+|\n+", answer or "")
        if s.strip()
    ]
    return len(sentences)


# ---------------------------------------------------------------------------
# Deterministic intent-relevance guard (no LLM involved).
#
# The semantic cache / retrieval can surface the RIGHT entity but the WRONG
# question (e.g. "أين تقع جامعة المنصورة الجديدة؟" retrieving the founding
# decree of the university). This gate rejects only answers that clearly talk
# about a DIFFERENT topic than the routed intent — it is deliberately
# conservative: marker-free answers pass, explicit insufficiency is accepted.
# ---------------------------------------------------------------------------

_REFUSAL_MARKERS = (
    "couldn't find", "could not find", "cannot confirm", "can't confirm",
    "not enough information", "unable to", "لا تكفي", "لا توجد",
    "لم أتمكن", "لم يتم العثور", "غير متوفر", "لا يمكنني",
)

_ESTABLISHMENT_MARKERS = (
    "قرار", "مرسوم", "إنشاء", "تأسيس", "تأسست", "أنشئت", "صدر",
    "decree", "established", "founded", "establishment", "issuance",
)

_INTENT_TOPIC_MARKERS: dict[str, tuple[str, ...]] = {
    "LOCATION": (
        "تقع", "موقع", "عنوان", "مكان", "مدينة", "محافظة", "حرم", "خريطة",
        "الطريق", "located", "location", "address", "city", "governorate",
        "campus", "coastal",
    ),
    "CONTACT": (
        "هاتف", "اتصال", "تواصل", "بريد", "رقم", "phone", "email",
        "contact", "telephone",
    ),
    "PERSON": (
        "رئيس", "عميد", "مدير", "أمين", "president", "dean", "director",
        "secretary",
    ),
    "TUITION": (
        "رسوم", "مصروفات", "مصاريف", "تكلفة", "تكاليف", "tuition", "fees",
        "cost", "expenses",
    ),
    "ADMISSION": (
        "قبول", "التحاق", "شروط", "تسجيل", "تنسيق", "تقديم", "admission",
        "admissions", "apply", "application", "enrollment", "requirements",
        "eligibility",
    ),
    "SCHOLARSHIP": (
        "منحة", "منح", "scholarship", "scholarships", "grant",
    ),
    "FACULTY": (
        "كلية", "كليات", "faculty", "faculties", "college", "colleges",
    ),
    "PROGRAM": (
        "برنامج", "برامج", "قسم", "أقسام", "program", "programs", "major",
        "majors", "course",
    ),
    "REGULATION": (
        "قواعد", "لوائح", "سياسة", "قانون", "قوانين", "regulation",
        "regulations", "rules", "policy", "law",
    ),
    "NEWS": (
        "أخبار", "فعاليات", "خبر", "news", "event", "events",
    ),
}

# Topic signals that clearly indicate a DIFFERENT topic. An answer for
# e.g. LOCATION that talks about the founding decree (and nothing about any
# location) is unmistakably off-topic and must be regenerated/refused.
_INTENT_MISMATCH_MARKERS: dict[str, tuple[str, ...]] = {
    "LOCATION": _ESTABLISHMENT_MARKERS,
    "CONTACT": _ESTABLISHMENT_MARKERS,
    "PERSON": _ESTABLISHMENT_MARKERS,
    "TUITION": _ESTABLISHMENT_MARKERS,
    "ADMISSION": _ESTABLISHMENT_MARKERS,
    "SCHOLARSHIP": _ESTABLISHMENT_MARKERS,
}


def answer_relevance_ok(answer: str, intent: str | None) -> bool:
    """Return True when the answer plausibly addresses the routed intent.

    Only well-defined topic intents are gated. An answer is rejected ONLY when
    it (a) contains none of the intent's own topic markers, (b) contains no
    explicit insufficiency/refusal, and (c) contains strong markers of a
    different topic. Everything else passes, keeping the guard safe for
    legitimate free-form answers.
    """
    text = (answer or "").strip()
    if not text:
        return False
    intent = (intent or "").strip().upper()
    if intent not in _INTENT_TOPIC_MARKERS:
        return True  # generic/unknown intents are not gated
    low = text.lower()
    if any(m in low for m in _REFUSAL_MARKERS):
        return True
    if any(m in low for m in _INTENT_TOPIC_MARKERS[intent]):
        return True
    negatives = _INTENT_MISMATCH_MARKERS.get(intent, ())
    if any(m in low for m in negatives):
        return False
    return True

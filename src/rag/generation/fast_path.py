"""Deterministic fast answers for highly structured queries.

For a small set of intents (faculties list, university location, contact
info) the answer can be produced directly from authoritative indexed chunks —
no LLM call, so no multi-minute generation on CPU.

Activation is strict:
- the intent must match (LIST / FACULTY / PROGRAM / LOCATION / CONTACT),
- a qualifying AUTHORITATIVE chunk must be present in the retrieved set,
- the extracted content must be non-empty.

No facts are hardcoded here: everything is read from the retrieved chunks.
When the gate fails, ``try_fast_answer`` returns ``(None, [])`` and the
pipeline falls back to normal RAG generation.
"""

from __future__ import annotations

import re

from ..config import get_config
from ..schemas.documents import RetrievedChunk
from ..utils.logging_utils import get_logger

logger = get_logger(__name__)

# Canonical directory page that holds the complete faculty list.
_DIRECTORY_URL_HINT = "all-faculties-programs"

# Strong location markers first, then the bare university name. The bare name
# ("new mansoura" / "المنصورة الجديدة") appears inside non-location sentences
# (e.g. the establishment decree), so it is treated as a WEAK marker.
_LOCATION_MARKERS_EN = (
    "international coastal road",
    "dakahlia",
    "coastal road",
    "headquarters",
    "new mansoura",
)
_LOCATION_MARKERS_AR = (
    "الطريق الدولي",
    "الدقهلية",
    "المقر",
    "بجوار جهاز المدينة",
    "مدينة المنصورة الجديدة",
    "المنصورة الجديدة",
)

_WEAK_MARKERS_EN = ("new mansoura",)
_WEAK_MARKERS_AR = ("المنصورة الجديدة",)

# When only a WEAK marker matched, the sentence must also carry a real location
# signal — otherwise it is NOT a location answer (e.g. the decree text).
_LOCATION_SIGNAL_EN = (
    "located", "city", "governorate", "headquarters", "address", "road", "coastal",
)
_LOCATION_SIGNAL_AR = (
    "تقع", "يقع", "توجد", "مدينة", "محافظة", "الطريق", "المقر", "العنوان", "بجوار",
)

_PHONE_RE = re.compile(r"(?<!\d)(01[0125]\d{8}|\d{8,11})(?!\d)")
_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_SENTENCE_BOUNDARY = re.compile(r"[.!؟؟\n]")

# Guards that restrict the faculties fast path to genuine LIST questions.
_FACULTIES_LIST_QUESTION = re.compile(
    r"(list of faculties|all faculties|what faculties|how many faculties|"
    r"name (?:the )?faculties|faculties (?:are|in|of)|the faculties|"
    r"colleges (?:in|of|are)|ما هي كليات|جميع الكليات|قائمة الكليات|"
    r"عدد الكليات|اذكر الكليات|كليات الجامعة|كليات جامعة)",
    re.IGNORECASE,
)
_ACADEMIC_STRUCTURE_QUESTION = re.compile(
    r"(departments?|programs?|majors?|اقسام|أقسام|قسم|برامج|برنامج)",
    re.IGNORECASE,
)
_DEPARTMENT_LINE_RE = re.compile(r"^(?:قسم\s+.+|Department\s+of\s+.+)$", re.IGNORECASE)
_PROGRAM_LINE_RE = re.compile(r"^(?:برنامج\s+.+|Program\s+of\s+.+|.+\s+Program)$", re.IGNORECASE)
_FEE_RE = re.compile(r"^\d[\d,.\s]*$")

_AR_DEAN_RE = re.compile(
    r"(?P<title>[اأ]\s*\.?\s*د\s*\.?)\s*"
    r"(?P<name>[\u0600-\u06FF]+(?:\s+[\u0600-\u06FF]+){1,4})\s*,?\s*"
    r"عميد\s+(?:كلية|الكلية)\s+"
    r"(?P<faculty>[\u0600-\u06FF ]{2,60}?)"
    r"(?=\s*(?:[،,.]|و(?:إ|ا)?شراف|وفي|$))",
)
_EN_DEAN_RE = re.compile(
    r"(?P<title>Prof(?:essor)?\.?\s*(?:Dr\.?)?)\s*"
    r"(?P<name>[A-Z][A-Za-z'-]+(?:\s+[A-Z][A-Za-z'-]+){1,4})\s*,?\s*"
    r"Dean\s+of\s+(?:the\s+)?(?:Faculty|College)\s+of\s+"
    r"(?P<faculty>[A-Za-z &-]{2,60}?)"
    r"(?=\s*(?:[,.]|and\s+under|$))",
    re.IGNORECASE,
)
_AR_TITLED_NAME_RE = re.compile(
    r"[اأ]\s*\.?\s*د\s*\.?\s*/?\s*"
    r"(?P<name>[\u0600-\u06FF]+(?:\s+[\u0600-\u06FF]+){2,4})"
    r"(?=\s*(?:[|،,\n]|$))"
)
_AR_DEAN_BLOCK_RE = re.compile(
    r"[اأ]\s*\.?\s*د\s*\.?\s*/?\s*"
    r"(?P<name>[\u0600-\u06FF]+(?:\s+[\u0600-\u06FF]+){1,4})"
    r"\s*(?:[|\n])\s*عميد\s+(?:الكلية|كلية)"
)


def _faculty_tokens(value: str) -> set[str]:
    value = re.sub(r"[^\w\u0600-\u06FF]+", " ", value or "").lower()
    tokens: set[str] = set()
    for token in value.split():
        if token in {"كلية", "الكلية", "faculty", "college", "of", "the"}:
            continue
        if token.startswith("ال") and len(token) > 4:
            token = token[2:]
        tokens.add(token)
    return tokens


def _requested_dean_faculty(question: str) -> str:
    patterns = (
        r"عميد\s+(?:كلية|الكلية)\s+(.+?)(?=\s+(?:في|ب)?جامعة|[؟?]|$)",
        r"عميد\s+(.+?)(?=\s+(?:في|ب)?جامعة|[؟?]|$)",
        r"dean\s+of\s+(?:the\s+)?(?:faculty|college)(?:\s+of)?\s+(.+?)(?=\s+(?:at|in)\s+|[?]|$)",
    )
    for pattern in patterns:
        match = re.search(pattern, question or "", re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return ""


def _split_items(text: str) -> list[str]:
    """Split a directory-page text into individual list items."""
    normalized = re.sub(r"\s*اعرف المزيد\s*", "|", text)
    return [
        part.strip()
        for part in re.split(r"[|\n]+", normalized)
        if part.strip() and len(part.strip()) > 1
    ]


def _sentence_around(text: str, idx: int) -> str:
    """Return the sentence containing ``idx`` (bounded to ~700 chars).

    The start is the LAST sentence boundary before ``idx`` and the end is the
    FIRST boundary after it, so adjacent sentences are never pulled in.
    """
    n = len(text)
    window_start = max(0, idx - 300)
    start = window_start
    pos = window_start
    m = None
    while True:
        m = _SENTENCE_BOUNDARY.search(text, pos, idx)
        if not m:
            break
        start = m.end()
        pos = m.end()
    end = idx + 400
    m2 = _SENTENCE_BOUNDARY.search(text[idx : idx + 400])
    if m2:
        end = idx + m2.end()
    return text[start : min(end, n)].strip()


def _is_faculties_list_question(question: str) -> bool:
    return bool(_FACULTIES_LIST_QUESTION.search(question or ""))


# -- extractors ------------------------------------------------------------

def _faculties_answer(chunks: list[RetrievedChunk], language: str) -> tuple[str, list[RetrievedChunk]] | None:
    dirs = [c for c in chunks if _DIRECTORY_URL_HINT in (c.source_url or "")]
    if not dirs:
        return None
    lang_dirs = [c for c in dirs if (c.language or "en").lower() == language] or dirs
    items: list[str] = []
    for c in lang_dirs:
        items.extend(_split_items(c.text))
    unique: list[str] = []
    seen: set[str] = set()
    for item in items:
        key = item.strip().lower()
        if key and key != "اعرف المزيد" and key not in seen:
            seen.add(key)
            unique.append(item.strip())
    if len(unique) < 4:
        return None
    lead = (
        "The faculties of New Mansoura University are:"
        if language == "en"
        else "كليات جامعة المنصورة الجديدة هي:"
    )
    answer = f"{lead}\n" + "\n".join(f"- {item}" for item in unique)
    return answer, lang_dirs


def _location_answer(chunks: list[RetrievedChunk], language: str) -> tuple[str, list[RetrievedChunk]] | None:
    # Check markers of the question language first, then the other set: the
    # site mixes Arabic/English content, so an Arabic question may be answered
    # by an English about/contact page (and vice versa). The lead phrase still
    # follows the question language.
    marker_sets = [
        _LOCATION_MARKERS_AR if language == "ar" else _LOCATION_MARKERS_EN,
        _LOCATION_MARKERS_EN if language == "ar" else _LOCATION_MARKERS_AR,
    ]
    weak_sets = [
        _WEAK_MARKERS_AR if language == "ar" else _WEAK_MARKERS_EN,
        _WEAK_MARKERS_EN if language == "ar" else _WEAK_MARKERS_AR,
    ]
    signal_sets = [
        _LOCATION_SIGNAL_AR if language == "ar" else _LOCATION_SIGNAL_EN,
        _LOCATION_SIGNAL_EN if language == "ar" else _LOCATION_SIGNAL_AR,
    ]
    for c in chunks:
        if (c.content_type or "").lower() not in ("about", "contact", "home"):
            continue
        text = c.text
        haystack = text.lower()
        for markers, weak, signals in zip(marker_sets, weak_sets, signal_sets):
            for marker in markers:
                idx = haystack.find(marker)
                if idx < 0:
                    continue
                sentence = _sentence_around(text, idx)
                if len(sentence) < 8:
                    continue
                # A bare university-name match is only a location if the
                # sentence also has a real location signal (governorate, city,
                # road, headquarters, "تقع", ...). This stops the decree /
                # "منارة علمية في قلب الدلتا" sentences from being returned.
                if marker in weak and not any(
                    s in sentence.lower() for s in signals
                ):
                    logger.info(
                        "Fast path location skipped: weak marker %r in a "
                        "non-location sentence", marker,
                    )
                    continue
                if language == "en":
                    if "new mansoura university" in sentence.lower():
                        return sentence, [c]
                    return f"New Mansoura University is located here:\n{sentence}", [c]
                if "جامعة المنصورة الجديدة" in sentence:
                    return sentence, [c]
                return f"تقع جامعة المنصورة الجديدة في:\n{sentence}", [c]
    return None


def _contact_answer(chunks: list[RetrievedChunk], language: str) -> tuple[str, list[RetrievedChunk]] | None:
    phones: list[str] = []
    emails: list[str] = []
    address: str | None = None
    used: list[RetrievedChunk] = []
    for c in chunks:
        if (c.content_type or "").lower() != "contact":
            continue
        used.append(c)
        text = c.text
        phones.extend(_PHONE_RE.findall(text))
        emails.extend(m.lower() for m in _EMAIL_RE.findall(text))
        m = re.search(r"(?:our address|address)\s*[:.]?\s*([^\n]{5,140})", text, re.I)
        if m:
            address = m.group(1).strip()
        m2 = re.search(r"العنوان\s*[:.]?\s*([^\n]{5,140})", text)
        if m2:
            address = m2.group(1).strip()
    if not used:
        return None
    phones = list(dict.fromkeys(phones))
    emails = list(dict.fromkeys(emails))
    if not phones and not emails and not address:
        return None
    parts: list[str] = []
    if phones:
        parts.append(
            ("Phone: " if language == "en" else "الهاتف: ") + " - ".join(phones)
        )
    if emails:
        parts.append(
            ("Email: " if language == "en" else "البريد الإلكتروني: ") + ", ".join(emails)
        )
    if address:
        parts.append(
            ("Address: " if language == "en" else "العنوان: ") + address
        )
    return "\n".join(parts), used


def person_evidence_answer(
    question: str,
    chunks: list[RetrievedChunk],
    language: str,
) -> tuple[str, list[RetrievedChunk]] | None:
    """Extract a dean's name only when retrieved evidence states it explicitly.

    This is a post-generation safety fallback, not a canned fast answer: the
    RAG retrieval and Qwen reasoning run first. It is used only if generation
    is empty/malformed, and it never invents a person absent from the retrieved
    chunks. Ambiguous or conflicting names deliberately fall back to refusal.
    """
    q = question or ""
    asks_dean = bool(re.search(r"\b(?:dean|عميد)\b", q, re.IGNORECASE))
    asks_president = bool(re.search(r"(?:president|رئيس\s+(?:جامعة|الجامعة))", q, re.IGNORECASE))
    if not asks_dean and not asks_president:
        return None

    if asks_president and language == "ar":
        matches: dict[str, RetrievedChunk] = {}
        for chunk in chunks:
            text = chunk.text or ""
            if not re.search(r"(?:رئيس جامعة المنصورة الجديدة|الرئيس المؤسس)", text):
                continue
            for match in _AR_TITLED_NAME_RE.finditer(text):
                name = re.sub(r"\s+", " ", match.group("name")).strip()
                matches.setdefault(name, chunk)
        if len(matches) == 1:
            name, used = next(iter(matches.items()))
            return (
                f"رئيس جامعة المنصورة الجديدة وفقًا لبيانات الجامعة المتاحة هو أ.د. {name}.",
                [used],
            )
        return None

    if not asks_dean:
        return None

    preferred = [c for c in chunks if (c.language or "").lower() == language]
    search_groups = [preferred, chunks] if preferred else [chunks]
    requested_tokens = _faculty_tokens(_requested_dean_faculty(q))
    for group in search_groups:
        matches: list[tuple[str, str, RetrievedChunk]] = []
        seen: set[tuple[str, str]] = set()
        pattern = _AR_DEAN_RE if language == "ar" else _EN_DEAN_RE
        for chunk in group:
            for match in pattern.finditer(chunk.text or ""):
                name = re.sub(r"\s+", " ", match.group("name")).strip(" ،,.")
                faculty = re.sub(r"\s+", " ", match.group("faculty")).strip(" ،,.")
                key = (name.casefold(), faculty.casefold())
                if key and key not in seen:
                    seen.add(key)
                    matches.append((name, faculty, chunk))
            if language == "ar":
                for match in _AR_DEAN_BLOCK_RE.finditer(chunk.text or ""):
                    name = re.sub(r"\s+", " ", match.group("name")).strip(" ،,.")
                    faculty = (chunk.title or chunk.faculty or "").strip()
                    key = (name.casefold(), faculty.casefold())
                    if key and key not in seen:
                        seen.add(key)
                        matches.append((name, faculty, chunk))
        if matches and requested_tokens:
            ranked: list[tuple[float, str, str, RetrievedChunk]] = []
            for name, faculty, chunk in matches:
                candidate_tokens = _faculty_tokens(faculty)
                union = requested_tokens | candidate_tokens
                score = len(requested_tokens & candidate_tokens) / len(union) if union else 0.0
                ranked.append((score, name, faculty, chunk))
            ranked.sort(key=lambda item: item[0], reverse=True)
            best_score = ranked[0][0]
            matches = [
                (name, faculty, chunk)
                for score, name, faculty, chunk in ranked
                if score == best_score and score > 0
            ]
        if len(matches) == 1:
            name, _faculty, used = matches[0]
            answer = (
                f"عميد الكلية وفقًا لبيانات الجامعة المتاحة هو أ.د {name}."
                if language == "ar"
                else f"According to the available university data, the dean is Prof. Dr. {name}."
            )
            return answer, [used]
        if len(matches) > 1:
            return None
    return None


def _academic_structure_answer(
    question: str,
    chunks: list[RetrievedChunk],
    language: str,
) -> tuple[str, list[RetrievedChunk]] | None:
    """Extract departments/programs from retrieved faculty/program chunks.

    This is evidence-driven and generic: it only uses lines explicitly labeled
    as departments or programs in the retrieved text, and it never assigns a
    program to a department unless the text already states that relationship.
    """
    if not _ACADEMIC_STRUCTURE_QUESTION.search(question or ""):
        return None

    departments: list[str] = []
    programs: list[str] = []
    used: list[RetrievedChunk] = []
    for c in chunks:
        if (c.content_type or "").lower() not in {"faculty", "program"}:
            continue
        lines = _clean_lines(c.text)
        chunk_used = False
        for line in lines:
            if _DEPARTMENT_LINE_RE.match(line):
                if _append_unique(departments, line):
                    chunk_used = True
            elif _PROGRAM_LINE_RE.match(line):
                low_line = line.lower()
                descriptive_markers = (
                    " يعتمد ", " يدرس ", " يشتد ", " يلاقى ", " تخصصي ",
                    " متعدد التخصصات ", " بيني ", " aims ", " provides ",
                    " focuses ", " prepares ",
                )
                # A program title is a compact label. Faculty pages also have
                # prose sentences beginning with "برنامج ..." inside the
                # description/features sections; never expose those as names.
                if len(line) > 80 or any(marker in f" {low_line} " for marker in descriptive_markers):
                    continue
                if _append_unique(programs, line):
                    chunk_used = True
        if chunk_used:
            used.append(c)

    want_departments = _contains_any(question, ("department", "departments", "قسم", "أقسام", "اقسام"))
    want_programs = _contains_any(question, ("program", "programs", "برنامج", "برامج"))
    if not want_departments and not want_programs:
        want_departments = want_programs = True
    if want_departments and not departments:
        return None
    if want_programs and not programs:
        return None

    parts: list[str] = []
    if want_departments and departments:
        heading = "## Departments" if language == "en" else "## الأقسام"
        parts.append(heading + "\n" + "\n".join(f"- {x}" for x in departments))
    if want_programs and programs:
        heading = "## Programs" if language == "en" else "## البرامج"
        parts.append(heading + "\n" + "\n".join(f"- {x}" for x in programs))
    return "\n\n".join(parts), used


def _tuition_answer(
    chunks: list[RetrievedChunk],
    language: str,
) -> tuple[str, list[RetrievedChunk]] | None:
    """Extract a clear tuition table when fee rows are explicit in evidence."""
    fee_chunks = [c for c in chunks if (c.content_type or "").lower() == "tuition"]
    if not fee_chunks:
        return None
    rows: list[tuple[str, str]] = []
    used: list[RetrievedChunk] = []
    by_source: dict[str, list[RetrievedChunk]] = {}
    for c in fee_chunks:
        by_source.setdefault(c.source_url or c.chunk_id, []).append(c)
    for group in by_source.values():
        group.sort(key=lambda c: c.chunk_index or 0)
        text = "\n".join(c.text or "" for c in group)
        lines = _clean_lines(text)
        table_started = False
        for i in range(len(lines) - 1):
            name, value = lines[i], lines[i + 1]
            if table_started and _is_fee_table_end(name):
                break
            if _is_fee_table_header(name) or _is_fee_table_header(value):
                table_started = True
                continue
            if not table_started:
                continue
            if _looks_like_faculty_fee_row(name, value):
                rows.append((name, value))
                for c in group:
                    if c not in used:
                        used.append(c)
        if rows:
            break
    unique: list[tuple[str, str]] = []
    seen: set[str] = set()
    for name, value in rows:
        key = name.lower()
        if key not in seen:
            seen.add(key)
            unique.append((name, value))
    if len(unique) < 3:
        return None
    if language == "ar":
        header = "| الكلية | الرسوم بالجنيه المصري |\n|---|---|"
    else:
        header = "| Faculty | Tuition fee |\n|---|---|"
    table = "\n".join([header, *[f"| {name} | {value} |" for name, value in unique]])
    return table, used


# -- dispatcher -------------------------------------------------------------

def try_fast_answer(
    question: str,
    intent: str | None,
    chunks: list[RetrievedChunk],
    language: str = "en",
    *,
    route_confidence: float | None = None,
) -> tuple[str | None, list[RetrievedChunk]]:
    """Return ``(answer, used_chunks)`` or ``(None, [])`` to skip the fast path.

    ``route_confidence`` is the deterministic router's confidence for this
    question. When provided and below ``FAST_PATH_MIN_CONFIDENCE`` (default
    0.55), the fast path is skipped so a weak/ambiguous route (e.g. FACULTY
    at 0.53) never auto-answers with a canned extract. ``None`` preserves the
    legacy behavior for direct callers.
    """
    if not get_config().get("fast_path_enabled", True):
        return None, []
    if not chunks:
        return None, []
    if route_confidence is not None:
        min_conf = float(get_config().get("fast_path_min_confidence", 0.55))
        guarded_intents = {"LIST", "FACULTY", "PROGRAM"}
        clear_faculty_list = (
            (intent or "").upper() in {"LIST", "FACULTY"}
            and _is_faculties_list_question(question)
        )
        if (
            (intent or "").upper() in guarded_intents
            and not clear_faculty_list
            and route_confidence < min_conf
        ):
            logger.info(
                "Fast path skipped (route confidence %.3f < %.2f)",
                route_confidence, min_conf,
            )
            return None, []
    intent = (intent or "").upper()
    language = language if language in ("ar", "en") else "en"

    if intent in ("LIST", "FACULTY", "PROGRAM"):
        if intent == "PROGRAM":
            result = _academic_structure_answer(question, chunks, language)
            if result is not None:
                return result
        if _is_faculties_list_question(question):
            result = _faculties_answer(chunks, language)
            if result is not None:
                return result
    elif intent == "TUITION":
        result = _tuition_answer(chunks, language)
        if result is not None:
            return result
    elif intent == "LOCATION":
        result = _location_answer(chunks, language)
        if result is not None:
            return result
    elif intent == "CONTACT":
        result = _contact_answer(chunks, language)
        if result is not None:
            return result
    return None, []


def _clean_lines(text: str) -> list[str]:
    return [
        part.strip(" \t:-")
        for part in re.split(r"[\n|]+", text or "")
        if part.strip(" \t:-")
    ]


def _append_unique(items: list[str], value: str) -> bool:
    key = re.sub(r"\s+", " ", value).strip().lower()
    if not key or key in {re.sub(r"\s+", " ", x).strip().lower() for x in items}:
        return False
    items.append(value.strip())
    return True


def _contains_any(text: str, markers: tuple[str, ...]) -> bool:
    low = (text or "").lower()
    return any(marker.lower() in low for marker in markers)


def _looks_like_faculty_fee_row(name: str, value: str) -> bool:
    low_name = name.lower()
    bad = {
        "faculty", "college", "tuition fees", "tuition fee", "students",
        "الكلية", "قيمة الرسوم بالجنيه المصري", "رسوم الدراسة",
    }
    if low_name in bad or len(name) > 80:
        return False
    if not _FEE_RE.match(value.replace(" ", "")):
        return False
    return bool(re.search(r"\d", value)) and not bool(re.search(r"\d", name))


def _is_fee_table_header(text: str) -> bool:
    low = (text or "").lower()
    return low in {
        "faculty", "college", "tuition fees", "tuition fee",
        "الكلية", "قيمة الرسوم بالجنيه المصري",
    }


def _is_fee_table_end(text: str) -> bool:
    low = (text or "").lower().strip(" :")
    return low.startswith(("رسوم إضافية", "additional fees", "other fees"))

"""Tests for the deterministic fast-answer path (no LLM generation)."""

from __future__ import annotations

import pytest

from rag.generation.fast_path import (
    _academic_structure_answer,
    _contact_answer,
    _faculties_answer,
    _location_answer,
    person_evidence_answer,
    _tuition_answer,
    try_fast_answer,
)
from rag.schemas.documents import RetrievedChunk


def _chunk(
    chunk_id: str,
    text: str,
    *,
    source_url: str = "https://nmu.edu.eg/en",
    content_type: str = "home",
    language: str = "en",
    score: float = 0.9,
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id="d",
        text=text,
        score=score,
        title="t",
        source_url=source_url,
        content_type=content_type,
        language=language,
    )


_EN_DIR = (
    "Business\nاعرف المزيد\nLaw\nاعرف المزيد\nEngineering\nاعرف المزيد\n"
    "Computer Science & Engineering\nاعرف المزيد\nScience\nاعرف المزيد\n"
    "Medicine\nاعرف المزيد\nDentistry\nاعرف المزيد\nPharmacy"
)
_AR_DIR = (
    "الأعمال\nاعرف المزيد\nالقانون\nاعرف المزيد\nالهندسة\nاعرف المزيد\n"
    "العلوم\nاعرف المزيد\nالطب\nاعرف المزيد\nطب الأسنان"
)


class TestFaculties:
    def test_faculties_answer_en(self):
        chunks = [_chunk("d1", _EN_DIR, source_url="https://nmu.edu.eg/en/all-faculties-programs", content_type="program")]
        answer, used = _faculties_answer(chunks, "en")
        assert answer is not None
        assert "Business" in answer
        assert "Medicine" in answer
        assert "اعرف المزيد" not in answer
        assert used == chunks

    def test_faculties_answer_ar(self):
        chunks = [_chunk("d1", _AR_DIR, source_url="https://nmu.edu.eg/ar/all-faculties-programs", content_type="program", language="ar")]
        answer, used = _faculties_answer(chunks, "ar")
        assert answer is not None
        assert "الأعمال" in answer
        assert "الطب" in answer

    def test_faculties_requires_enough_items(self):
        chunks = [_chunk("d1", "Business\nLaw", source_url="https://nmu.edu.eg/en/all-faculties-programs", content_type="program")]
        assert _faculties_answer(chunks, "en") is None

    def test_faculties_requires_directory_chunk(self):
        chunks = [_chunk("d1", _EN_DIR, content_type="program")]
        assert _faculties_answer(chunks, "en") is None


class TestLocation:
    def test_location_from_about_en(self):
        chunks = [
            _chunk(
                "a1",
                "The President of the Arab Republic of Egypt Decree No. 437 of 2020 "
                "was issued. Its headquarters will be in New Mansoura, Dakahlia "
                "Governorate.",
                content_type="about",
            )
        ]
        answer, used = _location_answer(chunks, "en")
        assert answer is not None
        assert "New Mansoura" in answer
        assert used == chunks

    def test_location_from_contact_ar(self):
        chunks = [
            _chunk(
                "c1",
                "Contact Us\nOur Address\nDakahlia Governorate - New Mansoura City - International Coastal Road",
                content_type="contact",
                language="ar",
            )
        ]
        answer, used = _location_answer(chunks, "ar")
        assert answer is not None
        assert "المنصورة" in answer or "الدقهلية" in answer

    def test_location_no_match_falls_back(self):
        chunks = [_chunk("n1", "News about an event at the university library.", content_type="news")]
        assert _location_answer(chunks, "en") is None


class TestContact:
    def test_contact_extracts_phones(self):
        chunks = [
            _chunk(
                "c1",
                "Our Phone Number\n01070004148 - 01070004149\nEmail: info@nmu.edu.eg",
                content_type="contact",
            )
        ]
        answer, used = _contact_answer(chunks, "en")
        assert answer is not None
        assert "01070004148" in answer
        assert "info@nmu.edu.eg" in answer
        assert used == chunks

    def test_contact_no_data_falls_back(self):
        chunks = [_chunk("c1", "Get in Touch\nFill out the form below.", content_type="contact")]
        assert _contact_answer(chunks, "en") is None


class TestPersonEvidenceFallback:
    def test_extracts_arabic_university_president_from_evidence(self):
        chunks = [_chunk(
            "p0",
            "عن رئيس الجامعة | أ.د. معوض محمد الخولي | الرئيس المؤسس لجامعة المنصورة الجديدة",
            source_url="https://nmu.edu.eg/ar/about-the-president",
            content_type="president",
            language="ar",
        )]
        answer, used = person_evidence_answer(
            "مين رئيس جامعة المنصورة الجديدة؟", chunks, "ar"
        )
        assert "معوض محمد الخولي" in answer
        assert used == chunks

    def test_extracts_arabic_dean_from_retrieved_evidence(self):
        chunks = [_chunk(
            "p1",
            "تحت رعاية رئيس الجامعة وبريادة أ.د وائل صديق عميد كلية الهندسة.",
            source_url="https://nmu.edu.eg/ar/news/51",
            content_type="news",
            language="ar",
        )]
        answer, used = person_evidence_answer(
            "مين عميد كلية الهندسة؟", chunks, "ar"
        )
        assert "وائل صديق" in answer
        assert used == chunks

    def test_extracts_dean_when_name_and_role_are_separate_page_fields(self):
        chunk = _chunk(
            "p-separate",
            "أ.د./ وائل صديق عبداللطيف\nعميد الكلية",
            source_url="https://nmu.edu.eg/ar/faculties/3/engineering",
            content_type="faculty",
            language="ar",
        )
        chunk.title = "كلية الهندسة"
        answer, used = person_evidence_answer(
            "مين عميد كلية الهندسة؟", [chunk], "ar"
        )
        assert "وائل صديق عبداللطيف" in answer
        assert used == [chunk]

    def test_person_fallback_never_invents_missing_name(self):
        chunks = [_chunk("p1", "أخبار كلية الهندسة", language="ar")]
        assert person_evidence_answer("مين عميد كلية الهندسة؟", chunks, "ar") is None

    def test_selects_requested_faculty_when_multiple_deans_are_retrieved(self):
        chunks = [
            _chunk("p1", "أ.د وائل صديق عميد كلية الهندسة.", language="ar"),
            _chunk("p2", "أ.د إبراهيم فتحي معوض عميد كلية علوم وهندسة الحاسب،", language="ar"),
        ]
        answer, used = person_evidence_answer(
            "مين عميد كلية الهندسة في جامعة المنصورة الجديدة؟", chunks, "ar"
        )
        assert "وائل صديق" in answer
        assert "إبراهيم" not in answer
        assert used == [chunks[0]]


class TestAcademicStructure:
    def test_extracts_departments_and_programs_without_courses(self):
        text = (
            "برنامج هندسة الحاسب\nوصف البرنامج\n"
            "قسم علوم وهندسة الحاسب\nالبرمجة\nالخوارزميات\n"
            "قسم علوم وهندسة الذكاء الاصطناعي\nتعلم الآلة\n"
            "برنامج علوم الحاسب\nوصف البرنامج"
        )
        chunks = [_chunk("f1", text, content_type="faculty", language="ar")]
        answer, used = _academic_structure_answer(
            "ما هي اقسام وبرامج كلية علوم وهندسة الحاسب؟", chunks, "ar"
        )
        assert "## الأقسام" in answer
        assert "## البرامج" in answer
        assert "- قسم علوم وهندسة الحاسب" in answer
        assert "- برنامج هندسة الحاسب" in answer
        assert "الخوارزميات" not in answer
        assert used == chunks


class TestTuition:
    def test_extracts_arabic_fee_table(self):
        text = (
            "Contact Us\n01070004148\n"
            "الكلية\nقيمة الرسوم بالجنيه المصري\n"
            "الطب البشري\n150,000\n"
            "طب الأسنان\n130,000\n"
            "علوم وهندسة الحاسب\n75,000\n"
            "رسوم إضافية أخرى"
        )
        chunks = [_chunk("t1", text, content_type="tuition", language="ar")]
        answer, used = _tuition_answer(chunks, "ar")
        assert "| الكلية | الرسوم بالجنيه المصري |" in answer
        assert "| علوم وهندسة الحاسب | 75,000 |" in answer
        assert "Contact Us" not in answer
        assert used == chunks


class TestDispatcher:
    def test_faculties_list_question_triggers_fast_path(self):
        chunks = [_chunk("d1", _EN_DIR, source_url="https://nmu.edu.eg/en/all-faculties-programs", content_type="program")]
        answer, used = try_fast_answer(
            "What are all the faculties at New Mansoura University?", "FACULTY", chunks, "en"
        )
        assert answer is not None
        assert used

    def test_specific_faculty_question_skips_fast_path(self):
        chunks = [_chunk("d1", _EN_DIR, source_url="https://nmu.edu.eg/en/all-faculties-programs", content_type="program")]
        answer, _ = try_fast_answer(
            "What is the address of the Faculty of Engineering?", "FACULTY", chunks, "en"
        )
        assert answer is None

    def test_arabic_list_question(self):
        chunks = [_chunk("d1", _AR_DIR, source_url="https://nmu.edu.eg/ar/all-faculties-programs", content_type="program", language="ar")]
        answer, _ = try_fast_answer("ما هي كليات جامعة المنصورة الجديدة؟", "FACULTY", chunks, "ar")
        assert answer is not None
        assert "الطب" in answer

    def test_program_structure_question_triggers_fast_path(self):
        text = "برنامج علوم الحاسب\nقسم علوم وهندسة الحاسب\nقسم المعلوماتية الطبية الحيوية"
        chunks = [_chunk("f1", text, content_type="faculty", language="ar")]
        answer, used = try_fast_answer(
            "ما هي اقسام وبرامج كلية علوم وهندسة الحاسب؟", "PROGRAM", chunks, "ar",
            route_confidence=0.9,
        )
        assert answer is not None
        assert "## الأقسام" in answer
        assert used

    def test_no_chunks_returns_none(self):
        assert try_fast_answer("Where is the university?", "LOCATION", [], "en") == (None, [])

    def test_low_route_confidence_allows_clear_faculty_list(self):
        # A clear faculty-list query may use the authoritative directory even
        # if the deterministic router confidence is only the weak 0.53 level.
        chunks = [_chunk("d1", _EN_DIR, source_url="https://nmu.edu.eg/en/all-faculties-programs", content_type="program")]
        answer, used = try_fast_answer(
            "What are all the faculties at New Mansoura University?",
            "FACULTY", chunks, "en", route_confidence=0.53,
        )
        assert answer is not None
        assert used

    def test_high_route_confidence_still_answers(self):
        chunks = [_chunk("d1", _EN_DIR, source_url="https://nmu.edu.eg/en/all-faculties-programs", content_type="program")]
        answer, used = try_fast_answer(
            "What are all the faculties at New Mansoura University?",
            "FACULTY", chunks, "en", route_confidence=0.9,
        )
        assert answer is not None
        assert used

    def test_no_confidence_preserves_legacy_behavior(self):
        chunks = [_chunk("d1", _EN_DIR, source_url="https://nmu.edu.eg/en/all-faculties-programs", content_type="program")]
        answer, used = try_fast_answer(
            "What are all the faculties at New Mansoura University?",
            "FACULTY", chunks, "en",
        )
        assert answer is not None
        assert used


class TestPerfConfig:
    def test_config_exposes_perf_keys(self):
        from rag.config import get_config

        cfg = get_config()
        assert cfg["fast_path_min_confidence"] > 0.53
        assert cfg["top_context_chunks"] >= 1
        assert cfg["reranker_batch_size"] >= 1
        assert cfg["reranker_device"] in ("cpu", "cuda", "mps")
        assert cfg["cpu_threads"] >= 0

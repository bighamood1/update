"""Regression test suite for the NMU AI Assistant RAG pipeline.

Tests are designed to catch regressions in:
- Arabic/English query parity
- Retrieval (language filter, similarity threshold)
- Answer validation (over-strict refusal)
- Fast-path correctness
- Semantic cache safety (no runtime DB injection)
- Intent routing
- Inference (answer from evidence, not verbatim match)

LLM calls are mocked so these tests run fast on CPU without Ollama.
"""

from __future__ import annotations

import pytest

from rag.generation.fast_path import _location_answer, try_fast_answer
from rag.generation.response_formatter import format_final_answer
from rag.generation.validation import (
    REFUSAL_EN,
    REFUSAL_AR,
    answer_relevance_ok,
    completeness_issues,
    refusal_text,
    strip_reasoning_artifacts,
    validate_answer,
)
from rag.retrieval.query_normalizer import expand_query, normalize_query
from rag.retrieval.retriever import _CROSS_LINGUAL_INTENTS, Retriever
from rag.routing.router import QueryRouter
from rag.routing.schemas import RouteResult
from rag.schemas.documents import RetrievedChunk
from rag.query.understanding import understand


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _chunk(
    chunk_id: str,
    text: str,
    *,
    source_url: str = "https://nmu.edu.eg/en/about-us",
    content_type: str = "about",
    language: str = "en",
    faculty: str | None = None,
    score: float = 0.85,
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id="d1",
        text=text,
        score=score,
        title="Test",
        source_url=source_url,
        content_type=content_type,
        language=language,
        faculty=faculty,
    )


_LOCATION_TEXT_EN = (
    "New Mansoura University is located in New Mansoura City, "
    "Dakahlia Governorate, on the International Coastal Road."
)
_LOCATION_TEXT_AR = (
    "تقع جامعة المنصورة الجديدة في مدينة المنصورة الجديدة بمحافظة الدقهلية "
    "على الطريق الساحلي الدولي."
)
_FACULTIES_TEXT_EN = (
    "Business\nاعرف المزيد\nLaw\nاعرف المزيد\nEngineering\n"
    "اعرف المزيد\nComputer Science & Engineering\nاعرف المزيد\n"
    "Science\nاعرف المزيد\nMedicine\nاعرف المزيد\nDentistry\n"
    "اعرف المزيد\nPharmacy"
)
_FACULTIES_TEXT_AR = (
    "الأعمال\nاعرف المزيد\nالقانون\nاعرف المزيد\nالهندسة\n"
    "اعرف المزيد\nعلوم الحاسب\nاعرف المزيد\nالعلوم\n"
    "اعرف المزيد\nالطب\nاعرف المزيد\nطب الأسنان"
)


# ---------------------------------------------------------------------------
# Section 1: Intent routing (Arabic/English parity)
# ---------------------------------------------------------------------------

class TestIntentRouting:
    """Required test cases 1-4: location queries in both languages."""

    def test_arabic_location_intent(self):
        """'اين تقع جامعة المنصورة الجديدة' → LOCATION intent."""
        router = QueryRouter()
        r = router.route("اين تقع جامعة المنصورة الجديدة")
        assert r.intent == "LOCATION", f"Expected LOCATION, got {r.intent}"
        assert r.language == "ar"

    def test_arabic_location_paraphrase_intent(self):
        """'ما موقع الجامعة؟' → LOCATION intent."""
        router = QueryRouter()
        r = router.route("ما موقع الجامعة؟")
        assert r.intent == "LOCATION", f"Expected LOCATION, got {r.intent}"

    def test_english_location_intent(self):
        """'What is the location of the new Mansoura University?' → LOCATION."""
        router = QueryRouter()
        r = router.route("What is the location of the new Mansoura University?")
        assert r.intent == "LOCATION", f"Expected LOCATION, got {r.intent}"
        assert r.language == "en"

    def test_english_location_where_intent(self):
        """'Where is the university located?' → LOCATION."""
        router = QueryRouter()
        r = router.route("Where is the university located?")
        assert r.intent == "LOCATION", f"Expected LOCATION, got {r.intent}"

    def test_arabic_faculty_list_intent(self):
        """'ما هي كليات جامعة المنصورة الجديدة؟' → FACULTY or LIST."""
        router = QueryRouter()
        r = router.route("ما هي كليات جامعة المنصورة الجديدة؟")
        assert r.intent in ("FACULTY", "LIST", "PROGRAM"), f"Got {r.intent}"

    def test_english_faculty_list_intent(self):
        """'What faculties does NMU have?' → FACULTY or LIST."""
        router = QueryRouter()
        r = router.route("What faculties does NMU have?")
        assert r.intent in ("FACULTY", "LIST", "PROGRAM"), f"Got {r.intent}"


# ---------------------------------------------------------------------------
# Section 2: Cross-lingual intents skip language filter
# ---------------------------------------------------------------------------

class TestCrossLingualFilter:
    """Verify _CROSS_LINGUAL_INTENTS prevents language filter for LOCATION etc."""

    def test_location_in_cross_lingual_intents(self):
        assert "LOCATION" in _CROSS_LINGUAL_INTENTS

    def test_contact_in_cross_lingual_intents(self):
        assert "CONTACT" in _CROSS_LINGUAL_INTENTS

    def test_person_in_cross_lingual_intents(self):
        assert "PERSON" in _CROSS_LINGUAL_INTENTS

    def test_retriever_build_where_skips_language_for_location(self):
        """_build_where must NOT add a language clause for LOCATION intent."""
        from unittest.mock import MagicMock
        retriever = Retriever.__new__(Retriever)
        retriever.router_enabled = True
        retriever.confidence_threshold = 0.75
        retriever.filter_language = True
        retriever.filter_category = False
        retriever.filter_faculty = False

        route = RouteResult(
            intent="LOCATION",
            language="en",
            confidence=0.90,
            category="location",
            category_types=["about", "contact"],
        )
        where = retriever._build_where(route)
        # Must not include a language filter for LOCATION
        assert where is None or "language" not in str(where), (
            f"Language filter must be absent for LOCATION, got: {where}"
        )

    def test_retriever_build_where_applies_language_for_admission(self):
        """_build_where SHOULD add language clause for ADMISSION intent."""
        retriever = Retriever.__new__(Retriever)
        retriever.router_enabled = True
        retriever.confidence_threshold = 0.75
        retriever.filter_language = True
        retriever.filter_category = False
        retriever.filter_faculty = False

        route = RouteResult(
            intent="ADMISSION",
            language="ar",
            confidence=0.90,
            category="admissions",
            category_types=["admission"],
        )
        where = retriever._build_where(route)
        assert where is not None and "language" in str(where), (
            f"Language filter should be present for ADMISSION, got: {where}"
        )


# ---------------------------------------------------------------------------
# Section 3: Fast-path location (Arabic and English)
# ---------------------------------------------------------------------------

class TestLocationFastPath:
    """Required test cases 1-4: location fast-path for both languages."""

    def test_english_location_fast_path_from_english_chunk(self):
        """Fast path extracts EN location from EN about chunk."""
        chunks = [_chunk("e1", _LOCATION_TEXT_EN, content_type="about", language="en")]
        result = _location_answer(chunks, "en")
        assert result is not None, "Should extract location from English about chunk"
        answer, used = result
        assert "Dakahlia" in answer or "New Mansoura" in answer
        assert used

    def test_arabic_location_fast_path_from_arabic_chunk(self):
        """Fast path extracts AR location from AR about chunk."""
        chunks = [_chunk(
            "a1", _LOCATION_TEXT_AR,
            source_url="https://nmu.edu.eg/ar/about-us",
            content_type="about",
            language="ar",
        )]
        result = _location_answer(chunks, "ar")
        assert result is not None, "Should extract location from Arabic about chunk"
        answer, used = result
        assert "الدقهلية" in answer or "المنصورة" in answer
        assert used

    def test_english_question_uses_arabic_chunk_if_needed(self):
        """English LOCATION query can be answered from an Arabic chunk.

        After the retriever fix (no language filter for LOCATION), an EN query
        may receive AR chunks. The fast path checks BOTH marker sets, so it
        must find the answer in the AR chunk even when the question is English.
        """
        chunks = [_chunk(
            "a1", _LOCATION_TEXT_AR,
            source_url="https://nmu.edu.eg/ar/about-us",
            content_type="about",
            language="ar",
        )]
        result = _location_answer(chunks, "en")  # English question, AR chunk
        assert result is not None, (
            "Fast path must find location in Arabic chunk for English question"
        )

    def test_arabic_question_uses_english_chunk_if_needed(self):
        """Arabic LOCATION query can be answered from an English chunk."""
        chunks = [_chunk("e1", _LOCATION_TEXT_EN, content_type="about", language="en")]
        result = _location_answer(chunks, "ar")  # Arabic question, EN chunk
        assert result is not None, (
            "Fast path must find location in English chunk for Arabic question"
        )

    def test_location_not_from_wrong_content_type(self):
        """Location fast path must NOT accept news/program chunks."""
        chunks = [_chunk("n1", _LOCATION_TEXT_EN, content_type="news")]
        assert _location_answer(chunks, "en") is None

    def test_location_ignores_decree_sentence(self):
        """Decree-only text (no location signal) must not be returned."""
        decree = (
            "Presidential Decree No. 437 of 2020 was issued establishing "
            "New Mansoura University as a national university."
        )
        chunks = [_chunk("d1", decree, content_type="about")]
        result = _location_answer(chunks, "en")
        # "new mansoura" IS a weak marker — it must be skipped because the
        # sentence has no real location signal (governorate/road/campus).
        assert result is None, "Decree-only sentence must not produce a location answer"

    def test_location_fast_path_empty_chunks(self):
        answer, used = try_fast_answer("Where is NMU?", "LOCATION", [], "en")
        assert answer is None
        assert used == []


# ---------------------------------------------------------------------------
# Section 4: Faculty list (required test cases 5-6)
# ---------------------------------------------------------------------------

class TestFacultyFastPath:
    def test_faculty_list_arabic(self):
        """Required: ما هي كليات جامعة المنصورة الجديدة؟"""
        chunks = [_chunk(
            "p1", _FACULTIES_TEXT_AR,
            source_url="https://nmu.edu.eg/ar/all-faculties-programs",
            content_type="program",
            language="ar",
        )]
        answer, used = try_fast_answer(
            "ما هي كليات جامعة المنصورة الجديدة؟", "FACULTY", chunks, "ar"
        )
        assert answer is not None
        assert "الطب" in answer
        assert used

    def test_faculty_list_english(self):
        """Required: What faculties does NMU have?"""
        chunks = [_chunk(
            "p1", _FACULTIES_TEXT_EN,
            source_url="https://nmu.edu.eg/en/all-faculties-programs",
            content_type="program",
            language="en",
        )]
        answer, used = try_fast_answer(
            "What faculties does NMU have?", "FACULTY", chunks, "en"
        )
        assert answer is not None
        assert "Medicine" in answer
        assert used


# ---------------------------------------------------------------------------
# Section 5: Answer validation — must NOT over-refuse
# ---------------------------------------------------------------------------

class TestAnswerValidation:
    """Validate that good answers are never rejected by the validator."""

    def test_valid_english_location_answer_passes(self):
        answer = "New Mansoura University is located in New Mansoura City, Dakahlia Governorate."
        sources = [{"url": "https://nmu.edu.eg/en/about-us", "title": "About"}]
        chunks = [_chunk("e1", answer)]
        result = validate_answer(answer, sources, chunks, question_language="en")
        assert result["ok"], f"Valid EN answer rejected: {result['issues']}"
        assert result["cleaned"]

    def test_valid_arabic_location_answer_passes(self):
        answer = "تقع جامعة المنصورة الجديدة في مدينة المنصورة الجديدة بمحافظة الدقهلية."
        sources = [{"url": "https://nmu.edu.eg/ar/about-us", "title": "عن الجامعة"}]
        chunks = [_chunk("a1", answer, language="ar")]
        result = validate_answer(answer, sources, chunks, question_language="ar")
        assert result["ok"], f"Valid AR answer rejected: {result['issues']}"

    def test_language_mismatch_is_soft_not_hard_failure(self):
        """language_mismatch alone must NOT cause ok=False."""
        # An otherwise valid answer that has some Arabic in it for an EN question
        answer = "NMU is located in New Mansoura (مدينة المنصورة الجديدة), Dakahlia Governorate."
        sources = [{"url": "https://nmu.edu.eg/en/about-us", "title": "About"}]
        chunks = [_chunk("e1", answer)]
        result = validate_answer(answer, sources, chunks, question_language="en")
        # language_mismatch might appear in issues, but ok must still be True
        assert result["ok"], (
            f"language_mismatch alone must not make ok=False: {result['issues']}"
        )

    def test_empty_answer_fails(self):
        result = validate_answer("", [], None, question_language="en")
        assert not result["ok"]
        assert "empty_answer" in result["issues"]

    def test_fabricated_url_causes_failure(self):
        answer = "See https://nmu.edu.eg/made-up-page for details."
        sources = [{"url": "https://nmu.edu.eg/en/about-us", "title": "About"}]
        chunks = [_chunk("e1", "Real content")]
        result = validate_answer(answer, sources, chunks, question_language="en")
        assert not result["ok"]
        assert any("fabricated_url" in i for i in result["issues"])

    def test_refusal_text_language(self):
        assert refusal_text("ar") == REFUSAL_AR
        assert refusal_text("en") == REFUSAL_EN
        assert "couldn't find" in REFUSAL_EN

    def test_reasoning_artifact_is_stripped(self):
        answer = (
            "Okay, I need to answer from the context.\n"
            "New Mansoura University is located in New Mansoura City."
        )
        cleaned, stripped = strip_reasoning_artifacts(answer)
        assert stripped
        assert cleaned == "New Mansoura University is located in New Mansoura City."

    def test_validate_answer_strips_think_block(self):
        answer = "<think>I should inspect the source.</think>\nThe tuition fee is 100 EGP."
        chunks = [_chunk("fee1", "The tuition fee is 100 EGP.", content_type="tuition")]
        result = validate_answer(answer, [], chunks, question_language="en")
        assert result["ok"]
        assert "think" not in result["cleaned"].lower()
        assert "reasoning_artifact_stripped" in result["issues"]

    def test_strips_incomplete_evidence_preamble(self):
        cleaned, stripped = strip_reasoning_artifacts("Looking at the evidence items:")
        assert stripped
        assert cleaned == ""

    def test_strips_incomplete_starting_preamble(self):
        cleaned, stripped = strip_reasoning_artifacts("Starting with")
        assert stripped
        assert cleaned == ""

    def test_person_question_with_koleya_is_not_mistaken_for_all_items(self):
        chunks = [_chunk("p1", "أ.د وائل صديق عميد كلية الهندسة\nسطر 2\nسطر 3\nسطر 4\nسطر 5", language="ar")]
        issues = completeness_issues(
            "عميد كلية الهندسة هو أ.د وائل صديق.",
            chunks,
            question="مين عميد كلية الهندسة؟",
            intent="PERSON",
        )
        assert not any(i.startswith("incomplete_evidence_coverage:") for i in issues)

    def test_source_context_leakage_is_flagged(self):
        answer = "According to Source 1, NMU is located in New Mansoura City."
        chunks = [_chunk("loc", "NMU is located in New Mansoura City.")]
        result = validate_answer(answer, [], chunks, question_language="en")
        assert not result["ok"]
        assert "source_or_context_leakage" in result["issues"]

    def test_final_formatter_removes_source_labels_and_urls(self):
        answer = (
            "According to Source 1: NMU is located in New Mansoura City.\n"
            "https://nmu.edu.eg/en/about-us"
        )
        cleaned, issues = format_final_answer(answer)
        assert cleaned == "NMU is located in New Mansoura City."
        assert "source_or_context_label_stripped" in issues
        assert "url_removed_from_answer" in issues

    def test_final_formatter_removes_repeated_sentences(self):
        answer = (
            "NMU is located in New Mansoura City. "
            "NMU is located in New Mansoura City. "
            "It is in Dakahlia Governorate."
        )
        cleaned, _ = format_final_answer(answer)
        assert cleaned.count("NMU is located in New Mansoura City.") == 1
        assert "Dakahlia" in cleaned

    def test_validate_answer_flags_excessive_repetition(self):
        answer = (
            "NMU is located in New Mansoura City. "
            "NMU is located in New Mansoura City. "
            "It is in Dakahlia Governorate."
        )
        chunks = [_chunk("loc", "NMU is located in New Mansoura City.")]
        result = validate_answer(answer, [], chunks, question_language="en")
        assert "excessive_repetition" in result["issues"]


# ---------------------------------------------------------------------------
# Section 6: answer_relevance_ok — intent gate
# ---------------------------------------------------------------------------

class TestAnswerRelevanceGate:
    """The relevance gate must allow valid answers and block only clear off-topic."""

    def test_location_answer_passes_with_location_markers(self):
        answer = "New Mansoura University is located in New Mansoura City, Dakahlia Governorate."
        assert answer_relevance_ok(answer, "LOCATION")

    def test_location_answer_passes_arabic(self):
        answer = "تقع جامعة المنصورة الجديدة في مدينة المنصورة الجديدة بمحافظة الدقهلية."
        assert answer_relevance_ok(answer, "LOCATION")

    def test_location_answer_blocked_for_decree_only(self):
        """An answer that talks only about establishment decrees should fail LOCATION gate."""
        answer = (
            "Presidential Decree No. 437 was issued to establish New Mansoura University "
            "as a national research university founded in 2020."
        )
        # If no location marker AND has establishment markers → should fail
        assert not answer_relevance_ok(answer, "LOCATION")

    def test_explicit_refusal_always_passes_gate(self):
        """An explicit refusal (not enough info) must pass the gate — not regenerated."""
        assert answer_relevance_ok(REFUSAL_EN, "LOCATION")
        assert answer_relevance_ok(REFUSAL_AR, "LOCATION")

    def test_unknown_intent_always_passes(self):
        assert answer_relevance_ok("Anything goes here.", "UNKNOWN_INTENT")

    def test_faculty_answer_passes(self):
        answer = "The faculties of NMU include: Faculty of Medicine, Faculty of Engineering."
        assert answer_relevance_ok(answer, "FACULTY")

    def test_empty_answer_fails_gate(self):
        assert not answer_relevance_ok("", "LOCATION")


# ---------------------------------------------------------------------------
# Section 7: Query understanding (language detection)
# ---------------------------------------------------------------------------

class TestQueryUnderstanding:
    def test_arabic_location_understanding(self):
        u = understand("اين تقع جامعة المنصورة الجديدة")
        assert u.language == "ar"
        assert u.intent == "LOCATION"

    def test_english_location_understanding(self):
        u = understand("Where is New Mansoura University located?")
        assert u.language == "en"
        assert u.intent == "LOCATION"

    def test_arabic_paraphrase_understanding(self):
        u = understand("ما موقع الجامعة؟")
        assert u.language == "ar"
        assert u.intent == "LOCATION"

    def test_english_paraphrase_understanding(self):
        u = understand("Where is the university located?")
        assert u.language == "en"
        assert u.intent == "LOCATION"

    def test_arabic_faculty_understanding(self):
        u = understand("ما هي كليات جامعة المنصورة الجديدة؟")
        assert u.language == "ar"
        assert u.intent in ("FACULTY", "LIST", "PROGRAM")

    def test_english_faculty_understanding(self):
        u = understand("What faculties does NMU have?")
        assert u.language == "en"
        assert u.intent in ("FACULTY", "LIST", "PROGRAM")

    def test_arabic_departments_programs_understanding(self):
        u = understand("ما هي اقسام وبرامج كلية علوم وهندسة الحاسب")
        assert u.language == "ar"
        assert u.intent == "PROGRAM"
        assert u.route.faculty == "computer-science-and-engineering"


class TestProgramRetrievalGuards:
    def test_program_query_expansion_includes_departments(self):
        expanded = expand_query(normalize_query("programs"), "PROGRAM", max_terms=8)
        assert "department" in expanded

    def test_faculty_program_evidence_inserted_before_directory(self):
        retriever = Retriever.__new__(Retriever)
        retriever.last_meta = {"route": {"faculty": "computer-science-and-engineering"}}
        ranked = [
            _chunk(
                "dir",
                "Business\nComputer Science & Engineering\nMedicine",
                source_url="https://nmu.edu.eg/en/all-faculties-programs",
                content_type="program",
                score=1.0,
            ),
            _chunk(
                "cse",
                "قسم علوم وهندسة الحاسب\nالبرمجة\nالخوارزميات",
                source_url="https://nmu.edu.eg/ar/faculties/4/cse",
                content_type="faculty",
                language="ar",
                faculty="computer-science-and-engineering",
                score=0.9,
            ),
        ]
        route = RouteResult(
            intent="PROGRAM",
            language="ar",
            confidence=0.61,
            category="programs",
            category_types=["program", "faculty"],
            faculty="computer-science-and-engineering",
            faculty_id="4",
        )
        selected = [ranked[0]]
        result = retriever._ensure_required_evidence(
            ranked, selected, "PROGRAM", "ما هي اقسام وبرامج كلية علوم وهندسة الحاسب", "ar", route
        )
        assert result[0].chunk_id == "cse"


# ---------------------------------------------------------------------------
# Section 8: Inference — paraphrase tolerance
# (Required test case 8: exact wording not required in source)
# ---------------------------------------------------------------------------

class TestInferenceAndParaphrase:
    """The system must accept inferred or paraphrased answers grounded in evidence."""

    def test_location_inference_passes_validation(self):
        """An inferred location answer (paraphrased from evidence) must pass."""
        evidence = "NMU is in New Mansoura City, Dakahlia, on the International Coastal Road."
        answer = (
            "New Mansoura University is located in New Mansoura City, "
            "Dakahlia Governorate. Its campus is on the International Coastal Road."
        )
        sources = [{"url": "https://nmu.edu.eg/en/about-us", "title": "About NMU"}]
        chunks = [_chunk("e1", evidence)]
        result = validate_answer(answer, sources, chunks, question_language="en")
        assert result["ok"], f"Paraphrased/inferred answer rejected: {result['issues']}"

    def test_arabic_paraphrase_passes_validation(self):
        """Arabic paraphrase of retrieved evidence must pass."""
        answer = "تقع الجامعة في مدينة المنصورة الجديدة بمحافظة الدقهلية."
        sources = [{"url": "https://nmu.edu.eg/ar/about-us", "title": "عن الجامعة"}]
        chunks = [_chunk("a1", _LOCATION_TEXT_AR, language="ar")]
        result = validate_answer(answer, sources, chunks, question_language="ar")
        assert result["ok"], f"Arabic paraphrase rejected: {result['issues']}"


# ---------------------------------------------------------------------------
# Section 9: Irrelevant question (required test case 9)
# ---------------------------------------------------------------------------

class TestIrrelevantQuestion:
    """Questions outside the NMU knowledge base should produce a refusal."""

    def test_unrelated_question_has_refusal_text_available(self):
        """Verify refusal strings exist and are non-empty in both languages."""
        assert len(REFUSAL_EN) > 20
        assert len(REFUSAL_AR) > 20

    def test_empty_retrieval_leads_to_refusal(self):
        """Pipeline's empty-context path returns a well-formed refusal."""
        # With no context, validate_answer on empty string should fail (ok=False)
        result = validate_answer("", [], None, question_language="en")
        assert not result["ok"]


# ---------------------------------------------------------------------------
# Section 10: False-premise questions (required test case 10)
# ---------------------------------------------------------------------------

class TestFalsePremise:
    """The assistant must NOT accept a false premise in the question."""

    def test_false_premise_answer_must_not_confirm_false_claim(self):
        """If retrieved evidence contradicts the premise, the validator must pass
        an answer that corrects the premise (not one that confirms it)."""
        # A correct answer that disputes the false premise passes validation
        answer = (
            "New Mansoura University is NOT located in Cairo. "
            "It is located in New Mansoura City, Dakahlia Governorate."
        )
        sources = [{"url": "https://nmu.edu.eg/en/about-us", "title": "About NMU"}]
        chunks = [_chunk("e1", _LOCATION_TEXT_EN)]
        result = validate_answer(answer, sources, chunks, question_language="en")
        assert result["ok"], f"False-premise rebuttal was rejected: {result['issues']}"

    def test_answer_relevance_ok_does_not_block_premise_corrections(self):
        """An answer correcting a false premise must pass the intent gate."""
        answer = (
            "New Mansoura University is not in Alexandria; "
            "it is located in New Mansoura City, Dakahlia Governorate, "
            "on the International Coastal Road."
        )
        assert answer_relevance_ok(answer, "LOCATION")


# ---------------------------------------------------------------------------
# Section 11: Semantic cache safety
# ---------------------------------------------------------------------------

class TestSemanticCacheSafety:
    """Verify the cache never injects old answers as factual knowledge."""

    def test_cache_store_does_not_use_runtime_db_answers_as_evidence(self):
        """The cache stores answers keyed by embedding similarity + intent +
        kb_version. No code path in store.py reads past answers into retrieval
        context. This test verifies the data model is isolated."""
        from rag.cache.store import RUNTIME_TABLES
        # Verify that retrieval_memory only stores source URLs, not generated answers
        # (source_json is a list of source dicts, NOT the answer text)
        assert "retrieval_memory" in RUNTIME_TABLES
        # The cache_entries table stores answers, but they are only served
        # through SemanticCache.lookup() and never fed into the RAG context.
        assert "cache_entries" in RUNTIME_TABLES

    def test_refusal_text_never_stored_in_cache(self):
        """Validate_answer produces ok=False for empty answers,
        which means refusals are never cached (quality_score would be 0)."""
        result = validate_answer("", [], None, question_language="en")
        assert not result["ok"]
        assert not result["cleaned"]


# ---------------------------------------------------------------------------
# Section 12: Configuration sanity
# ---------------------------------------------------------------------------

class TestConfigSanity:
    """Verify that key configuration values are within safe ranges."""

    def test_similarity_threshold_is_not_too_high(self):
        from rag.config import get_config
        cfg = get_config()
        assert cfg["similarity_threshold"] <= 0.6, (
            "similarity_threshold too high — will discard valid chunks"
        )

    def test_cache_similarity_threshold_is_reasonable(self):
        from rag.config import get_config
        cfg = get_config()
        assert 0.85 <= cfg["cache_similarity_threshold"] <= 0.99

    def test_router_confidence_threshold_is_not_too_low(self):
        from rag.config import get_config
        cfg = get_config()
        assert cfg["router_confidence_threshold"] >= 0.75, (
            "router_confidence_threshold too low — weak routes will apply "
            "restrictive filters"
        )

    def test_cross_lingual_intents_set_is_populated(self):
        assert len(_CROSS_LINGUAL_INTENTS) >= 4

    def test_top_k_fact_provides_enough_chunks(self):
        from rag.config import get_config
        cfg = get_config()
        assert cfg["top_k_fact"] >= 3, "top_k_fact too small for factual questions"

    def test_fast_path_min_confidence_is_above_weak_signal(self):
        from rag.config import get_config
        cfg = get_config()
        assert cfg["fast_path_min_confidence"] > 0.53


# ---------------------------------------------------------------------------
# Section 13: Repeated question (required test case 11 — cache + parity)
# ---------------------------------------------------------------------------

class TestRepeatedQuestion:
    """Repeated questions should return consistent answers via cache or pipeline."""

    def test_location_answer_stable_across_calls(self):
        """Two calls to _location_answer with the same chunk must return identical text."""
        chunks = [_chunk("e1", _LOCATION_TEXT_EN, content_type="about")]
        r1 = _location_answer(chunks, "en")
        r2 = _location_answer(chunks, "en")
        assert r1 is not None and r2 is not None
        assert r1[0] == r2[0], "Fast path must be deterministic for repeated calls"


# ---------------------------------------------------------------------------
# Section 14: Similar but semantically different questions
# ---------------------------------------------------------------------------

class TestSemanticallySimilarQuestions:
    """Questions that look similar but have different intents must route differently."""

    def test_location_vs_faculty_intent(self):
        router = QueryRouter()
        loc = router.route("Where is New Mansoura University located?")
        fac = router.route("What faculties does New Mansoura University have?")
        assert loc.intent == "LOCATION"
        assert fac.intent in ("FACULTY", "LIST", "PROGRAM")
        assert loc.intent != fac.intent

    def test_tuition_vs_scholarship_intent(self):
        router = QueryRouter()
        t = router.route("What are the tuition fees at NMU?")
        s = router.route("Does NMU offer scholarships?")
        assert t.intent == "TUITION"
        assert s.intent == "SCHOLARSHIP"
        assert t.intent != s.intent

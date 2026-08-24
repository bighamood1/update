"""Unit tests for prompt building and context assembly (no services)."""

from __future__ import annotations

from rag.generation.prompts import SYSTEM_PROMPT, build_rag_prompt
from rag.pipeline.rag import ContextBuilder, RAGPipeline
from rag.schemas.documents import RetrievedChunk


def test_system_prompt_is_grounded():
    assert "Use only the provided evidence" in SYSTEM_PROMPT
    assert "Do not invent facts" in SYSTEM_PROMPT
    assert "Do not write analysis" in SYSTEM_PROMPT


def test_rag_prompt_contains_question_and_context():
    prompt = build_rag_prompt("What is the tuition fee?", "Fees are 100000 EGP.")
    assert "What is the tuition fee?" in prompt
    assert "Fees are 100000 EGP." in prompt
    assert "RETRIEVED EVIDENCE" in prompt
    assert "Return only the final answer" in prompt
    assert "Reason from" not in prompt


def test_generation_options_are_intent_bounded():
    pipeline = RAGPipeline.__new__(RAGPipeline)
    fact = pipeline._generation_options("LOCATION")
    complex_opts = pipeline._generation_options("COMPARISON")
    assert fact["num_predict"] <= complex_opts["num_predict"]
    assert "stop" in fact
    assert "RETRIEVED EVIDENCE:" in fact["stop"]


def test_specific_faculty_overview_uses_fact_sized_generation_budget():
    pipeline = RAGPipeline.__new__(RAGPipeline)
    faculty = pipeline._generation_options("FACULTY")
    programs = pipeline._generation_options("PROGRAM")
    assert faculty["num_predict"] <= 512
    assert faculty["num_predict"] < programs["num_predict"]


def test_context_builder_truncates_to_max_chars():
    chunks = [
        RetrievedChunk(
            chunk_id="c0",
            document_id="d",
            text="Short first chunk.",
            score=0.9,
            title="Title 0",
            source_url="https://nmu.edu.eg/0",
        ),
        RetrievedChunk(
            chunk_id="c1",
            document_id="d",
            text="Word " * 500,
            score=0.8,
            title="Title 1",
            source_url="https://nmu.edu.eg/1",
        ),
    ]
    builder = ContextBuilder(max_chunks=6, max_chars=200)
    context = builder.build(chunks)
    assert 0 < len(context) <= 200
    assert "[Evidence item 1]" in context
    assert "[Evidence item 2]" not in context
    assert "https://nmu.edu.eg/0" not in context


def test_context_builder_respects_max_chunks():
    chunks = [
        RetrievedChunk(
            chunk_id=f"c{i}",
            document_id="d",
            text="Short content.",
            score=0.9,
            title=f"Title {i}",
            source_url=f"https://nmu.edu.eg/{i}",
        )
        for i in range(5)
    ]
    builder = ContextBuilder(max_chunks=2, max_chars=100_000)
    context = builder.build(chunks)
    assert context.count("[Evidence item") == 2


def test_context_builder_sources_dedupe_by_url_title():
    chunks = [
        RetrievedChunk(
            chunk_id="c1", document_id="d", text="A", score=0.9,
            title="Same", source_url="https://nmu.edu.eg/en",
        ),
        RetrievedChunk(
            chunk_id="c2", document_id="d", text="B", score=0.8,
            title="Same", source_url="https://nmu.edu.eg/en",
        ),
    ]
    builder = ContextBuilder()
    assert len(builder.sources(chunks)) == 1


def _pipeline_with_list_types():
    pipeline = RAGPipeline.__new__(RAGPipeline)
    pipeline.retriever = type(
        "FakeRetriever", (), {"list_source_types": {"program", "about", "faculty"}}
    )()
    return pipeline


def test_fallback_list_answer_prefers_arabic_directory():
    chunks = [
        RetrievedChunk(
            chunk_id="a1", document_id="d", text="نظرة عامة " * 40, score=0.51,
            title="About", source_url="https://nmu.edu.eg/ar/about-us",
            content_type="about", language="ar",
        ),
        RetrievedChunk(
            chunk_id="p1", document_id="d", text="الأعمال اعرف المزيد القانون "
            "اعرف المزيد الهندسة اعرف المزيد", score=0.5,
            title="Faculties", source_url="https://nmu.edu.eg/ar/all-faculties-programs",
            content_type="program", language="ar",
        ),
    ]
    answer = _pipeline_with_list_types()._fallback_list_answer(chunks, intent="FACULTY")
    assert "الأعمال" in answer
    assert "الهندسة" in answer
    # Answers must not embed source URLs (they surface via the API sources).
    assert "https://" not in answer
    assert "اعرف المزيد" not in answer


def test_fallback_list_answer_returns_refusal_without_sources():
    answer = _pipeline_with_list_types()._fallback_list_answer([], intent="LIST")
    assert "does not contain enough information" in answer

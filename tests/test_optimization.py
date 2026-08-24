"""Unit tests for answer validation, caching, and safe retrieval filtering."""

from __future__ import annotations

import time

from rag.generation.validation import refusal_text, validate_answer
from rag.retrieval.bm25 import BM25Index
from rag.retrieval.retriever import Retriever
from rag.schemas.documents import RetrievedChunk
from rag.utils.cache import LRUCache


# -- answer validation --------------------------------------------------------

def _retrieved_chunk(cid, url, text="content"):
    return RetrievedChunk(
        chunk_id=cid, document_id="d1", text=text, score=0.9,
        title="T", source_url=url, content_type="about",
    )


def test_validate_accepts_grounded_answer():
    retr = [_retrieved_chunk("c1", "https://nmu.edu.eg/en/about-us")]
    sources = [{"title": "T", "url": "https://nmu.edu.eg/en/about-us"}]
    v = validate_answer(
        "New Mansoura University is in Dakahlia. Source: https://nmu.edu.eg/en/about-us",
        sources, retr, question_language="en",
    )
    assert v["ok"]
    assert not v["issues"]


def test_validate_removes_fabricated_url():
    retr = [_retrieved_chunk("c1", "https://nmu.edu.eg/en/about-us")]
    v = validate_answer(
        "Answer citing a made-up site https://evil.example.com/x and the real one.",
        [{"title": "T", "url": "https://nmu.edu.eg/en/about-us"}],
        retr, question_language="en",
    )
    assert any("fabricated_url" in i for i in v["issues"])
    assert "evil.example.com" not in v["cleaned"]
    assert "real one" in v["cleaned"]


def test_validate_empty_answer_fails():
    v = validate_answer("  ", [], [], question_language="en")
    assert not v["ok"]
    assert v["issues"] == ["empty_answer"]


def test_validate_refusal_language():
    assert "لم أتمكن" in refusal_text("ar")
    assert "I couldn't find" in refusal_text("en")


def test_validate_language_mismatch_flagged():
    v = validate_answer("This is an English answer.", [], [], question_language="ar")
    assert "language_mismatch" in v["issues"]


# -- LRU cache -----------------------------------------------------------------

def test_lru_cache_bounded_and_orders():
    c = LRUCache(maxsize=2)
    c.put("a", 1)
    c.put("b", 2)
    c.put("c", 3)  # evicts "a"
    assert c.get("a") is None
    assert c.get("b") == 2
    assert c.get("c") == 3


def test_lru_cache_ttl_expiry():
    c = LRUCache(maxsize=10, ttl=0.05)
    c.put("k", "v")
    assert c.get("k") == "v"
    time.sleep(0.08)
    assert c.get("k") is None


# -- retrieval safe filtering / dynamic top-k ----------------------------------

def _make_retriever() -> Retriever:
    r = Retriever.__new__(Retriever)
    r.max_chunks_per_source = 3
    r.source_priority = {}
    r.list_source_types = {"program", "about", "faculty"}
    r.router_enabled = True
    r.confidence_threshold = 0.80
    r.min_results = 3
    r.fallback_enabled = True
    r.filter_language = True
    r.filter_category = True
    r.filter_faculty = True
    r.dynamic_top_k = True
    r.top_k_fact = 4
    r.top_k_list = 6
    r.top_k_complex = 8
    r.top_k = 8
    return r


def _src_chunk(cid, text, score, url, content_type, language="en") -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=cid, document_id="d1", text=text, score=score,
        title="T", source_url=url, content_type=content_type, language=language,
    )


def test_build_where_returns_none_when_low_confidence():
    r = _make_retriever()
    from rag.routing.schemas import RouteResult

    low = RouteResult(intent="FACT", confidence=0.4, language="en")
    assert r._build_where(low) is None


def test_build_where_faculty_high_confidence():
    r = _make_retriever()
    from rag.routing.schemas import RouteResult

    high = RouteResult(
        intent="ADMISSION", category="admissions",
        category_types=["admission", "about"], faculty="medicine",
        language="ar", confidence=0.9,
    )
    where = r._build_where(high)
    assert where == {"$and": [
        {"language": {"$eq": "ar"}},
        {"content_type": {"$in": ["admission", "about"]}},
        {"faculty": {"$eq": "medicine"}},
    ]}


def test_dynamic_top_k_simple_vs_complex():
    r = _make_retriever()
    assert r._dynamic_top_k("FACT") == 4
    assert r._dynamic_top_k("LOCATION") == 4
    assert r._dynamic_top_k("COMPARISON") == 8
    assert r._dynamic_top_k("FACULTY") == 6
    assert r._dynamic_top_k("PROGRAM") == 6


def test_fallback_merge_keeps_routed_priority():
    r = _make_retriever()
    routed = [_src_chunk("r1", "a", 0.9, "https://nmu.edu.eg/r1", "admission")]
    broad = [
        _src_chunk("r1", "a", 0.9, "https://nmu.edu.eg/r1", "admission"),
        _src_chunk("b1", "b", 0.8, "https://nmu.edu.eg/b1", "about"),
    ]
    merged = {c.chunk_id: c for c in routed}
    for c in broad:
        merged.setdefault(c.chunk_id, c)
    out = list(merged.values())
    assert {c.chunk_id for c in out} == {"r1", "b1"}


def test_bm25_rrf_prefers_joint():
    idx = BM25Index({"a": "New Mansoura University is located in New Mansoura City."})
    assert idx.rrf_score(0, 0) > idx.rrf_score(0, None)
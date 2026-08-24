"""Unit tests for retrieval deduplication (no vector store required)."""

from __future__ import annotations

from rag.retrieval.retriever import Retriever
from rag.schemas.documents import RetrievedChunk


def _chunk(chunk_id: str, text: str, score: float, doc_hash: str | None = "h1") -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id="d1",
        text=text,
        score=score,
        title="T",
        source_url="https://nmu.edu.eg/en",
        content_type="home",
        document_hash=doc_hash,
    )


def test_dedupe_by_document_hash_keeps_highest():
    a = _chunk("c1", "same text", 0.7, "hash-a")
    b = _chunk("c2", "same text", 0.9, "hash-a")
    out = Retriever._dedupe([a, b])
    assert len(out) == 1
    assert out[0].score == 0.9


def test_dedupe_by_text_collapses_mirror_pages():
    # Mirrored pages have different hashes but identical text.
    a = _chunk("c1", "Same visible body text on both pages.", 0.8, "hash-www")
    b = _chunk("c2", "Same visible body text on both pages.", 0.6, "hash-non-www")
    out = Retriever._dedupe([a, b])
    assert len(out) == 1
    assert out[0].chunk_id == "c1"


def test_dedupe_keeps_distinct_content():
    a = _chunk("c1", "Completely different content one.", 0.8, "h1")
    b = _chunk("c2", "Completely different content two.", 0.7, "h2")
    out = Retriever._dedupe([a, b])
    assert len(out) == 2


def _make_retriever() -> Retriever:
    r = Retriever.__new__(Retriever)
    r.max_chunks_per_source = 3
    r.source_priority = {}
    r.list_source_types = {"program", "about", "faculty"}
    return r


def _src_chunk(cid, text, score, url, content_type, language="en") -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=cid,
        document_id="d1",
        text=text,
        score=score,
        title="T",
        source_url=url,
        content_type=content_type,
        language=language,
    )


def test_source_diversity_caps_per_url():
    r = _make_retriever()
    chunks = [
        _src_chunk(f"c{i}", f"text {i}", 0.9 - i / 100, "https://nmu.edu.eg/fac", "faculty")
        for i in range(5)
    ]
    out = r._apply_source_diversity(chunks, list_mode=False)
    assert len(out) == 3


def test_source_diversity_keeps_distinct_sources():
    r = _make_retriever()
    chunks = [
        _src_chunk("c1", "a", 0.9, "https://nmu.edu.eg/a", "faculty"),
        _src_chunk("c2", "b", 0.8, "https://nmu.edu.eg/b", "about"),
        _src_chunk("c3", "c", 0.7, "https://nmu.edu.eg/c", "home"),
    ]
    out = r._apply_source_diversity(chunks, list_mode=False)
    assert len(out) == 3


def test_directory_coverage_reinserts_list_source():
    r = _make_retriever()
    ranked = [
        _src_chunk("c1", "reg", 0.9, "https://nmu.edu.eg/transfer", "regulation"),
        _src_chunk("c2", "list", 0.7, "https://nmu.edu.eg/all-faculties", "program"),
    ]
    diverse = [ranked[0]]  # only the regulation chunk survived rerank
    out = r._guarantee_directory_coverage(ranked, diverse, query_language="en")
    assert out[0].chunk_id == "c2"
    assert {c.chunk_id for c in out} == {"c1", "c2"}


def test_directory_coverage_prefers_query_language():
    r = _make_retriever()
    ranked = [
        _src_chunk("c-en", "en list", 0.9, "https://nmu.edu.eg/en/list", "program", "en"),
        _src_chunk("c-ar", "ar list", 0.8, "https://nmu.edu.eg/ar/list", "program", "ar"),
    ]
    diverse = [ranked[1]]
    out = r._guarantee_directory_coverage(ranked, diverse, query_language="ar")
    # c-ar already present by URL, so nothing changes.
    assert out[0].chunk_id == "c-ar"

    diverse2 = [ranked[0]]
    out2 = r._guarantee_directory_coverage(ranked, diverse2, query_language="ar")
    assert out2[0].chunk_id == "c-ar"


def test_directory_coverage_no_op_when_no_list_source():
    r = _make_retriever()
    ranked = [_src_chunk("c1", "reg", 0.9, "https://nmu.edu.eg/x", "regulation")]
    diverse = list(ranked)
    out = r._guarantee_directory_coverage(ranked, diverse, query_language="en")
    assert [c.chunk_id for c in out] == ["c1"]
"""Unit tests for hybrid retrieval: BM25, query normalization, RRF fusion."""

from __future__ import annotations

from rag.retrieval.bm25 import BM25Index, tokenize
from rag.retrieval.query_normalizer import expand_query, is_arabic, normalize_query
from rag.retrieval.retriever import Retriever
from rag.schemas.documents import RetrievedChunk


# -- tokenization -----------------------------------------------------------

def test_tokenize_english_and_arabic():
    toks = tokenize("New Mansoura University جامعة المنصورة الجديدة")
    assert "new" in toks
    assert "mansoura" in toks
    # Arabic is light-normalized (hamza unification, ة->ه, ال stripped).
    assert "جامعه" in toks  # جامعة -> جامعه
    assert "منصوره" in toks  # المنصورة -> منصوره
    assert all(not t.isupper() for t in toks)


def test_tokenize_arabic_article_and_hamza_forms():
    assert tokenize("الشروط القبول أين الأعمال") == ["شروط", "قبول", "اين", "اعمال"]


def test_tokenize_drops_punctuation_and_whitespace():
    assert tokenize("  Hello, (NMU)!   ") == ["hello", "nmu"]


# -- BM25 --------------------------------------------------------------------

def _index() -> BM25Index:
    texts = {
        "a": "New Mansoura University is located in New Mansoura City.",
        "b": "The Faculty of Engineering offers civil and computer programs.",
        "c": "شروط القبول في جامعة المنصورة الجديدة تشمل الثانوية العامة.",
        "d": "Mansoura university tuition fees are announced each year.",
    }
    return BM25Index(texts)


def test_bm25_finds_exact_terms():
    idx = _index()
    hits = idx.search("tuition fees", top_k=4)
    assert hits[0][0] == "d"
    assert hits[0][1] > 0


def test_bm25_arabic_terms():
    idx = _index()
    hits = idx.search("شروط القبول", top_k=4)
    assert hits[0][0] == "c"


def test_bm25_rrf_prefers_joint_hits():
    idx = _index()
    both = idx.rrf_score(0, 0)
    dense_only = idx.rrf_score(0, None)
    assert both > dense_only


# -- query normalization -----------------------------------------------------

def test_normalize_collapses_whitespace_and_strips_diacritics():
    out = normalize_query("  جامعةُ   المنصورةَ   ")
    assert "  " not in out
    assert "\u064b" not in out  # no diacritics


def test_normalize_alef_forms():
    assert normalize_query("أحمد إبراهيم آية") == normalize_query("احمد ابراهيم اية")


def test_normalize_colloquial_ayh():
    out = normalize_query("ايه البرامج الموجوده في كليه الهندسه؟")
    assert "ما هي" in out
    assert "البرامج" in out
    assert "تقع موقع" not in out


def test_normalize_cse_aliases():
    ar = normalize_query("ايه تخصصات كلية الحاسب؟")
    en = normalize_query("CSE programs at NMU")
    assert "علوم وهندسة الحاسب" in ar
    assert "Computer Science Engineering" in en


def test_expand_adds_intent_keywords():
    base = normalize_query("What faculties are there?")
    out = expand_query(base, "FACULTY", max_terms=8)
    assert "colleges" in out
    assert "كليات" in out


def test_expand_does_not_repeat_existing_words():
    base = normalize_query("What faculties and colleges?")
    out = expand_query(base, "FACULTY", max_terms=8)
    assert out.count("colleges") == 1


def test_is_arabic_dominant_script():
    assert is_arabic("ما هي كليات الجامعة؟")
    assert is_arabic("جامعة المنصورة الجديدة Faculty")
    assert not is_arabic("Where is the university?")


# -- RRF fusion --------------------------------------------------------------

def _dense_chunk(cid, text, dense_score):
    return RetrievedChunk(
        chunk_id=cid,
        document_id="d1",
        text=text,
        score=dense_score,
        dense_score=dense_score,
        title="T",
        source_url=f"https://nmu.edu.eg/{cid}",
        content_type="home",
    )


def _retriever_with_chunks(chunks) -> Retriever:
    r = Retriever.__new__(Retriever)
    r._chunk_map = {c.chunk_id: c for c in chunks}
    r._bm25 = BM25Index({c.chunk_id: c.text for c in chunks})
    r.candidate_k = 10
    r.hybrid_enabled = True
    r.query_expansion_enabled = True
    r.max_query_expansion_terms = 8
    r.hybrid_fusion = "rrf"
    r.dense_weight = 0.65
    r.bm25_weight = 0.35
    return r


def test_fuse_keeps_bm25_only_chunks():
    dense_chunks = [_dense_chunk("d1", "location of university", 0.8)]
    bm25_only = RetrievedChunk(
        chunk_id="b1",
        document_id="d2",
        text="New Mansoura University located in New Mansoura City",
        score=0.0,
        title="T2",
        source_url="https://nmu.edu.eg/b1",
        content_type="about",
    )
    r = _retriever_with_chunks(dense_chunks + [bm25_only])
    hits = r._bm25.search("New Mansoura located", top_k=10)
    out = r._fuse(dense_chunks, hits)
    assert any(c.chunk_id == "b1" for c in out)
    b = next(c for c in out if c.chunk_id == "b1")
    assert b.dense_score is None  # lexical-only -> threshold bypass
    assert b.bm25_score is not None and b.bm25_score > 0


def test_fuse_joint_hits_rank_above_single_system():
    dense = [_dense_chunk("d1", "located in New Mansoura City", 0.9)]
    lexical_other = RetrievedChunk(
        chunk_id="b1", document_id="d2", text="Faculty of engineering programs",
        score=0.0, title="T", source_url="https://nmu.edu.eg/b1", content_type="program",
    )
    r = _retriever_with_chunks(dense + [lexical_other])
    hits = r._bm25.search("New Mansoura City located", top_k=10)
    out = r._fuse(dense, hits)
    assert out[0].chunk_id == "d1"  # matches both dense + bm25

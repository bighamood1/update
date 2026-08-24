"""Tests for the Phase 4 recovery fixes (no models, no Ollama, no DB).

Covers:
- config defaults for the recovery knobs,
- list-intent context budget (no more truncated directory pages),
- compressor keeping whole list blocks under the list token budget,
- pipeline gating: refusals / invalid answers are never cached or written to
  retrieval memory; validated answers are.
"""

from __future__ import annotations

import pytest

from rag.config import get_config as real_get_config
from rag.context.builder import ContextBuilder
from rag.context.compressor import ContextCompressor
from rag.generation.validation import REFUSAL_AR, REFUSAL_EN, refusal_text
from rag.pipeline.rag import RAGPipeline
from rag.query.understanding import understand
from rag.schemas.documents import RetrievedChunk


def _chunk(
    chunk_id: str,
    text: str,
    *,
    source_url: str = "https://nmu.edu.eg/en/page",
    content_type: str = "news",
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


class _FakeVectorStore:
    def is_built(self) -> bool:
        return True

    def compatibility_errors(self) -> list:
        return []

    def compatibility_warnings(self) -> list:
        return []

    def kb_version(self) -> str:
        return "test-kb"


class _FakeEmbedder:
    def __getattr__(self, name):  # never used in these tests
        raise AssertionError(f"Embedder.{name} should not be called")


class _FakeRetriever:
    def __init__(self, chunks: list[RetrievedChunk]) -> None:
        self.chunks = chunks
        self.last_timings: dict = {}
        self.last_meta: dict = {}
        self.list_source_types = {"program", "faculty", "about", "administration"}

    def retrieve(self, question, intent=None, route=None, query_variants=None,
                 memory_seed_urls=None, candidate_factor=None, force_refresh=False):
        self.last_meta = {
            "candidate_count": len(self.chunks),
            "reranked_count": len(self.chunks),
            "final_count": len(self.chunks),
            "top_k_used": len(self.chunks),
            "routed": False,
            "force_refresh": force_refresh,
        }
        return list(self.chunks)


class _FakeOllama:
    def __init__(self, answers: list[str]) -> None:
        self.answers = list(answers)
        self.model = "qwen3-vl:8b"
        self.calls = 0

    def generate(self, system_prompt: str, user_prompt: str, **kwargs) -> str:
        self.calls += 1
        return self.answers.pop(0) if self.answers else ""


class _FakeCache:
    def __init__(self) -> None:
        self.stored: list[dict] = []
        self.looked_up = 0

    def lookup(self, question, understanding, kb_version):
        self.looked_up += 1
        return None, None

    def store(self, **kwargs) -> None:
        self.stored.append(kwargs)


class _FakeMemory:
    def __init__(self) -> None:
        self.remembered: list[dict] = []

    def hint(self, kb_version, normalized_question, **kwargs) -> list[str]:
        return []

    def should_diversify(self, kb_version, normalized_question, understanding) -> bool:
        return False

    def remember(self, **kwargs) -> None:
        self.remembered.append(kwargs)


def _build_pipeline(ollama_answers: list[str], chunks: list[RetrievedChunk]):
    """A RAGPipeline with every model/DB dependency replaced by fakes."""
    cache = _FakeCache()
    memory = _FakeMemory()
    pipeline = RAGPipeline(
        vectorstore=_FakeVectorStore(),
        embedder=_FakeEmbedder(),
        retriever=_FakeRetriever(chunks),
        ollama=_FakeOllama(ollama_answers),
        context_builder=ContextBuilder(),
    )
    pipeline._semantic_cache_instance = cache
    pipeline._memory_instance = memory
    return pipeline, cache, memory


@pytest.fixture
def recovery_cfg(monkeypatch):
    """Config where feedback is disabled (no runtime DB writes) but the
    recovery flags are on."""
    cfg = dict(real_get_config())
    cfg["feedback_enabled"] = False
    cfg["cache_enabled"] = True
    cfg["retrieval_memory_enabled"] = True
    cfg["memory_gate_on_quality"] = True
    cfg["memory_min_quality_score"] = 0.5
    cfg["fast_path_enabled"] = True
    monkeypatch.setattr("rag.pipeline.rag.get_config", lambda: cfg)
    return cfg


# -- config defaults ---------------------------------------------------------


class TestConfigDefaults:
    def test_recovery_defaults(self):
        cfg = real_get_config()
        assert cfg["similarity_threshold"] <= 0.25
        assert cfg["cache_min_quality_score"] >= 0.6
        assert cfg["context_max_chars_list"] >= 2 * cfg["context_max_chars"]
        assert cfg["context_max_tokens_list"] > cfg["max_context_tokens"]
        assert cfg["memory_gate_on_quality"] is True
        assert cfg["memory_min_quality_score"] >= 0.5


# -- context budget for lists -------------------------------------------------


class TestListContextBudget:
    def test_list_budget_keeps_more_evidence(self):
        long_text = " ".join([f"Item {i} of the official list page." for i in range(220)])
        assert len(long_text) > 4000
        chunks = [
            _chunk("a", long_text, content_type="program"),
            _chunk("b", "A second supporting paragraph.", content_type="about"),
        ]
        small = ContextBuilder().build(chunks, max_chunks=2, max_chars=4000)
        big = ContextBuilder().build(chunks, max_chunks=2, max_chars=8000)
        assert len(big) > len(small)
        assert "Item 100" in big

    def test_list_compressor_keeps_whole_block(self):
        long_text = "word " * 5000  # ~5000 tokens, no sentence punctuation
        kept = ContextCompressor(max_tokens=8000).compress(long_text)
        assert kept.strip() == long_text.strip()
        cut = ContextCompressor(max_tokens=4096).compress(long_text)
        assert cut != long_text


# -- pipeline gating ----------------------------------------------------------


_LOCATION_Q = "أين تقع جامعة المنصورة الجديدة؟"
_VALID_AR = "تقع جامعة المنصورة الجديدة في مدينة المنصورة الجديدة بمحافظة الدقهلية"


class TestPipelineGating:
    def test_valid_answer_is_cached_and_remembered(self, recovery_cfg):
        chunks = [_chunk("n1", "News item that must not trigger the fast path.")]
        pipeline, cache, memory = _build_pipeline([_VALID_AR], chunks)
        result = pipeline.ask(_LOCATION_Q)
        assert result.answer.strip() == _VALID_AR
        assert cache.stored, "validated answer should be cached"
        assert memory.remembered, "validated answer should be remembered"

    def test_refusal_never_cached_or_remembered(self, recovery_cfg):
        chunks = [_chunk("n1", "News item that must not trigger the fast path.")]
        pipeline, cache, memory = _build_pipeline(["", ""], chunks)
        result = pipeline.ask(_LOCATION_Q)
        assert result.answer.strip() == refusal_text("ar").strip()
        assert not cache.stored, "refusal must not be cached"
        assert not memory.remembered, "refusal must not be remembered"

    def test_offtopic_answer_invalidated_not_remembered(self, recovery_cfg):
        # A location question answered with the founding decree -> relevance
        # gate regenerates, then refuses.
        decree = "صدر قرار رئيس جمهورية مصر العربية رقم 437 لسنة 2020 بإنشاء الجامعة."
        chunks = [_chunk("n1", "News item that must not trigger the fast path.")]
        pipeline, cache, memory = _build_pipeline([decree, decree], chunks)
        result = pipeline.ask(_LOCATION_Q)
        assert not result.answer.strip() or result.answer.strip() == refusal_text("ar").strip()
        assert not cache.stored
        assert not memory.remembered

    def test_memory_quality_gate_direct(self, recovery_cfg):
        pipeline, _cache, memory = _build_pipeline([_VALID_AR], [])
        understanding = understand(_LOCATION_Q)
        sources = [{"url": "https://nmu.edu.eg/en/page", "title": "t"}]
        pipeline._memory_remember(
            understanding, "test-kb", sources, "routed", quality_score=0.3
        )
        assert not memory.remembered, "low-quality answer must not be remembered"
        pipeline._memory_remember(
            understanding, "test-kb", sources, "routed", quality_score=0.9
        )
        assert memory.remembered

    def test_cache_store_skips_refusal_text(self, recovery_cfg):
        pipeline, cache, _memory = _build_pipeline([_VALID_AR], [])
        understanding = understand(_LOCATION_Q)
        pipeline._cache_store(
            "q", understanding, "test-kb", REFUSAL_AR, [], 0.9, None
        )
        pipeline._cache_store(
            "q", understanding, "test-kb", REFUSAL_EN, [], 0.9, None
        )
        pipeline._cache_store(
            "q", understanding, "test-kb", "", [], 0.9, None
        )
        assert not cache.stored

    def test_not_useful_feedback_skips_cache_and_fast_path(self, monkeypatch, recovery_cfg):
        recovery_cfg["feedback_enabled"] = True
        recovery_cfg["retrieval_memory_enabled"] = False
        location_text = (
            "تقع جامعة المنصورة الجديدة في مدينة المنصورة الجديدة بمحافظة الدقهلية، "
            "على الطريق الساحلي الدولي."
        )
        chunks = [_chunk("loc", location_text, content_type="about", language="ar")]
        pipeline, cache, _memory = _build_pipeline([location_text], chunks)

        class _FeedbackStore:
            def semantic_group(self, **kwargs):
                return "|".join([
                    kwargs.get("intent") or "",
                    kwargs.get("language") or "",
                    kwargs.get("category") or "",
                    kwargs.get("faculty") or "general",
                    kwargs.get("topic") or "",
                    kwargs.get("subtopic") or "",
                ])

            def question_fingerprint(self, normalized_question):
                return "fp"

            def latest_feedback_for_query(self, **kwargs):
                return {
                    "question_id": "q-old",
                    "rating": "not_useful",
                    "answer": "old fast-path answer",
                    "sources_json": "[]",
                }

            def record_question_event(self, **kwargs):
                pass

            def record_cluster(self, **kwargs):
                pass

        monkeypatch.setattr("rag.pipeline.rag.get_runtime_store", lambda: _FeedbackStore())
        result = pipeline.ask(_LOCATION_Q)
        assert cache.looked_up == 0
        assert pipeline.retriever.last_meta["force_refresh"] is True
        assert pipeline.ollama.calls == 1
        assert result.llm_used is True
        assert result.answer == location_text

    def test_reasoning_or_source_leakage_triggers_controlled_regeneration(self, recovery_cfg):
        recovery_cfg["fast_path_enabled"] = False
        chunks = [_chunk(
            "loc",
            "New Mansoura University is located in New Mansoura City, Dakahlia Governorate.",
            content_type="about",
            language="en",
        )]
        pipeline, cache, memory = _build_pipeline(
            [
                "According to Source 1, New Mansoura University is located in New Mansoura City.",
                "New Mansoura University is located in New Mansoura City, Dakahlia Governorate.",
            ],
            chunks,
        )
        result = pipeline.ask("Where is New Mansoura University located?")
        assert pipeline.ollama.calls == 2
        assert "Source 1" not in result.answer
        assert result.answer.startswith("New Mansoura University is located")
        assert cache.stored
        assert memory.remembered

    def test_incomplete_list_answer_triggers_controlled_regeneration(self, recovery_cfg):
        recovery_cfg["fast_path_enabled"] = False
        list_chunk = _chunk(
            "fac",
            "Business\nLaw\nEngineering\nComputer Science\nMedicine",
            content_type="program",
            language="en",
        )
        pipeline, _cache, _memory = _build_pipeline(
            [
                "- Business\n- Law",
                "- Business\n- Law\n- Engineering\n- Computer Science\n- Medicine",
            ],
            [list_chunk],
        )
        result = pipeline.ask("What faculties are available at New Mansoura University?")
        assert pipeline.ollama.calls == 2
        assert "Engineering" in result.answer
        assert "Computer Science" in result.answer
        assert "Medicine" in result.answer


# -- semantic cache shadowing regression --------------------------------------


class TestSemanticCacheShadowing:
    def test_store_method_not_shadowed_by_runtime_store_attribute(self):
        """Regression: the RuntimeStore instance was stored as self.store,
        shadowing the SemanticCache.store() method, so every cache write
        raised ``TypeError: 'RuntimeStore' object is not callable`` which was
        swallowed — the semantic cache silently never persisted anything."""
        from rag.cache.semantic_cache import SemanticCache

        class _FakeEmbedder:
            def embed_query(self, text: str) -> list[float]:
                return [0.0] * 384

        class _FakeStore:
            enabled = True
            written = []

            def find_cache_hits(self, kb_version):
                return []

            def upsert_cache_entry(self, **kwargs) -> int:
                self.written.append(kwargs)
                return 7

            def remember_kb_version(self, kb_version) -> None:
                pass

            def bump_cache_usage(self, entry_id: int) -> None:
                pass

        fake = _FakeStore()
        cache = SemanticCache(embedder=_FakeEmbedder(), store=fake)
        assert callable(cache.store), (
            "SemanticCache.store() must not be shadowed by the runtime store"
        )
        understanding = understand(_LOCATION_Q)
        cache.store(
            kb_version="kb-v1", question="q", understanding=understanding,
            answer="answer text", sources=[], quality_score=0.9,
        )
        assert fake.written, "store() must reach upsert_cache_entry"
        assert fake.written[0]["answer"] == "answer text"

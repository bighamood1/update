"""Centralized configuration for the NMU RAG system.

All runtime settings are read from environment variables, with sensible
defaults. A local ``.env`` file (never committed) can override defaults.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

# Project root is the parent of the ``src`` package directory.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Load .env from the project root if present (missing file is fine).
load_dotenv(PROJECT_ROOT / ".env")


def _parse_source_priority(raw: str) -> dict[str, float]:
    """Parse ``type:weight,type:weight`` into a dict of float weights.

    Unknown types default to 1.0. Malformed entries are skipped.
    """
    result: dict[str, float] = {}
    for token in raw.split(","):
        token = token.strip()
        if not token or ":" not in token:
            continue
        key, _, value = token.partition(":")
        try:
            result[key.strip().lower()] = float(value)
        except ValueError:
            continue
    return result


def _split_csv(raw: str) -> list[str]:
    return [part.strip() for part in raw.split(",") if part.strip()]


def _parse_intent_chunks(raw: str, defaults: dict[str, int]) -> dict[str, int]:
    """Parse ``intent:chunks,intent:chunks`` over a default mapping.

    Keys are lowercased and unknown intents are skipped so a typo in the
    environment never crashes startup.
    """
    result = dict(defaults)
    for token in raw.split(","):
        token = token.strip()
        if not token or ":" not in token:
            continue
        key, _, value = token.partition(":")
        try:
            result[key.strip().lower()] = int(value)
        except ValueError:
            continue
    return result


def _resolve(path_value: str) -> Path:
    """Resolve a configured path either as absolute or relative to project root."""
    p = Path(path_value).expanduser()
    if p.is_absolute():
        return p
    return PROJECT_ROOT / p


@lru_cache(maxsize=1)
def get_config() -> dict:
    """Return the full configuration dictionary (cached after first call)."""
    return {
        # --- Ollama / LLM ------------------------------------------------
        "ollama_base_url": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        # Empty means "auto": pick the first installed model from
        # OLLAMA_PREFERRED_MODELS. An explicit value must be installed or the
        # server refuses to start with a clear error. On this CPU-only machine
        # a small text model (qwen3:4b / qwen3:1.7b) is far faster than
        # qwen3-vl:8b; pull one with `ollama pull qwen3:4b` and set OLLAMA_MODEL.
        "ollama_model": os.getenv("OLLAMA_MODEL", "").strip(),
        # Preference order for automatic model selection (never downloaded).
        "ollama_preferred_models": _split_csv(
            os.getenv(
                "OLLAMA_PREFERRED_MODELS",
                "qwen3:4b,qwen3:1.7b,qwen3:8b,qwen3:0.6b,qwen3-vl:8b",
            )
        ),
        "ollama_timeout": float(os.getenv("OLLAMA_TIMEOUT", "600")),
        "ollama_options": {
            "temperature": float(os.getenv("OLLAMA_TEMPERATURE", "0.1")),
            "top_p": float(os.getenv("OLLAMA_TOP_P", "0.9")),
            "num_ctx": int(os.getenv("OLLAMA_NUM_CTX", "8192")),
# Generation budget. Conservative default: qwen3-vl:8b on CPU can
        # emit a large internal "thinking" block even with think=false, so
        # an overly small cap truncates the visible answer to empty. Raise
        # via OLLAMA_NUM_PREDICT / OLLAMA_MAX_OUTPUT_TOKENS for long lists.
        # The safe output limit is the authoritative ``ollama_max_output_tokens``
        # key (OLLAMA_MAX_OUTPUT_TOKENS / OLLAMA_NUM_PREDICT).
        "num_predict": int(
            os.getenv("OLLAMA_NUM_PREDICT")
            or os.getenv("OLLAMA_MAX_OUTPUT_TOKENS", "2500")
        ),
        "think": os.getenv("OLLAMA_THINK", "false").lower() == "true",
        },
        "ollama_max_output_tokens": int(
            os.getenv("OLLAMA_NUM_PREDICT")
            or os.getenv("OLLAMA_MAX_OUTPUT_TOKENS", "2500")
        ),
        # How many threads the embedding/reranker models may use on CPU
        # (0 = let the framework decide). Pinning avoids oversubscription on
        # the shared i7-1165G7 (4 cores / 8 threads).
        "cpu_threads": int(os.getenv("CPU_THREADS", "0")),
        # How long the model stays resident in memory between requests
        # (seconds). Keeping it warm avoids long cold-start timeouts.
        "ollama_keep_alive": os.getenv("OLLAMA_KEEP_ALIVE", "30m"),
        # --- Data / paths ------------------------------------------------
        "data_path": _resolve(os.getenv("DATA_PATH", "data/documents.jsonl")),
        "vector_db_path": _resolve(os.getenv("VECTOR_DB_PATH", "vectorstore")),
        "logs_dir": _resolve(os.getenv("LOGS_DIR", "logs")),
        # --- Chunking -----------------------------------------------------
        "chunk_size": int(os.getenv("CHUNK_SIZE", "800")),
        "chunk_overlap": int(os.getenv("CHUNK_OVERLAP", "100")),
        "min_chunk_chars": int(os.getenv("MIN_CHUNK_CHARS", "40")),
        # --- Embeddings ----------------------------------------------------
        "embedding_model": os.getenv(
            "EMBEDDING_MODEL", "intfloat/multilingual-e5-small"
        ),
        "embedding_device": os.getenv("EMBEDDING_DEVICE", "cpu"),
        "embedding_batch_size": int(os.getenv("EMBEDDING_BATCH_SIZE", "32")),
        "embedding_normalize": os.getenv("EMBEDDING_NORMALIZE", "true").lower() == "true",
        # E5-style models (intfloat/multilingual-e5-*) require explicit
        # "query:" / "passage:" prefixes for best results.
        "embedding_query_prefix": os.getenv("EMBEDDING_QUERY_PREFIX", "query: "),
        "embedding_passage_prefix": os.getenv("EMBEDDING_PASSAGE_PREFIX", "passage: "),
        # --- Retrieval ------------------------------------------------------
        "top_k": int(os.getenv("TOP_K") or os.getenv("RETRIEVAL_TOP_K", "8")),
        # Number of candidate chunks that reach the reranker / final pass.
        # (Friendly name; CANDIDATE_K kept as a raw alias.)
        "rerank_candidates": int(
            os.getenv("RERANK_CANDIDATES") or os.getenv("CANDIDATE_K", "20")
        ),
        "candidate_k": int(
            os.getenv("RERANK_CANDIDATES") or os.getenv("CANDIDATE_K", "20")
        ),
        "similarity_threshold": float(os.getenv("SIMILARITY_THRESHOLD", "0.25")),
        # Hybrid retrieval: lexical BM25 fused with dense via RRF.
        "hybrid_enabled": os.getenv("HYBRID_ENABLED", "true").lower() == "true",
        "rrf_k": int(os.getenv("RRF_K", "60")),
        # Local query normalization + keyword expansion (no LLM involved).
        "query_expansion_enabled": os.getenv("QUERY_EXPANSION_ENABLED", "true").lower()
        == "true",
        "max_query_expansion_terms": int(os.getenv("MAX_QUERY_EXPANSION_TERMS", "8")),
        # Number of chunks / characters actually sent to the LLM.
        "final_context_chunks": int(os.getenv("FINAL_CONTEXT_CHUNKS", "4")),
        "context_max_chars": int(os.getenv("CONTEXT_MAX_CHARS", "4000")),
        # Recovery (Phase 4): list-bearing intents (LIST/FACULTY/PROGRAM/...) get
        # a LARGER char budget so a complete directory page survives into the
        # context. Without this the 4000-char cap truncated long list pages and
        # the LLM returned incomplete lists ("incomplete answers"). The
        # compressor gets the matching token budget so it never drops list items
        # mid-section. Toggle with CONTEXT_MAX_CHARS_LIST / CONTEXT_MAX_TOKENS_LIST.
        "context_max_chars_list": int(
            os.getenv("CONTEXT_MAX_CHARS_LIST", "8000")
        ),
        "context_max_tokens_list": int(
            os.getenv("CONTEXT_MAX_TOKENS_LIST", "8000")
        ),
        # Deterministic query routing (metadata-first retrieval).
        "router_enabled": os.getenv("ROUTER_ENABLED", "true").lower() == "true",
        "router_confidence_threshold": float(
            os.getenv("ROUTER_CONFIDENCE_THRESHOLD", "0.80")
        ),
        "router_min_results": int(os.getenv("ROUTER_MIN_RESULTS", "3")),
        "router_fallback_enabled": os.getenv("ROUTER_FALLBACK_ENABLED", "true").lower()
        == "true",
        "router_filter_language": os.getenv("ROUTER_FILTER_LANGUAGE", "true").lower()
        == "true",
        "router_filter_category": os.getenv("ROUTER_FILTER_CATEGORY", "true").lower()
        == "true",
        "router_filter_faculty": os.getenv("ROUTER_FILTER_FACULTY", "true").lower()
        == "true",
        # Dynamic top-k: final context size varies by query complexity.
        "dynamic_top_k_enabled": os.getenv("DYNAMIC_TOP_K_ENABLED", "true").lower()
        == "true",
        "top_k_fact": int(os.getenv("TOP_K_FACT", "4")),
        "top_k_list": int(os.getenv("TOP_K_LIST", "6")),
        "top_k_complex": int(os.getenv("TOP_K_COMPLEX", "8")),
        # Intent-aware context sizing: how many chunk groups may reach the LLM
        # per intent (upper-bounded by MAX_CONTEXT_CHUNKS). Keeps generation
        # prompts small for simple facts while allowing full evidence for
        # lists/comparisons. The retriever's top_k is NOT reduced here.
        "intent_context_chunks": _parse_intent_chunks(
            os.getenv(
                "INTENT_CONTEXT_CHUNKS",
                "fact:3,location:3,contact:3,tuition:3,person:3,faq:3,"
                "news:4,facility:4,scholarship:4,unknown:4,admission:5,"
                "regulation:5,administration:5,faculty:6,program:6,list:6,"
                "comparison:6",
            ),
            {
                "fact": 3,
                "location": 3,
                "contact": 3,
                "tuition": 3,
                "person": 3,
                "faq": 3,
                "news": 4,
                "facility": 4,
                "scholarship": 4,
                "unknown": 4,
                "admission": 5,
                "regulation": 5,
                "administration": 5,
                "faculty": 6,
                "program": 6,
                "list": 6,
                "comparison": 6,
            },
        ),
        # Deterministic fast answers for highly structured queries (location,
        # faculties list, contact info) extracted from authoritative indexed
        # sources. Never activates without a matching authoritative chunk.
        "fast_path_enabled": os.getenv("FAST_PATH_ENABLED", "true").lower() == "true",
        # Fast answers activate ONLY when the deterministic router is confident
        # (default 0.55 > the weak-signal 0.53 level, so e.g. a low-confidence
        # FACULTY question never auto-answers). The authoritative-chunk
        # requirement still applies on top of this gate.
        "fast_path_min_confidence": float(
            os.getenv("FAST_PATH_MIN_CONFIDENCE", "0.55")
        ),
        # --- Query understanding / expansion ---------------------------------
        "query_understanding_enabled": os.getenv(
            "QUERY_UNDERSTANDING_ENABLED", "true"
        ).lower() == "true",
        # Bounded number of lexical retrieval variants (Phase 3). Simple
        # questions use only the normalized query; ambiguous ones add the
        # original + intent-expanded text (never more than this many).
        "max_retrieval_variants": int(os.getenv("MAX_RETRIEVAL_VARIANTS", "5")),
        "multi_intent_enabled": os.getenv("MULTI_INTENT_ENABLED", "true").lower()
        == "true",
        "multi_intent_confidence": float(os.getenv("MULTI_INTENT_CONFIDENCE", "0.55")),
        # --- Adaptive retrieval ------------------------------------------------
        "adaptive_retrieval_enabled": os.getenv(
            "ADAPTIVE_RETRIEVAL_ENABLED", "true"
        ).lower() == "true",
        # High-confidence routed queries may search a smaller candidate window;
        # multi-intent sub-queries use a reduced per-intent pool. Bounded so
        # recall is never silently destroyed.
        "adaptive_narrow_factor": float(os.getenv("ADAPTIVE_NARROW_FACTOR", "0.8")),
        "adaptive_multi_factor": float(os.getenv("ADAPTIVE_MULTI_FACTOR", "0.6")),
        "adaptive_min_candidates": int(os.getenv("ADAPTIVE_MIN_CANDIDATES", "10")),
        # --- Retrieval memory (learned patterns) --------------------------------
        "retrieval_memory_enabled": os.getenv("RETRIEVAL_MEMORY_ENABLED", "true").lower()
        == "true",
        # How many historical source URLs may be seeded into the candidate pool
        # as soft hints (always re-ranked and still subject to fallback).
        "memory_seed_urls": int(os.getenv("MEMORY_SEED_URLS", "2")),
        # Recovery (Phase 4): retrieval memory is ONLY written for answers that
        # passed the hard validation gate (non-refusal, grounded). This stops a
        # wrong or refused answer's sources from being re-seeded forever, which
        # previously produced "repeats old/wrong answers". Toggle the gate with
        # MEMORY_GATE_ON_QUALITY and set the required score with
        # MEMORY_MIN_QUALITY_SCORE.
        "memory_gate_on_quality": os.getenv("MEMORY_GATE_ON_QUALITY", "true").lower()
        == "true",
        "memory_min_quality_score": float(
            os.getenv("MEMORY_MIN_QUALITY_SCORE", "0.5")
        ),
        # --- Semantic response cache (persistent) -------------------------------
        "cache_similarity_threshold": float(
            os.getenv("CACHE_SIMILARITY_THRESHOLD", "0.92")
        ),
        # Stricter similarity required when either side of a candidate pair is a
        # generic/uncertain intent (LIST/FACT/GENERAL/FAQ/UNKNOWN): the entity
        # name alone can otherwise push different questions over the threshold.
        "cache_generic_similarity_threshold": float(
            os.getenv("CACHE_GENERIC_SIMILARITY_THRESHOLD", "0.97")
        ),
        "cache_min_quality_score": float(
            os.getenv("CACHE_MIN_QUALITY_SCORE", "0.6")
        ),
        "cache_max_entries": int(os.getenv("CACHE_MAX_ENTRIES", "2000")),
        # Runtime SQLite database (feedback + cache + clusters + memory). It is
        # isolated from ChromaDB and never touches the vector data.
        "runtime_db_path": _resolve(
            os.getenv("RUNTIME_DB_PATH", "data/runtime/nmu_runtime.db")
        ),
        # --- Feedback / analytics ------------------------------------------------
        "feedback_enabled": os.getenv("FEEDBACK_ENABLED", "true").lower() == "true",
        "feedback_reasons_enabled": os.getenv(
            "FEEDBACK_REASONS_ENABLED", "true"
        ).lower() == "true",
        # --- Answer quality -------------------------------------------------------
        "quality_validation_enabled": os.getenv(
            "QUALITY_VALIDATION_ENABLED", "true"
        ).lower() == "true",
        # Soft cap: a validated answer above this length is flagged as verbose
        # unless the intent asks for a list/comparison.
        "answer_max_chars": int(os.getenv("ANSWER_MAX_CHARS", "4000")),
        # --- Resource limits.
        "max_retrieval_results": int(os.getenv("MAX_RETRIEVAL_RESULTS", "30")),
        "max_rerank_results": int(os.getenv("MAX_RERANK_RESULTS", "20")),
        # Hard upper bound on how many chunks may actually reach the LLM
        # (dedup + source diversity happen earlier in the retriever). Simple
        # intents use fewer via INTENT_CONTEXT_CHUNKS; this caps every intent.
        "top_context_chunks": int(
            os.getenv("TOP_CONTEXT_CHUNKS")
            or os.getenv("MAX_CONTEXT_CHUNKS", "6")
        ),
        "max_context_chunks": int(os.getenv("MAX_CONTEXT_CHUNKS", "6")),
        "max_context_tokens": int(os.getenv("MAX_CONTEXT_TOKENS", "4096")),
        "max_generation_tokens": int(os.getenv("MAX_GENERATION_TOKENS", "2500")),
        # How many RAG generations may run simultaneously (CPU safety). A
        # request that arrives when all slots are busy gets a controlled
        # ``busy`` response instead of crashing or exhausting RAM.
        "max_concurrent_generations": int(
            os.getenv("MAX_CONCURRENT_GENERATIONS", "1")
        ),
        # Caching (embedding + retrieval result).
        "cache_enabled": os.getenv("CACHE_ENABLED", "true").lower() == "true",
        "cache_embedding_size": int(os.getenv("CACHE_EMBEDDING_SIZE", "512")),
        "cache_retrieval_size": int(os.getenv("CACHE_RETRIEVAL_SIZE", "256")),
        "cache_retrieval_ttl": int(os.getenv("CACHE_RETRIEVAL_TTL", "3600")),
        # Hybrid fusion weights (used for weighted linear fusion; RRF is used
        # when HYBRID_FUSION=rrf, the default).
        "dense_weight": float(os.getenv("DENSE_WEIGHT", "0.65")),
        "bm25_weight": float(os.getenv("BM25_WEIGHT", "0.35")),
        "hybrid_fusion": os.getenv("HYBRID_FUSION", "rrf").lower(),
        "reranker_enabled": (
            os.getenv("RERANKER_ENABLED")
            or os.getenv("ENABLE_RERANKER", "true")
        ).lower() == "true",
        "reranker_model": os.getenv(
            "RERANKER_MODEL", "BAAI/bge-reranker-base"
        ),
        # Cross-encoder runs locally on CPU: an explicit device and a bounded
        # batch keep CPU time and peak memory predictable (RERANKER_DEVICE,
        # RERANKER_BATCH_SIZE).
        "reranker_device": os.getenv("RERANKER_DEVICE", "cpu"),
        "reranker_batch_size": int(os.getenv("RERANKER_BATCH_SIZE", "32")),
        "rerank_top_k": int(os.getenv("RERANK_TOP_K", "8")),
        # Source diversity: cap how many chunks from one source URL may enter
        # the final context (0 = unlimited). Distinct informative chunks of the
        # same document are still allowed up to this cap.
        "max_chunks_per_source": int(os.getenv("MAX_CHUNKS_PER_SOURCE", "3")),
        # List queries: how many extra chunks per source to fetch when the
        # retriever expands to recover complete lists/sections.
        "expansion_chunks_per_source": int(
            os.getenv("EXPANSION_CHUNKS_PER_SOURCE", "12")
        ),
        # Content types considered authoritative list/directory sources.
        "list_source_types": _split_csv(
            os.getenv(
                "LIST_SOURCE_TYPES",
                "program,about,faculty,administration,faq,tuition,admission",
            )
        ),
        # Canonical directory pages that hold the complete list for common
        # list questions (faculties, programs). Injected into the candidate
        # pool for list intents so a terse index page is never drowned out by
        # individual detail pages.
        "list_seed_urls": _split_csv(
            os.getenv(
                "LIST_SEED_URLS",
                "https://nmu.edu.eg/en/all-faculties-programs,"
                "https://nmu.edu.eg/ar/all-faculties-programs,"
                "https://www.nmu.edu.eg/en/all-faculties-programs,"
                "https://www.nmu.edu.eg/ar/all-faculties-programs",
            )
        ),
        "list_seed_enabled": os.getenv("LIST_SEED_ENABLED", "true").lower() == "true",
        # Content-type priority used as a light relevance tie-breaker /
        # boost. Higher value = more authoritative. Configurable per type.
        "source_priority": _parse_source_priority(
            os.getenv(
                "SOURCE_PRIORITY",
                "program:1.02,about:1.02,faculty:1.01,admission:1.0,"
                "tuition:1.0,scholarship:1.0,faq:1.0,president:1.0,"
                "administration:1.0,contact:1.0,news:0.99,regulation:0.98,"
                "home:1.0,event:1.0,facility:1.0,policy:0.99,guide:1.0,"
                "career:1.0,tender:0.99",
            )
        ),
        # --- Index build ------------------------------------------------------
        "chroma_collection": os.getenv("CHROMA_COLLECTION", "nmu_documents"),
        "force_rebuild": os.getenv("FORCE_REBUILD", "false").lower() == "true",
        # --- Misc ---------------------------------------------------------------
        "log_level": os.getenv("LOG_LEVEL", "INFO"),
    }


class Settings:
    """Simple attribute-style accessor over the configuration dict."""

    def __init__(self, config: dict | None = None) -> None:
        self._config = config if config is not None else get_config()

    def __getattr__(self, name: str):
        try:
            return self._config[name]
        except KeyError as exc:  # pragma: no cover - defensive
            raise AttributeError(f"Unknown setting: {name}") from exc

    @property
    def as_dict(self) -> dict:
        return dict(self._config)


settings = Settings()

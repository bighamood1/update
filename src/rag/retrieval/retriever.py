"""Two-stage hybrid retriever with routing, reranking and list expansion.

Architecture
------------
Stage 0 — deterministic routing: intent, category, faculty, language and a
confidence score are computed locally (no LLM). When confidence is high, a
safe Chroma metadata filter (language / content_type / faculty) is applied
as PART of vector retrieval (never in Python afterward).

Stage 1a — dense retrieval (``multilingual-e5-small``) over the candidate
window, optionally restricted by the router's metadata ``where`` clause.

Stage 1b — lexical BM25 (Okapi) over a normalized + expanded query.

Stage 1c — candidate fusion (reciprocal-rank fusion, RRF) of both systems.

Stage 2 — cross-encoder reranking (``BAAI/bge-reranker-base``) over the fused
pool, then source-aware diversity and a dynamic final top-k.

Safety:
- BROAD fallback: if routed retrieval returns too few candidates, a broad
  search over the full collection runs automatically and is merged, so routing
  can never silently destroy recall.
- Dynamic top-k: simple facts use fewer chunks; list/complex questions use more.
- Bounded caches: identical queries within the TTL skip embedding + retrieval.
"""

from __future__ import annotations

import re
import time

from ..config import get_config
from ..embeddings.embedder import Embedder
from ..routing.router import QueryRouter
from ..routing.schemas import RouteResult
from ..schemas.documents import RetrievedChunk
from ..utils.logging_utils import get_logger
from ..vectorstore.store import VectorStore
from .bm25 import BM25Index
from .intents import is_list_intent
from .query_normalizer import expand_query, is_arabic, normalize_query
from ..routing.rules import PRIORITY_TYPES

logger = get_logger(__name__)

# Score lift applied to primary-type chunks on a confident route.
_PRIORITY_BOOST = 0.25

# Intents that need only a few chunks (simple factual answers).
_SIMPLE_INTENTS = {"FACT", "LOCATION", "PERSON", "FAQ", "CONTACT", "TUITION"}
# Intents that need a fuller evidence set (lists / comparisons / policy).
_COMPLEX_INTENTS = {"COMPARISON", "ADMINISTRATION", "REGULATION", "NEWS", "FACILITY"}

# Intents where the authoritative source may be in a DIFFERENT language than
# the question (e.g. the Arabic about/contact page holds location info even
# when the user asks in English). For these, do NOT apply a language filter
# at the Chroma level — the re-ranker and LLM can handle cross-lingual evidence.
_CROSS_LINGUAL_INTENTS = {
    "LOCATION", "CONTACT", "PERSON", "ADMINISTRATION", "FAQ", "FACT", "FACILITY",
}


class Retriever:
    """Retrieve relevant chunks for a question with routing + reranking."""

    def __init__(
        self,
        vectorstore: VectorStore | None = None,
        embedder: Embedder | None = None,
        top_k: int | None = None,
        candidate_k: int | None = None,
        similarity_threshold: float | None = None,
        reranker=None,
        reranker_enabled: bool | None = None,
    ) -> None:
        cfg = get_config()
        self.vectorstore = vectorstore or VectorStore()
        self.embedder = embedder or Embedder()
        self.top_k = top_k if top_k is not None else cfg["top_k"]
        self.candidate_k = (
            candidate_k if candidate_k is not None else cfg["rerank_candidates"]
        )
        self.similarity_threshold = (
            similarity_threshold
            if similarity_threshold is not None
            else cfg["similarity_threshold"]
        )
        self.reranker = reranker
        self.reranker_enabled = (
            reranker_enabled
            if reranker_enabled is not None
            else (cfg["reranker_enabled"] and reranker is not None)
        )
        self.rerank_top_k = cfg["rerank_top_k"]
        self.max_chunks_per_source = cfg["max_chunks_per_source"]
        self.source_priority = cfg["source_priority"]
        self.list_source_types = set(cfg["list_source_types"])
        self.expansion_chunks_per_source = cfg["expansion_chunks_per_source"]
        self.list_seed_urls = list(cfg["list_seed_urls"])
        self.list_seed_enabled = cfg["list_seed_enabled"]
        # Hybrid dense + lexical (BM25) retrieval.
        self.hybrid_enabled = cfg["hybrid_enabled"]
        self.query_expansion_enabled = cfg["query_expansion_enabled"]
        self.max_query_expansion_terms = cfg["max_query_expansion_terms"]
        self.hybrid_fusion = cfg["hybrid_fusion"]
        self.dense_weight = cfg["dense_weight"]
        self.bm25_weight = cfg["bm25_weight"]
        # Routing / safe filtering.
        self.router_enabled = cfg["router_enabled"]
        self.confidence_threshold = cfg["router_confidence_threshold"]
        self.min_results = cfg["router_min_results"]
        self.fallback_enabled = cfg["router_fallback_enabled"]
        self.filter_language = cfg["router_filter_language"]
        self.filter_category = cfg["router_filter_category"]
        self.filter_faculty = cfg["router_filter_faculty"]
        self._router = QueryRouter() if self.router_enabled else None
        # Dynamic top-k.
        self.dynamic_top_k = cfg["dynamic_top_k_enabled"]
        self.top_k_fact = cfg["top_k_fact"]
        self.top_k_list = cfg["top_k_list"]
        self.top_k_complex = cfg["top_k_complex"]
        # Resource limits.
        self.max_retrieval_results = cfg["max_retrieval_results"]
        self.max_rerank_results = cfg["max_rerank_results"]
        self.cache_enabled = cfg["cache_enabled"]
        self._bm25: BM25Index | None = None
        self._chunk_map: dict[str, RetrievedChunk] | None = None
        # Populated by retrieve() for pipeline timing instrumentation.
        self.last_timings: dict[str, float] = {}
        self.last_meta: dict = {}
        self.last_trace: dict = {}

    # -- public API ----------------------------------------------------------

    def warmup(self) -> None:
        """Pre-load the lexical index once (startup) so the first query is fast.

        Best-effort: never raises. The BM25 index over all chunks is built on
        first retrieval anyway; calling this at server startup moves that cost
        out of the first /chat request and makes the [INDEX] lifecycle visible.
        """
        try:
            self._bm25_index()
        except Exception:  # pragma: no cover - warm-up must never break startup
            logger.exception("[INDEX] Warm-up failed; index will build lazily")

    def retrieve(
        self,
        question: str,
        intent: str | None = None,
        route: RouteResult | None = None,
        query_variants: list[str] | None = None,
        memory_seed_urls: list[str] | None = None,
        candidate_factor: float | None = None,
        force_refresh: bool = False,
    ) -> list[RetrievedChunk]:
        """Return filtered, reranked, diverse chunks for a question.

        ``query_variants``: bounded lexical variants for BM25 (query
        understanding / expansion); merges hits across variants.
        ``memory_seed_urls``: historical source URLs (retrieval memory) seeded
        as soft candidates that still go through reranking.
        ``candidate_factor``: extra candidate-pool reduction for multi-intent
        sub-retrievals (never below a safe floor).
        """
        question = question.strip()
        if not question:
            raise ValueError("Question must not be empty.")

        t_start = time.perf_counter()
        self.last_timings = {}
        self.last_meta = {}
        self.last_trace = {
            "query": question,
            "intent": intent,
            "query_variants": list(query_variants or []),
            "stages": {},
            "removed": [],
            "coverage": {},
        }

        # --- route (deterministic, no LLM) ---
        if self._router is not None and route is None:
            route = self._router.route(question)
        if route is not None and intent is None:
            intent = route.intent
        self.last_meta["route"] = (
            route.to_dict() if route is not None else None
        )

        # Adaptive candidate-pool size (narrow on confident routes / per
        # multi-intent sub-query, broad otherwise — recall floor enforced).
        candidate_k = self._candidate_k(route, candidate_factor)
        self.last_meta["candidate_k_used"] = candidate_k

        query_language = "ar" if is_arabic(question) else "en"
        specific_faculty_overview = bool(
            route is not None
            and route.faculty
            and (intent or "").upper() == "FACULTY"
        )
        list_mode = is_list_intent(intent or "") and not specific_faculty_overview
        logger.info("Retrieving (intent=%s route=%s) for: %s", intent,
                    self.last_meta["route"], question)

        # --- retrieval cache ---
        cache_key = None
        if self.cache_enabled and not force_refresh:
            from ..utils.cache import get_cache_registry

            registry = get_cache_registry()
            cache_key = self._cache_key(question, intent, route)
            cached = registry.retrieval.get(cache_key)
            if cached is not None:
                self.last_meta["cache_hit"] = True
                self.last_timings["retrieval_time"] = 0.0
                chunks = [RetrievedChunk(**c) for c in cached]
                self.last_meta["candidate_count"] = len(chunks)
                self.last_meta["final_count"] = len(chunks)
                self.last_meta["top_k_used"] = len(chunks)
                return chunks
        elif force_refresh:
            self.last_meta["cache_bypass_reason"] = "feedback_requires_fresh_retrieval"
        self.last_meta["cache_hit"] = False

        # --- build optional Chroma metadata filter ---
        where = self._build_where(route) if route is not None else None
        self.last_meta["where"] = where
        self.last_meta["routed"] = where is not None

        # --- stage 1a: dense candidates ---
        t0 = time.perf_counter()
        query_vec = self.embedder.embed_query(question)
        dense = self.vectorstore.query(query_vec, top_k=candidate_k, where=where)
        self._trace_stage("dense", dense)
        self.last_timings["embedding_time"] = round(time.perf_counter() - t0, 3)

        # --- stage 1b: lexical BM25 (same routed scope as dense; broad falls
        # back automatically when the routed pool is too thin) ---
        bm25_hits: list[tuple[str, float]] = []
        if self.hybrid_enabled:
            t0 = time.perf_counter()
            bm25_hits = self._bm25_search(
                question, intent, route, variants=query_variants, top_k=candidate_k
            )
            chunk_map = self._chunk_index()
            bm25_rows = []
            for rank, (cid, score) in enumerate(bm25_hits, start=1):
                chunk = chunk_map.get(cid)
                if chunk is None:
                    bm25_rows.append(
                        {"rank": rank, "chunk_id": cid, "bm25_score": round(score, 4)}
                    )
                else:
                    row = _trace_chunk(chunk, rank)
                    row["bm25_score"] = round(score, 4)
                    bm25_rows.append(row)
            self.last_trace["stages"]["bm25"] = bm25_rows
            self.last_timings["bm25_time"] = round(time.perf_counter() - t0, 3)

        t0 = time.perf_counter()
        results = self._fuse(dense, bm25_hits, route)
        self._trace_stage("fused", results)
        fallback_used = False

        # --- BROAD fallback: routed retrieval too thin? ---
        if (
            self.fallback_enabled
            and where is not None
            and len(results) < self.min_results
            and not (route is not None and route.faculty)
        ):
            logger.info("Routed retrieval too thin (%d); running BROAD fallback",
                        len(results))
            t0 = time.perf_counter()
            dense_broad = self.vectorstore.query(query_vec, top_k=candidate_k)
            broad = self._fuse(dense_broad, bm25_hits, route)
            merged: dict[str, RetrievedChunk] = {c.chunk_id: c for c in results}
            for c in broad:
                merged.setdefault(c.chunk_id, c)
            results = list(merged.values())
            self._trace_stage("broad_fallback_merged", results)
            self.last_timings["fallback_time"] = round(time.perf_counter() - t0, 3)
            fallback_used = True
        self.last_meta["fallback_used"] = fallback_used
        self.last_timings["fusion_time"] = round(time.perf_counter() - t0, 3)

        # Cap candidate pool before the expensive stages.
        results = results[: self.max_retrieval_results]
        self._trace_stage("candidate_cap", results)

        # Deduplicate near-identical content (mirrors, repeated sections).
        results = self._dedupe(results)
        self._trace_stage("deduped", results)

        # List expansion + canonical directory seeding for list intents.
        if list_mode:
            results = self._expand_for_lists(results)
            results = self._dedupe(results)
            if self.list_seed_enabled:
                results = self._seed_directory_pages(results)
            self._trace_stage("list_expanded", results)

        faculty_scope_mode = (
            route is not None
            and bool(route.faculty)
            and (intent or "").upper() in {"PROGRAM", "FACULTY"}
        )
        if faculty_scope_mode:
            results = self._expand_for_faculty_scope(results, route, intent or "", question)
            results = self._dedupe(results)
            self._trace_stage("faculty_scope_expanded", results)

        # Numeric / fee questions need high-recall evidence that contains
        # actual values. Generic faculty/about chunks may still help entity
        # grounding, but fee-bearing chunks must survive through reranking and
        # final context selection.
        fee_mode = (intent or "").upper() == "TUITION"
        if fee_mode:
            results = self._expand_for_fees(results, question)
            results = self._dedupe(results)
            self._trace_stage("fee_expanded", results)

        scholarship_mode = (intent or "").upper() == "SCHOLARSHIP"
        if scholarship_mode:
            results = self._expand_for_scholarships(results)
            results = self._dedupe(results)
            self._trace_stage("scholarship_expanded", results)

        # Historical retrieval-memory hints (soft; always re-ranked below).
        if memory_seed_urls:
            results = self._seed_memory_sources(results, memory_seed_urls)

        # Similarity filtering (dense cosine threshold; lexical-only matches
        # are retained because they matched query terms verbatim).
        filtered = []
        for r in results:
            if r.dense_score is None or r.dense_score >= self.similarity_threshold:
                filtered.append(r)
            else:
                self._trace_removed(
                    r,
                    "similarity_threshold",
                    f"dense_score={r.dense_score:.4f} < {self.similarity_threshold:.4f}",
                )
        if len(filtered) < len(results):
            logger.info(
                "Filtered %d of %d results below threshold %.3f",
                len(results) - len(filtered),
                len(results),
                self.similarity_threshold,
            )
        self.last_meta["candidate_count"] = len(filtered)
        self._trace_stage("threshold_filtered", filtered)
        evidence_pool = list(filtered)

        # Stage 2: rerank the candidate pool (bounded).
        if self.reranker_enabled and self.reranker is not None and filtered:
            t0 = time.perf_counter()
            pool = filtered[: self.max_rerank_results]
            for r in filtered[self.max_rerank_results:]:
                self._trace_removed(
                    r,
                    "max_rerank_results",
                    f"rank beyond max_rerank_results={self.max_rerank_results}",
                )
            filtered = self.reranker.rerank(question, pool)
            self.last_timings["reranking_time"] = round(time.perf_counter() - t0, 3)
        else:
            self.last_timings["reranking_time"] = 0.0
        self.last_meta["reranked_count"] = len(filtered)
        if faculty_scope_mode:
            scoped = [c for c in filtered if c.faculty == route.faculty]
            if scoped:
                # Keep a named faculty query inside its authoritative page.
                # Directory and memory-seed chunks may be useful for broad
                # lists, but must not replace the requested faculty profile.
                filtered = scoped
        if fee_mode and _is_broad_fee_question(question):
            filtered = self._preserve_full_fee_table(filtered, evidence_pool, query_language)
        self._trace_stage("reranked", filtered)

        # Source-aware diversity + source-priority preference.
        final = self._apply_source_diversity(
            filtered, list_mode, query_language,
            fee_mode=fee_mode, faculty_scope_mode=faculty_scope_mode,
            scholarship_mode=scholarship_mode,
        )
        final = self._ensure_required_evidence(
            filtered, final, intent or "FACT", question, query_language, route
        )
        self._trace_stage("source_diverse", final)

        # Dynamic final top-k.
        final_top_k = self._dynamic_top_k(intent or "FACT", question)
        for r in final[final_top_k:]:
            self._trace_removed(
                r,
                "final_top_k",
                f"rank beyond dynamic top_k={final_top_k}",
            )
        final = final[: final_top_k]
        self.last_meta["final_count"] = len(final)
        self.last_meta["top_k_used"] = final_top_k
        self._trace_stage("final", final)
        self.last_trace["coverage"] = self._coverage(question, intent, final)
        self.last_timings["retrieval_time"] = round(time.perf_counter() - t_start, 3)
        logger.info("Retrieval done: candidates=%d final=%d", len(filtered), len(final))

        # Store a safe snapshot in the retrieval cache.
        if cache_key is not None:
            from ..utils.cache import get_cache_registry

            get_cache_registry().retrieval.put(
                cache_key, [c.model_dump() for c in final]
            )
        return final

    def _preserve_full_fee_table(
        self,
        reranked: list[RetrievedChunk],
        evidence_pool: list[RetrievedChunk],
        query_language: str,
    ) -> list[RetrievedChunk]:
        """Keep complete tuition tables even if the reranker prefers snippets."""
        if any((c.content_type or "").lower() == "tuition" and _fee_row_count(c.text) >= 3 for c in reranked):
            return reranked
        candidates = [
            c for c in evidence_pool
            if (c.content_type or "").lower() == "tuition"
            and _fee_row_count(c.text) >= 3
        ]
        if not candidates:
            return reranked
        candidates.sort(
            key=lambda c: (
                (c.language or "").lower() == (query_language or "").lower(),
                _fee_row_count(c.text),
                c.score,
            ),
            reverse=True,
        )
        best = candidates[0]
        best.dense_score = None
        best.score = max(best.score, reranked[0].score if reranked else best.score)
        logger.info("Preserved complete tuition table chunk %s after reranking", best.chunk_id)
        return [best, *[c for c in reranked if c.chunk_id != best.chunk_id]]

    # -- routing helpers -------------------------------------------------------

    def _build_where(self, route: RouteResult) -> dict | None:
        """Build a Chroma ``where`` clause from a confident route (or None).

        Language filtering is intentionally skipped for cross-lingual intents
        (LOCATION, CONTACT, PERSON, etc.) because the authoritative source is
        often stored in only one language regardless of the question language.
        Applying a strict language filter for those intents causes the retriever
        to silently discard the very chunks that contain the correct answer.
        """
        if not self.router_enabled or route is None:
            return None
        if route.confidence < self.confidence_threshold:
            return None
        clauses: list[dict] = []
        # Skip language filtering for intents where the authoritative content
        # may live in a different language than the question (e.g. Arabic about
        # page answering an English location question, or vice versa).
        intent_upper = (route.intent or "").upper()
        if (
            self.filter_language
            and route.language in ("ar", "en")
            and intent_upper not in _CROSS_LINGUAL_INTENTS
        ):
            clauses.append({"language": {"$eq": route.language}})
        if self.filter_category and route.category_types:
            clauses.append({"content_type": {"$in": route.category_types}})
        if self.filter_faculty and route.faculty:
            clauses.append({"faculty": {"$eq": route.faculty}})
        if not clauses:
            return None
        return clauses[0] if len(clauses) == 1 else {"$and": clauses}

    def _cache_key(self, question: str, intent: str | None, route: RouteResult | None) -> str:
        from ..utils.cache import retrieval_signature

        route_part = route.to_dict() if route is not None else None
        return f"{retrieval_signature()}|q={question}|i={intent}|r={route_part}"

    def _dynamic_top_k(self, intent: str, question: str = "") -> int:
        """Final top-k, varying with query complexity (never exceeds top_k)."""
        if not self.dynamic_top_k:
            return self.top_k
        q = (question or "").lower()
        broad_fee = intent == "TUITION" and any(
            marker in q
            for marker in (
                "all", "faculties", "faculty fees", "annual", "سنوي",
                "السنوية", "الكليات", "كل الكليات", "جميع",
            )
        )
        if broad_fee:
            base = self.top_k_list
        elif intent in _SIMPLE_INTENTS:
            base = self.top_k_fact
        elif intent in _COMPLEX_INTENTS:
            base = self.top_k_complex
        else:
            base = self.top_k_list
        return max(2, min(base, self.top_k))

    def _candidate_k(self, route: RouteResult | None, candidate_factor: float | None) -> int:
        """Adaptive candidate-pool size (never below a recall floor).

        High-confidence routed queries may search a narrower window
        (``ADAPTIVE_NARROW_FACTOR``); multi-intent sub-retrievals pass a
        smaller ``candidate_factor`` (``ADAPTIVE_MULTI_FACTOR``). Always at
        least ``ADAPTIVE_MIN_CANDIDATES`` and never more than the configured
        candidate pool.
        """
        if not get_config().get("adaptive_retrieval_enabled", True):
            return self.candidate_k
        base = self.candidate_k
        factor = 1.0
        if route is not None and route.confidence >= self.confidence_threshold:
            factor *= float(get_config().get("adaptive_narrow_factor", 0.8))
        if candidate_factor is not None:
            factor *= candidate_factor
        k = int(base * factor)
        floor = int(get_config().get("adaptive_min_candidates", 10) or 10)
        return max(floor, min(k, base))

    # -- hybrid lexical stage ------------------------------------------------

    def _chunk_index(self) -> dict[str, RetrievedChunk]:
        """Lazily load every chunk once (for BM25 construction / lookup)."""
        if self._chunk_map is None:
            self._chunk_map = {c.chunk_id: c for c in self.vectorstore.get_all()}
            logger.info(
                "[INDEX] Loaded %d chunks for the lexical index (once)",
                len(self._chunk_map),
            )
        return self._chunk_map

    def _bm25_index(self) -> BM25Index:
        if self._bm25 is None:
            self._bm25 = BM25Index(
                {cid: c.text for cid, c in self._chunk_index().items()}
            )
            logger.info("[INDEX] Built BM25 index over %d chunks", len(self._chunk_map or {}))
        return self._bm25

    def _bm25_search(
        self,
        question: str,
        intent: str | None,
        route: RouteResult | None = None,
        variants: list[str] | None = None,
        top_k: int | None = None,
    ) -> list[tuple[str, float]]:
        """BM25 candidates over the normalized (+expanded) query text.

        ``variants``: a bounded set of lexical query strings produced by query
        understanding/expansion. When provided, each variant is searched and
        the hits merged (per chunk the best score wins). Otherwise the legacy
        single query (normalized + intent expansion) is used.

        For a confident routed query, BM25 is restricted to the same metadata
        scope as dense retrieval so broad news/home pages never crowd out the
        routed content type. A too-thin pool triggers the BROAD fallback, so
        recall is never silently lost.
        """
        top_k = top_k if top_k is not None else self.candidate_k
        index = self._bm25_index()
        allowed = self._bm25_allowed_ids(route)
        if variants:
            merged: dict[str, float] = {}
            for variant in variants[: self.max_query_expansion_terms + 1]:
                for cid, score in index.search(variant, top_k=top_k, allowed=allowed):
                    if score > merged.get(cid, -1.0):
                        merged[cid] = score
            ranked = sorted(merged.items(), key=lambda kv: kv[1], reverse=True)
            return ranked[:top_k]
        lex_query = normalize_query(question)
        if self.query_expansion_enabled and intent:
            lex_query = expand_query(
                lex_query, intent, max_terms=self.max_query_expansion_terms
            )
        return index.search(lex_query, top_k=top_k, allowed=allowed)

    def _bm25_allowed_ids(self, route: RouteResult | None) -> set[str] | None:
        """Chunk ids allowed by the routed metadata scope (None = broad).

        Language filtering is skipped for cross-lingual intents (same logic as
        _build_where): the authoritative Arabic about/contact page must not be
        excluded from BM25 scope when the user asks in English (or vice versa).
        """
        if not self.router_enabled or route is None:
            return None
        if route.confidence < self.confidence_threshold:
            return None
        allowed: set[str] | None = None
        intent_upper = (route.intent or "").upper()
        if (
            self.filter_language
            and route.language in ("ar", "en")
            and intent_upper not in _CROSS_LINGUAL_INTENTS
        ):
            allowed = {
                cid for cid, c in self._chunk_index().items()
                if c.language == route.language
            }
        if self.filter_category and route.category_types:
            wanted = {t.lower() for t in route.category_types}
            ids = {
                cid for cid, c in self._chunk_index().items()
                if (c.content_type or "").lower() in wanted
            }
            allowed = ids if allowed is None else (allowed & ids)
        if self.filter_faculty and route.faculty:
            ids = {
                cid for cid, c in self._chunk_index().items()
                if (c.faculty or "") == route.faculty
            }
            allowed = ids if allowed is None else (allowed & ids)
        return allowed

    def _fuse(
        self,
        dense: list[RetrievedChunk],
        bm25_hits: list[tuple[str, float]],
        route: RouteResult | None = None,
    ) -> list[RetrievedChunk]:
        """Merge dense + BM25 candidate lists using RRF (or weighted linear).

        On a confident route, chunks whose content type is primary for the
        routed intent receive a small score lift so the on-topic page (e.g. the
        actual admission page) is not crowded out by generic news/home pages.
        """
        if not bm25_hits:
            return dense
        boost_types = None
        if route is not None and route.confidence >= self.confidence_threshold:
            boost_types = PRIORITY_TYPES.get(route.intent)
        chunk_map = self._chunk_index()
        dense_ranks = {c.chunk_id: i for i, c in enumerate(dense)}
        bm25_ranks = {cid: i for i, (cid, _) in enumerate(bm25_hits)}
        bm25_scores = dict(bm25_hits)
        index = self._bm25_index()

        by_id: dict[str, RetrievedChunk] = {}
        for c in dense:
            by_id[c.chunk_id] = c
        for cid, _ in bm25_hits:
            if cid not in by_id and cid in chunk_map:
                by_id[cid] = chunk_map[cid]

        fused = []
        for cid, chunk in by_id.items():
            d_rank = dense_ranks.get(cid)
            b_rank = bm25_ranks.get(cid)
            if d_rank is None and b_rank is None:
                continue
            # Keep the original dense cosine score (None for lexical-only hits
            # so the similarity threshold still applies only to dense results).
            if d_rank is None:
                chunk.dense_score = None
            chunk.bm25_score = round(bm25_scores.get(cid, 0.0), 4)
            if self.hybrid_fusion == "linear" and d_rank is not None and b_rank is not None:
                norm_d = 1.0 - (d_rank / max(1, len(dense)))
                norm_b = 1.0 - (b_rank / max(1, len(bm25_hits)))
                chunk.score = round(
                    self.dense_weight * norm_d + self.bm25_weight * norm_b, 6
                )
            else:
                chunk.score = index.rrf_score(
                    d_rank if d_rank is not None else 10 ** 6,
                    b_rank,
                )
            if boost_types and (chunk.content_type or "").lower() in boost_types:
                chunk.score = round(chunk.score + _PRIORITY_BOOST, 6)
            fused.append(chunk)

        fused.sort(key=lambda c: c.score, reverse=True)
        return fused[: self.candidate_k]

    # -- list expansion -------------------------------------------------------

    def _expand_for_lists(self, results: list[RetrievedChunk]) -> list[RetrievedChunk]:
        """Fetch complete sections of the strongest list-bearing sources."""
        by_source: dict[str, list[RetrievedChunk]] = {}
        for chunk in results:
            url = chunk.source_url or ""
            if url not in by_source:
                by_source[url] = []
            by_source[url].append(chunk)

        extra: list[RetrievedChunk] = []
        for url, chunks in by_source.items():
            top = max(chunks, key=lambda c: c.score)
            content_type = (top.content_type or "").lower()
            if content_type not in self.list_source_types:
                continue
            doc_id = top.document_id
            section_id = top.section_id
            fetched = []
            if section_id:
                fetched = self.vectorstore.get_by_section(doc_id, section_id)
            if not fetched:
                fetched = self.vectorstore.get_by_document(doc_id)
            # Keep only chunks from the same source page (mirrors differ by URL).
            fetched = [c for c in fetched if (c.source_url or "") == url]
            fetched = fetched[: self.expansion_chunks_per_source]
            for c in fetched:
                # These chunks are authoritative section/document expansion,
                # not dense hits. Mark dense_score as None so the dense
                # similarity threshold does not erase the very evidence that
                # list expansion recovered for completeness.
                c.dense_score = None
                c.score = max(c.score, top.score)
            extra.extend(fetched)
            logger.info(
                "List expansion: +%d chunks from %s (%s)", len(fetched), content_type, url[:60]
            )

        merged = {c.chunk_id: c for c in results}
        for c in extra:
            merged.setdefault(c.chunk_id, c)
        return list(merged.values())

    def _expand_for_fees(
        self, results: list[RetrievedChunk], question: str
    ) -> list[RetrievedChunk]:
        """Preserve fee-bearing evidence for tuition/numeric questions.

        The first-stage candidate pool can include a correct tuition page plus
        generic faculty/about pages. This expansion hydrates the strongest
        tuition/program/admission sources already found, then marks fetched
        chunks as lexical/authority expansion so dense-threshold filtering
        cannot remove chunks that contain exact numbers.
        """
        fee_sources = []
        for chunk in results:
            ctype = (chunk.content_type or "").lower()
            text = chunk.text or ""
            if ctype in {"tuition", "program", "admission", "faq"} or _has_fee_signal(text):
                fee_sources.append(chunk)
        if not fee_sources:
            return results

        extra: list[RetrievedChunk] = []
        seen_sources: set[str] = set()
        broad_fee = _is_broad_fee_question(question)
        for top in sorted(fee_sources, key=lambda c: c.score, reverse=True)[:4]:
            url = top.source_url or ""
            if not url or url in seen_sources:
                continue
            seen_sources.add(url)
            if broad_fee and (top.content_type or "").lower() == "tuition":
                for full in self.vectorstore.get_source_document_by_url(url):
                    full.dense_score = None
                    full.score = max(full.score, top.score + 0.01)
                    extra.append(full)
            fetched = self.vectorstore.get_by_url(url)
            if not fetched:
                fetched = self.vectorstore.get_by_document(top.document_id)
            for c in fetched[: self.expansion_chunks_per_source]:
                if not _has_fee_signal(c.text) and (c.content_type or "").lower() not in {
                    "tuition", "program", "admission",
                }:
                    continue
                c.dense_score = None
                c.score = max(c.score, top.score)
                extra.append(c)
        merged = {c.chunk_id: c for c in results}
        for c in extra:
            merged.setdefault(c.chunk_id, c)
        if extra:
            logger.info("Fee expansion: +%d fee-bearing chunks", len(extra))
        return list(merged.values())

    def _expand_for_scholarships(
        self, results: list[RetrievedChunk]
    ) -> list[RetrievedChunk]:
        """Hydrate scholarship source pages for complete rules/types evidence."""
        scholarship_sources = [
            c for c in results
            if (c.content_type or "").lower() == "scholarship"
            or _has_scholarship_signal(c.text)
        ]
        if not scholarship_sources:
            return results

        extra: list[RetrievedChunk] = []
        seen_sources: set[str] = set()
        for top in sorted(scholarship_sources, key=lambda c: c.score, reverse=True)[:3]:
            url = top.source_url or ""
            if not url or url in seen_sources:
                continue
            seen_sources.add(url)
            fetched = self.vectorstore.get_by_url(url)
            if not fetched:
                fetched = self.vectorstore.get_by_document(top.document_id)
            fetched.sort(
                key=lambda c: (
                    (c.content_type or "").lower() == "scholarship",
                    _has_scholarship_signal(c.text),
                    c.chunk_index or 0,
                ),
                reverse=True,
            )
            for c in fetched[: self.expansion_chunks_per_source]:
                if (
                    (c.content_type or "").lower() != "scholarship"
                    and not _has_scholarship_signal(c.text)
                ):
                    continue
                c.dense_score = None
                c.score = max(c.score, top.score + 0.01)
                extra.append(c)

        merged = {c.chunk_id: c for c in results}
        for c in extra:
            merged.setdefault(c.chunk_id, c)
        if extra:
            logger.info("Scholarship expansion: +%d scholarship chunks", len(extra))
        return list(merged.values())

    def _expand_for_faculty_scope(
        self,
        results: list[RetrievedChunk],
        route: RouteResult,
        intent: str,
        question: str,
    ) -> list[RetrievedChunk]:
        """Hydrate pages for a named faculty without hardcoding its answer.

        Faculty-specific program/department questions often retrieve one late
        chunk from the correct faculty page plus broad directory/about chunks.
        This expansion keeps all evidence from the named faculty's own pages
        in the candidate pool so reranking and coverage checks can select the
        precise department/program passage.
        """
        faculty = (route.faculty or "").strip()
        if not faculty:
            return results

        candidate_sources: dict[str, RetrievedChunk] = {}
        for chunk in results:
            if (chunk.faculty or "") == faculty:
                candidate_sources.setdefault(chunk.source_url or chunk.chunk_id, chunk)

        # If the first-stage pool missed the faculty page, use metadata from
        # the local chunk index to seed it. This reads existing Chroma-backed
        # chunks only; it does not fabricate or scrape fresh data.
        if not candidate_sources:
            wanted_types = {"faculty", "program"}
            indexed = [
                c for c in self._chunk_index().values()
                if (c.faculty or "") == faculty
                and (c.content_type or "").lower() in wanted_types
            ]
            indexed.sort(
                key=lambda c: (
                    (c.language or "").lower() == (route.language or "").lower(),
                    _has_program_signal(c.text),
                    c.score,
                ),
                reverse=True,
            )
            for chunk in indexed[:3]:
                candidate_sources.setdefault(chunk.source_url or chunk.chunk_id, chunk)

        extra: list[RetrievedChunk] = []
        intent_upper = (intent or "").upper()
        for top in sorted(candidate_sources.values(), key=lambda c: c.score, reverse=True)[:3]:
            url = top.source_url or ""
            fetched = self.vectorstore.get_by_url(url) if url else []
            if not fetched:
                fetched = self.vectorstore.get_by_document(top.document_id)
            fetched = [
                c for c in fetched
                if (c.faculty or "") == faculty
                and (c.content_type or "").lower() in {"faculty", "program"}
            ]
            fetched.sort(
                key=lambda c: (
                    (c.language or "").lower() == (route.language or "").lower(),
                    _asks_departments(question) and _has_department_signal(c.text),
                    _has_program_signal(c.text),
                    c.chunk_index or 0,
                ),
                reverse=True,
            )
            for c in fetched[: self.expansion_chunks_per_source]:
                if intent_upper == "PROGRAM" and not _has_program_signal(c.text):
                    continue
                c.dense_score = None
                c.score = max(c.score, top.score + 0.02)
                extra.append(c)

        merged = {c.chunk_id: c for c in results}
        for c in extra:
            merged.setdefault(c.chunk_id, c)
        if extra:
            logger.info(
                "Faculty-scope expansion: +%d chunks for %s", len(extra), faculty
            )
        return list(merged.values())

    def _seed_directory_pages(self, results: list[RetrievedChunk]) -> list[RetrievedChunk]:
        """Merge canonical directory pages (complete lists) into the pool."""
        merged = {c.chunk_id: c for c in results}
        for url in self.list_seed_urls:
            fetched = self.vectorstore.get_by_url(url)
            for chunk in fetched:
                if chunk.chunk_id in merged:
                    continue
                chunk.score = 1.0
                chunk.dense_score = None
                merged[chunk.chunk_id] = chunk
                logger.info("Seeded directory page chunk %s (%s)", chunk.chunk_id, url)
        return list(merged.values())

    def _seed_memory_sources(
        self, results: list[RetrievedChunk], urls: list[str]
    ) -> list[RetrievedChunk]:
        """Merge historical retrieval-memory sources as SOFT candidates.

        The seeded chunks still pass through similarity filtering, reranking,
        source diversity and the final dynamic top-k, so a stale hint can
        never crowd out a genuinely relevant chunk. Full fallback retrieval is
        always preserved.
        """
        if not urls:
            return results
        merged = {c.chunk_id: c for c in results}
        added = 0
        for url in urls[: self.max_chunks_per_source]:
            for chunk in self.vectorstore.get_by_url(url):
                if chunk.chunk_id in merged:
                    continue
                chunk.score = 1.0
                chunk.dense_score = None
                merged[chunk.chunk_id] = chunk
                added += 1
        if added:
            logger.info("Seeded %d historical retrieval-memory chunks", added)
        return list(merged.values())

    # -- source diversity -----------------------------------------------------

    def _apply_source_diversity(
        self,
        chunks: list[RetrievedChunk],
        list_mode: bool,
        query_language: str = "en",
        *,
        fee_mode: bool = False,
        faculty_scope_mode: bool = False,
        scholarship_mode: bool = False,
    ) -> list[RetrievedChunk]:
        """Cap chunks per source URL and apply a light type-priority boost."""
        if not chunks:
            return chunks

        # Apply source-priority boost for list questions (re-ranking order).
        if list_mode or fee_mode or faculty_scope_mode:
            boosted = []
            for c in chunks:
                priority = self.source_priority.get((c.content_type or "").lower(), 1.0)
                ctype = (c.content_type or "").lower()
                if fee_mode and ctype == "tuition":
                    lang_boost = (
                        1.03
                        if (c.language or "").lower() == (query_language or "").lower()
                        else 1.0
                    )
                    boosted.append((c, c.score * max(priority, 1.05) * lang_boost))
                elif faculty_scope_mode and ctype in {"faculty", "program"} and c.faculty:
                    lang_boost = (
                        1.03
                        if (c.language or "").lower() == (query_language or "").lower()
                        else 1.0
                    )
                    signal_boost = 1.05 if _has_program_signal(c.text) else 1.0
                    if _has_department_signal(c.text):
                        signal_boost *= 1.04
                    boosted.append((c, c.score * max(priority, 1.05) * lang_boost * signal_boost))
                elif list_mode and ctype in self.list_source_types:
                    boosted.append((c, c.score * priority))
                else:
                    boosted.append((c, c.score))
            boosted.sort(key=lambda item: item[1], reverse=True)
            chunks = [c for c, _ in boosted]

        if self.max_chunks_per_source <= 0:
            return chunks

        per_source: dict[str, int] = {}
        diverse: list[RetrievedChunk] = []
        for c in chunks:
            url = c.source_url or "no-url"
            cap = self.max_chunks_per_source
            if fee_mode and (c.content_type or "").lower() == "tuition":
                cap = max(cap, self.expansion_chunks_per_source + 1)
            if faculty_scope_mode and (c.content_type or "").lower() in {"faculty", "program"} and c.faculty:
                cap = max(cap, self.expansion_chunks_per_source + 1)
            if scholarship_mode and (
                (c.content_type or "").lower() == "scholarship"
                or _has_scholarship_signal(c.text)
            ):
                cap = max(cap, self.expansion_chunks_per_source)
            if per_source.get(url, 0) >= cap:
                continue
            per_source[url] = per_source.get(url, 0) + 1
            diverse.append(c)

        if list_mode:
            diverse = self._guarantee_directory_coverage(
                chunks, diverse, query_language
            )
        return diverse

    def _guarantee_directory_coverage(
        self,
        ranked: list[RetrievedChunk],
        diverse: list[RetrievedChunk],
        query_language: str = "en",
    ) -> list[RetrievedChunk]:
        """Ensure a complete-list source survives into the final set."""
        candidates = [
            c
            for c in ranked
            if "all-faculties-programs" in (c.source_url or "")
        ]
        if not candidates:
            candidates = [
                c
                for c in ranked
                if (c.content_type or "").lower() in self.list_source_types
            ]
        if not candidates:
            return diverse
        preferred = [
            c for c in candidates if (c.language or "").lower() == query_language
        ]
        best = (preferred or candidates)[0]

        # Directory/list evidence is coverage-critical. If it is already
        # present but not first (often after reranking), move it to the front
        # instead of letting generic biography/about chunks consume the early
        # context budget.
        diverse = [c for c in diverse if c.chunk_id != best.chunk_id]
        diverse.insert(0, best)
        return diverse

    def _ensure_required_evidence(
        self,
        ranked: list[RetrievedChunk],
        selected: list[RetrievedChunk],
        intent: str,
        question: str,
        query_language: str = "en",
        route: RouteResult | None = None,
    ) -> list[RetrievedChunk]:
        """Keep coverage-critical chunks before the final top-k cut.

        This does not broaden thresholds or disable reranking. It only checks
        whether an already-retrieved/reranked chunk contains a required fact
        that the current selected set lacks, then inserts the best such chunk
        so final_top_k cannot clip it away.
        """
        intent = (intent or "").upper()
        if not ranked:
            return selected
        chosen = {c.chunk_id for c in selected}
        selected_text = "\n".join(c.text or "" for c in selected)

        def insert_best(candidates: list[RetrievedChunk], reason: str) -> None:
            for cand in candidates:
                if cand.chunk_id in chosen:
                    return
                selected.insert(0, cand)
                chosen.add(cand.chunk_id)
                logger.info(
                    "Coverage preservation inserted %s chunk %s",
                    reason, cand.chunk_id,
                )
                return

        if intent == "LOCATION":
            candidates = [
                c for c in ranked
                if _contains_any(
                    c.text,
                    ("road", "address", "coastal", "الطريق", "العنوان", "بجوار"),
                )
                and _contains_any(
                    c.text,
                    ("dakahlia", "governorate", "الدقهلية", "محافظة"),
                )
                and (c.content_type or "").lower()
                in {"about", "contact", "home", "facility", "faculty"}
            ]
            candidates.sort(
                key=lambda c: (
                    (c.language or "").lower() == (query_language or "").lower(),
                    (c.content_type or "").lower() in {"about", "contact", "home"},
                    c.score,
                ),
                reverse=True,
            )
            selected_rich = next(
                (c for c in candidates if c.chunk_id in chosen),
                None,
            )
            if selected_rich is not None:
                selected[:] = [c for c in selected if c.chunk_id != selected_rich.chunk_id]
                selected.insert(0, selected_rich)
            else:
                insert_best(candidates, "location-address")
        elif intent == "TUITION":
            if _is_broad_fee_question(question):
                full_table_candidates = [
                    c for c in ranked
                    if (c.content_type or "").lower() == "tuition"
                    and _fee_row_count(c.text) >= 3
                ]
                full_table_candidates.sort(
                    key=lambda c: (
                        (c.language or "").lower() == (query_language or "").lower(),
                        _fee_row_count(c.text),
                        c.score,
                    ),
                    reverse=True,
                )
                selected_full = next(
                    (c for c in full_table_candidates if c.chunk_id in chosen),
                    None,
                )
                if selected_full is not None:
                    selected[:] = [c for c in selected if c.chunk_id != selected_full.chunk_id]
                    selected.insert(0, selected_full)
                else:
                    insert_best(full_table_candidates, "tuition-full-table")
            if not _has_fee_signal("\n".join(c.text or "" for c in selected)):
                candidates = [
                    c for c in ranked
                    if _has_fee_signal(c.text)
                    and (c.content_type or "").lower()
                    in {"tuition", "program", "admission", "faq", "faculty"}
                ]
                insert_best(candidates, "tuition-fee")
        elif intent == "PROGRAM" and route is not None and route.faculty:
            faculty = route.faculty
            need_department = _asks_departments(question)
            candidates = [
                c for c in ranked
                if (c.faculty or "") == faculty
                and (
                    _has_department_signal(c.text)
                    if need_department
                    else _has_program_signal(c.text)
                )
                and (c.content_type or "").lower() in {"faculty", "program"}
            ]
            candidates.sort(
                key=lambda c: (
                    (c.language or "").lower() == (query_language or "").lower(),
                    _has_department_signal(c.text),
                    c.score,
                ),
                reverse=True,
            )
            selected_program_ids = {c.chunk_id for c in selected}
            selected_program = next(
                (c for c in candidates if c.chunk_id in selected_program_ids),
                None,
            )
            if selected_program is not None:
                selected[:] = [c for c in selected if c.chunk_id != selected_program.chunk_id]
                selected.insert(0, selected_program)
            else:
                insert_best(candidates, "faculty-program")
        elif is_list_intent(intent) and not (
            intent == "FACULTY" and route is not None and route.faculty
        ):
            if not any("all-faculties-programs" in (c.source_url or "") for c in selected):
                candidates = [
                    c for c in ranked
                    if "all-faculties-programs" in (c.source_url or "")
                ]
                insert_best(candidates, "directory-list")
        return selected

    # -- debug trace / coverage ---------------------------------------------

    def _trace_stage(self, name: str, chunks: list[RetrievedChunk]) -> None:
        self.last_trace.setdefault("stages", {})[name] = [
            _trace_chunk(c, i) for i, c in enumerate(chunks, start=1)
        ]

    def _trace_removed(self, chunk: RetrievedChunk, reason: str, detail: str) -> None:
        self.last_trace.setdefault("removed", []).append(
            {**_trace_chunk(chunk, 0), "reason": reason, "detail": detail}
        )

    def _coverage(
        self, question: str, intent: str | None, chunks: list[RetrievedChunk]
    ) -> dict:
        intent = (intent or "").upper()
        text = "\n".join(c.text or "" for c in chunks)
        route = self.last_meta.get("route") or {}
        if intent == "LOCATION":
            checks = {
                "city": _contains_any(text, ("new mansoura city", "مدينة المنصورة الجديدة")),
                "governorate": _contains_any(text, ("dakahlia", "الدقهلية")),
                "road_or_address": _contains_any(
                    text, ("road", "address", "coastal", "الطريق", "العنوان", "بجوار")
                ),
            }
        elif intent == "TUITION":
            checks = {
                "fee_signal": _has_fee_signal(text),
                "currency_or_number": bool(re.search(r"\d|egp|جنيه|دولار", text, re.I)),
                "fee_source": any(
                    (c.content_type or "").lower() in {"tuition", "program", "admission"}
                    for c in chunks
                ),
            }
        elif intent == "SCHOLARSHIP":
            checks = {
                "scholarship_signal": _has_scholarship_signal(text),
                "scholarship_source": any(
                    (c.content_type or "").lower() == "scholarship"
                    for c in chunks
                ),
                "rules_or_types": _contains_any(
                    text,
                    (
                        "full scholarship", "half scholarship", "quarter scholarship",
                        "financial aid", "rules", "منحة كاملة", "نصف منحة",
                        "ربع منحة", "الدعم الاجتماعي", "قواعد المنح",
                    ),
                ),
            }
        elif intent == "PROGRAM" and route.get("faculty"):
            faculty = route.get("faculty")
            checks = {
                "faculty_source": any((c.faculty or "") == faculty for c in chunks),
                "program_or_department_signal": any(
                    (c.faculty or "") == faculty and _has_program_signal(c.text)
                    for c in chunks
                ),
            }
            if _asks_departments(question):
                checks["department_signal"] = any(
                    (c.faculty or "") == faculty and _has_department_signal(c.text)
                    for c in chunks
                )
        elif is_list_intent(intent) and not (
            intent == "FACULTY" and route.get("faculty")
        ):
            checks = {
                "directory_source": any(
                    "all-faculties-programs" in (c.source_url or "")
                    for c in chunks
                ),
                "list_like_content": len(re.split(r"\n|اعرف المزيد|\|", text)) >= 4,
            }
        else:
            checks = {"non_empty_evidence": bool(chunks)}
        missing = [k for k, ok in checks.items() if not ok]
        return {"checks": checks, "ok": not missing, "missing": missing}

    # -- deduplication ---------------------------------------------------------

    @staticmethod
    def _dedupe(results: list[RetrievedChunk]) -> list[RetrievedChunk]:
        """Drop near-duplicate chunks sharing a content hash or text."""
        seen: dict[str, RetrievedChunk] = {}

        def signature(r: RetrievedChunk) -> str:
            norm = re.sub(r"[\s\u00a0]+", " ", r.text).strip()
            return f"text:{norm[:200]}"

        for r in results:
            key = signature(r)
            if key not in seen or r.score > seen[key].score:
                seen[key] = r
        return list(seen.values())


_FEE_SIGNAL_RE = re.compile(
    r"(\bfees?\b|\btuition\b|\bcost\b|\bprice\b|\begp\b|\$\s*\d|"
    r"\d[\d,.\s]*(?:egp|جنيه|دولار)|رسوم|مصروفات|مصاريف|تكلفة|تكاليف)",
    re.IGNORECASE,
)

_SCHOLARSHIP_SIGNAL_RE = re.compile(
    r"(scholarships?|financial aid|grant|grants|full scholarship|half scholarship|"
    r"quarter scholarship|منح|منحة|المنح|الدعم الاجتماعي|منحة كاملة|نصف منحة|"
    r"ربع منحة|قواعد المنح)",
    re.IGNORECASE,
)


def _has_fee_signal(text: str) -> bool:
    return bool(_FEE_SIGNAL_RE.search(text or ""))


def _has_scholarship_signal(text: str) -> bool:
    return bool(_SCHOLARSHIP_SIGNAL_RE.search(text or ""))


def _fee_row_count(text: str) -> int:
    lines = [p.strip(" \t:-") for p in re.split(r"[\n|]+", text or "") if p.strip()]
    count = 0
    for i in range(len(lines) - 1):
        name, value = lines[i], lines[i + 1]
        if re.fullmatch(r"\d[\d,.\s]*", value.replace(" ", "")) and not re.search(r"\d", name):
            if len(name) <= 80 and name.lower() not in {"faculty", "college", "الكلية"}:
                count += 1
    return count


_PROGRAM_SIGNAL_RE = re.compile(
    r"(programs?|departments?|majors?|courses?|academic\s+program|"
    r"برامج|برنامج|اقسام|أقسام|قسم|تخصص|الخطة الدراسية)",
    re.IGNORECASE,
)


def _has_program_signal(text: str) -> bool:
    return bool(_PROGRAM_SIGNAL_RE.search(text or ""))


_DEPARTMENT_SIGNAL_RE = re.compile(
    r"(departments?|department\s+of|اقسام|أقسام|قسم)",
    re.IGNORECASE,
)


def _has_department_signal(text: str) -> bool:
    return bool(_DEPARTMENT_SIGNAL_RE.search(text or ""))


def _asks_departments(question: str) -> bool:
    return _contains_any(
        question,
        ("department", "departments", "department of", "قسم", "أقسام", "اقسام"),
    )


def _is_broad_fee_question(question: str) -> bool:
    q = (question or "").lower()
    return any(
        marker in q
        for marker in (
            "all", "faculties", "faculty fees", "annual", "الكليات",
            "كل الكليات", "جميع", "سنوي", "السنوية",
        )
    )


def _contains_any(text: str, markers: tuple[str, ...]) -> bool:
    low = (text or "").lower()
    return any(m.lower() in low for m in markers)


def _trace_chunk(c: RetrievedChunk, rank: int) -> dict:
    return {
        "rank": rank,
        "chunk_id": c.chunk_id,
        "score": c.score,
        "dense_score": c.dense_score,
        "bm25_score": c.bm25_score,
        "rerank_score": c.rerank_score,
        "title": c.title,
        "url": c.source_url,
        "type": c.content_type,
        "language": c.language,
        "faculty": c.faculty,
        "text_preview": re.sub(r"\s+", " ", (c.text or ""))[:220],
    }

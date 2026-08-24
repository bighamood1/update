"""RAG pipeline: retrieve -> filter -> rerank -> context -> Ollama -> answer.

Public API::

    rag = RAGPipeline()
    result = rag.ask("ما هي كليات جامعة المنصورة الجديدة؟")
    # result == {"answer": "...", "sources": [...], "retrieved_chunks": [...],
    #            "intent": "...", "timings": {...}}
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass, field
from typing import Any

from ..cache.semantic_cache import SemanticCache
from ..cache.store import get_runtime_store, new_id
from ..config import get_config
from ..context.builder import ContextBuilder
from ..context.compressor import ContextCompressor
from ..embeddings.embedder import Embedder
from ..feedback.analytics import initial_quality_score
from ..generation.fast_path import person_evidence_answer, try_fast_answer
from ..generation.ollama_client import OllamaClient
from ..generation.prompts import SYSTEM_PROMPT, build_rag_prompt, build_repair_prompt
from ..generation.response_formatter import format_final_answer
from ..generation.validation import (
    REFUSAL_AR,
    REFUSAL_EN,
    answer_relevance_ok,
    completeness_issues,
    refusal_text,
    strip_reasoning_artifacts,
    validate_answer,
)
from ..query.conversation import resolve_conversation
from ..query.expansion import retrieval_variants
from ..query.multi_intent import split_question
from ..query.understanding import QueryUnderstanding, understand
from ..quality.validator import evaluate_answer
from ..retrieval.intents import is_list_intent
from ..retrieval.reranker import Reranker
from ..retrieval.retrieval_memory import RetrievalMemory
from ..retrieval.retriever import Retriever
from ..routing.router import QueryRouter
from ..schemas.documents import RetrievedChunk
from ..utils.logging_utils import get_logger
from ..vectorstore.store import VectorStore

logger = get_logger(__name__)

# Intents whose answer is (usually) a complete list / directory extract. These
# get a larger context char + token budget so the FULL list reaches the LLM
# instead of being truncated at the generic 4000-char cap.
_LIST_LIKE_INTENTS = {
    "LIST", "FACULTY", "PROGRAM", "COMPARISON", "REGULATION",
    "ADMINISTRATION", "SCHOLARSHIP", "TRANSFER",
}


@dataclass
class RAGResult:
    """Structured result of a single RAG question."""

    answer: str
    sources: list[dict[str, Any]] = field(default_factory=list)
    retrieved_chunks: list[RetrievedChunk] = field(default_factory=list)
    intent: str | None = None
    timings: dict[str, float] = field(default_factory=dict)
    route: dict[str, Any] | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)
    question_id: str = ""
    cache_hit: bool = False
    cached_question_id: str | None = None
    llm_used: bool = False

    def to_dict(self) -> dict:
        return {
            "answer": self.answer,
            "sources": self.sources,
            "intent": self.intent,
            "timings": self.timings,
            "route": self.route,
            "diagnostics": self.diagnostics,
            "retrieved_chunks": [c.model_dump() for c in self.retrieved_chunks],
            "question_id": self.question_id,
            "cache_hit": self.cache_hit,
            "cached_question_id": self.cached_question_id,
            "llm_used": self.llm_used,
        }


# ContextBuilder is provided by rag.context.builder (imported above) and
# re-exported here for backwards compatibility.
ContextBuilder = ContextBuilder


class RAGPipeline:
    """End-to-end RAG orchestration."""

    def __init__(
        self,
        vectorstore: VectorStore | None = None,
        embedder: Embedder | None = None,
        retriever: Retriever | None = None,
        ollama: OllamaClient | None = None,
        context_builder: ContextBuilder | None = None,
        router: QueryRouter | None = None,
    ) -> None:
        self.vectorstore = vectorstore or VectorStore()
        self.embedder = embedder or Embedder()
        if retriever is None:
            reranker = Reranker() if get_config().get("reranker_enabled", True) else None
            retriever = Retriever(
                vectorstore=self.vectorstore,
                embedder=self.embedder,
                reranker=reranker,
            )
        self.retriever = retriever
        self.ollama = ollama or OllamaClient()
        self.context_builder = context_builder or ContextBuilder()
        self.router = router or QueryRouter()
        self._fallback_used = False
        # Lazily created on first ask() so tests / bare instantiation never
        # open the SQLite runtime store or the embedding model eagerly.
        self._semantic_cache_instance: SemanticCache | None = None
        self._memory_instance: RetrievalMemory | None = None

    def ask(
        self,
        question: str,
        debug: bool = False,
        *,
        skip_cache: bool = False,
        history: list[dict] | None = None,
    ) -> RAGResult:
        """Answer a question using the full RAG pipeline.

        ``skip_cache`` forces full retrieval + generation (evaluation / debug
        paths); the question is still recorded for analytics.

        ``history`` is an OPTIONAL ordered list of ``{"role", "content"}``
        turns used ONLY to resolve references in follow-up questions. It is
        conservative: explicit information in the current message always wins,
        and conversation context never feeds the semantic cache or events.
        """
        question = (question or "").strip()
        if not question:
            raise ValueError("Question must not be empty.")
        conversation_request = bool(history)
        t_total_start = time.perf_counter()

        if not self.vectorstore.is_built():
            raise RuntimeError(
                "Vector index is not built. Run: python scripts/build_index.py"
            )

        # Index compatibility: never silently use an out-of-date index.
        compat_errors = self.vectorstore.compatibility_errors()
        if compat_errors:
            raise RuntimeError("; ".join(compat_errors))
        for warn in self.vectorstore.compatibility_warnings():
            logger.warning("%s", warn)

        timings: dict[str, float] = {}
        question_id = new_id("q")

        # 0. Query understanding (deterministic; routes intent/language/entities).
        t0 = time.perf_counter()
        understanding = understand(question)
        route = understanding.route
        intent = understanding.intent
        timings["query_understanding_ms"] = round((time.perf_counter() - t0) * 1000, 3)
        if debug:
            logger.info("Understanding: %s", understanding.to_dict())

        kb_version = self.vectorstore.kb_version()
        feedback_state = self._feedback_state(kb_version, understanding)
        feedback_rating = (feedback_state or {}).get("rating")
        feedback_requires_regen = feedback_rating in {"somewhat", "not_useful"}
        if feedback_state:
            logger.info(
                "[FEEDBACK] previous_state=%s response_id=%s normalized=%s",
                feedback_rating,
                feedback_state.get("question_id"),
                understanding.normalized_question,
            )

        if (
            feedback_rating == "useful"
            and not skip_cache
            and not conversation_request
            and get_config().get("cache_enabled", True)
        ):
            approved_answer = (feedback_state.get("answer") or "").strip()
            if approved_answer and approved_answer not in (REFUSAL_EN, REFUSAL_AR):
                approved_answer, formatter_issues = format_final_answer(approved_answer)
                approved_sources = self._event_sources(feedback_state)
                timings["total_time"] = round(time.perf_counter() - t_total_start, 3)
                self._record_event(
                    question_id=question_id, kb_version=kb_version,
                    understanding=understanding, answer=approved_answer,
                    sources=approved_sources,
                    latency_ms=timings.get("total_time", 0.0),
                    cache_hit=True, cache_entry_id=None,
                    retrieval_meta={
                        "generation_route": "approved_feedback",
                        "context_strategy": "runtime_exact_approved",
                        "feedback_source_response_id": feedback_state.get("question_id"),
                        "formatter_issues": formatter_issues,
                    },
                )
                self._record_cluster(
                    question_id, kb_version, understanding, timings, cache_hit=True
                )
                self._log_perf(
                    question, timings, extra="approved_feedback",
                    cache_hit=True, llm_used=False,
                )
                return RAGResult(
                    answer=approved_answer,
                    sources=approved_sources,
                    intent=intent,
                    timings=timings,
                    route=route.to_dict() if route else None,
                    diagnostics=self._diagnostics(
                        route,
                        extra={
                            "cache_hit": True,
                            "approved_feedback": True,
                            "source_response_id": feedback_state.get("question_id"),
                        },
                    ),
                    question_id=question_id,
                    cache_hit=True,
                    cached_question_id=feedback_state.get("question_id"),
                    llm_used=False,
                )

        # 0b. Semantic response-cache lookup BEFORE retrieval / generation.
        cache_hit = None
        query_vector = None
        t0 = time.perf_counter()
        if (
            not skip_cache
            and not conversation_request
            and not feedback_requires_regen
            and get_config().get("cache_enabled", True)
        ):
            try:
                cache_hit, query_vector = self._semantic_cache().lookup(
                    question, understanding, kb_version
                )
            except Exception:  # pragma: no cover - cache must never break RAG
                logger.exception("Semantic cache lookup failed; running full RAG")
        elif feedback_requires_regen:
            logger.info(
                "[CACHE] decision=skip reason=feedback_%s response_id=%s",
                feedback_rating,
                feedback_state.get("question_id") if feedback_state else None,
            )
        timings["cache_lookup_ms"] = round((time.perf_counter() - t0) * 1000, 3)

        if cache_hit is not None:
            cached_answer, formatter_issues = format_final_answer(cache_hit.answer)
            timings["total_time"] = round(time.perf_counter() - t_total_start, 3)
            logger.info("Semantic cache hit (sim=%.3f) from %s",
                        cache_hit.similarity, cache_hit.question)
            self._record_event(
                question_id=question_id, kb_version=kb_version,
                understanding=understanding, answer=cached_answer,
                sources=cache_hit.sources,
                latency_ms=timings.get("total_time", 0.0),
                cache_hit=True, cache_entry_id=cache_hit.entry_id,
                retrieval_meta={
                    "cache_similarity": round(cache_hit.similarity, 4),
                    "formatter_issues": formatter_issues,
                },
            )
            self._record_cluster(
                question_id, kb_version, understanding, timings, cache_hit=True
            )
            self._log_perf(
                question, timings, extra="cache_hit", cache_hit=True, llm_used=False
            )
            return RAGResult(
                answer=cached_answer,
                sources=cache_hit.sources,
                intent=intent,
                timings=timings,
                route=route.to_dict() if route else None,
                diagnostics=self._diagnostics(route, extra={"cache_hit": True}),
                question_id=question_id,
                cache_hit=True,
                cached_question_id=cache_hit.question,
                llm_used=False,
            )

        # 0c. Follow-up / conversation context (conservative). Resolved AFTER
        # the cache lookup so conversation context can never contaminate the
        # semantic cache: only retrieval and the prompt use the merged context.
        conv_ctx = resolve_conversation(question, history) if history else None
        retrieval_question = (
            conv_ctx.retrieval_question if conv_ctx is not None and conv_ctx.active
            else question
        )
        retrieval_understanding = (
            conv_ctx.understanding if conv_ctx is not None and conv_ctx.active
            else understanding
        )
        if conv_ctx is not None and conv_ctx.active and retrieval_understanding is not None:
            intent = retrieval_understanding.intent
            route = retrieval_understanding.route

        # 0d. Multi-intent detection: split into sub-questions and retrieve each
        # separately, then answer ONE coherent, evidence-based response.
        sub_questions = [retrieval_question]
        strategy = "routed"
        if (
            get_config().get("multi_intent_enabled", True)
            and understanding.is_multi_intent
        ):
            split = split_question(retrieval_question, retrieval_understanding)
            if len(split) > 1:
                sub_questions = split
                strategy = "multi_intent"
                logger.info("Multi-intent split -> %d sub-question(s)", len(split))

        # 1. Retrieve (routing-aware with safe fallback to broad search).
        force_retrieval_refresh = feedback_requires_regen
        force_diversify = feedback_rating == "not_useful"
        if len(sub_questions) > 1:
            chunks = self._retrieve_multi(
                sub_questions,
                retrieval_understanding,
                kb_version,
                force_refresh=force_retrieval_refresh,
                force_diversify=force_diversify,
            )
        else:
            chunks = self._retrieve(
                retrieval_question,
                retrieval_understanding,
                kb_version,
                force_refresh=force_retrieval_refresh,
                force_diversify=force_diversify,
            )
        timings.update(self.retriever.last_timings)
        logger.info("Retrieved %d chunks after filtering", len(chunks))

        query_language = (
            understanding.language if understanding.language in ("ar", "en") else "en"
        )
        specific_faculty_overview = (
            (intent or "").upper() == "FACULTY"
            and route is not None
            and bool(route.faculty)
        )
        faculty_dean_question = (
            (intent or "").upper() == "PERSON"
            and route is not None
            and bool(route.faculty)
            and bool(re.search(r"(?:dean|عميد)", question, re.IGNORECASE))
        )
        university_president_question = (
            (intent or "").upper() == "PERSON"
            and route is not None
            and not route.faculty
            and bool(re.search(r"(?:president|رئيس\s+(?:جامعة|الجامعة))", question, re.IGNORECASE))
        )
        if specific_faculty_overview or faculty_dean_question:
            # Semantic ranking often puts a faculty's staff-name chunk first
            # because the short query repeats the page title. For a broad
            # "tell me about this faculty" request, hydrate the authoritative
            # page and lead with its opening vision/mission/objective sections.
            faculty_pages = [
                c for c in chunks
                if c.faculty == route.faculty
                and (c.content_type or "").lower() == "faculty"
                and c.source_url
            ]
            preferred_pages = [
                c for c in faculty_pages if (c.language or "").lower() == query_language
            ]
            if preferred_pages or faculty_pages:
                top_page = max(preferred_pages or faculty_pages, key=lambda c: c.score)
                hydrated = self.vectorstore.get_by_document(top_page.document_id)
                hydrated = [
                    c for c in hydrated
                    if c.faculty == route.faculty
                    and (c.source_url or "") == (top_page.source_url or "")
                    and (c.language or "").lower() == (top_page.language or "").lower()
                ]
                hydrated.sort(key=lambda c: c.chunk_index or 0)
                if hydrated:
                    chunks = hydrated
        if university_president_question:
            president_pages = [
                c for c in chunks
                if (c.content_type or "").lower() == "president"
                or "about-the-president" in (c.source_url or "")
                or "the-president-speech" in (c.source_url or "")
            ]
            if president_pages:
                top_page = max(
                    president_pages,
                    key=lambda c: (
                        "about-the-president" in (c.source_url or ""),
                        (c.language or "").lower() == query_language,
                        c.score,
                    ),
                )
                hydrated = self.vectorstore.get_by_document(top_page.document_id)
                hydrated = [
                    c for c in hydrated
                    if (c.source_url or "") == (top_page.source_url or "")
                ]
                hydrated.sort(key=lambda c: c.chunk_index or 0)
                if hydrated:
                    chunks = hydrated

        # 1b. Fast path: deterministic answer for structured queries
        # (location / faculties list / contact info) straight from
        # authoritative indexed chunks — no LLM call. Gated on router
        # confidence so a weak route (e.g. FACULTY 0.53) never auto-answers.
        fast_path_allowed = (
            (len(sub_questions) == 1 or (intent or "").upper() == "TUITION")
            and not feedback_requires_regen
        )
        if fast_path_allowed and get_config().get("fast_path_enabled", True):
            t0 = time.perf_counter()
            fast_path_chunks = chunks
            # A specific faculty's programs can span many adjacent chunks on
            # one authoritative faculty page. Retrieval keeps only the top
            # context chunks for LLM efficiency, but the deterministic list
            # extractor needs the complete page to avoid partial answers.
            if (
                (intent or "").upper() == "PROGRAM"
                and route is not None
                and route.faculty
            ):
                faculty_pages = [
                    c for c in chunks
                    if (c.content_type or "").lower() == "faculty"
                    and c.source_url
                ]
                if faculty_pages:
                    top_page = max(faculty_pages, key=lambda c: c.score)
                    hydrated = self.vectorstore.get_by_document(top_page.document_id)
                    hydrated = [
                        c for c in hydrated
                        if (c.source_url or "") == (top_page.source_url or "")
                    ]
                    if hydrated:
                        fast_path_chunks = hydrated
            person_fast = (
                person_evidence_answer(question, fast_path_chunks, query_language)
                if (intent or "").upper() == "PERSON"
                else None
            )
            if person_fast is not None:
                fast_answer, used_chunks = person_fast
            else:
                fast_answer, used_chunks = try_fast_answer(
                    question, intent, fast_path_chunks, query_language,
                    route_confidence=route.confidence if route is not None else None,
                )
            if fast_answer is not None:
                fast_answer, formatter_issues = format_final_answer(fast_answer)
                timings["fast_path_time"] = round(time.perf_counter() - t0, 3)
                timings["total_time"] = round(time.perf_counter() - t_total_start, 3)
                fast_sources = self.context_builder.sources(used_chunks)
                logger.info(
                    "Fast path answered (intent=%s, %d source(s))",
                    intent, len(fast_sources),
                )
                self._record_event(
                    question_id=question_id, kb_version=kb_version,
                    understanding=understanding, answer=fast_answer,
                    sources=fast_sources,
                    latency_ms=timings.get("total_time", 0.0),
                    cache_hit=False, cache_entry_id=None,
                    retrieval_meta=self._retrieval_metadata(
                        understanding, chunks, fast_sources,
                        strategy=strategy,
                        generation_route="fast_path",
                        context_strategy="deterministic",
                        validation={"issues": []},
                        extra={
                        "fast_path": True,
                        "formatter_issues": formatter_issues,
                        },
                    ),
                )
                self._record_cluster(
                    question_id, kb_version, understanding, timings, cache_hit=False
                )
                self._log_perf(
                    question, timings, extra="fast_path", llm_used=False
                )
                return RAGResult(
                    answer=fast_answer,
                    sources=fast_sources,
                    retrieved_chunks=chunks,
                    intent=intent,
                    timings=timings,
                    route=route.to_dict() if route else None,
                    diagnostics=self._diagnostics(route, extra={"fast_path": True}),
                    question_id=question_id,
                    llm_used=False,
                )

        # 2. Build context + sources (compressed to the token budget).
        # List-bearing intents use a larger char + token budget so a complete
        # directory page is never truncated (Phase 4 recovery fix for
        # "incomplete answers" / "over-compression").
        t0 = time.perf_counter()
        intent_max_chunks = self._intent_context_chunks(intent, question)
        if specific_faculty_overview:
            # A concise overview needs only the strongest page sections. Long
            # faculty pages can otherwise keep a 4B CPU model busy for minutes.
            intent_max_chunks = min(intent_max_chunks, 3)
        broad_fee = (intent or "").upper() == "TUITION" and any(
            marker in question.lower()
            for marker in (
                "all", "faculties", "faculty fees", "annual", "سنوي",
                "السنوية", "الكليات", "كل الكليات", "جميع",
            )
        )
        list_like = (
            ((intent or "").upper() in _LIST_LIKE_INTENTS)
            and not specific_faculty_overview
        ) or broad_fee
        cfg = get_config()
        context = self.context_builder.build(
            chunks,
            max_chunks=intent_max_chunks,
            max_chars=(
                int(cfg.get("context_max_chars_list", 8000)) if list_like else None
            ),
        )
        compressed = ContextCompressor(
            max_tokens=(
                int(cfg.get("context_max_tokens_list", 8000)) if list_like else None
            )
        ).compress(context, question)
        context_was_compressed = compressed != context
        if context_was_compressed:
            logger.info("Context compressed: %d -> %d chars",
                        len(context), len(compressed))
            context = compressed
        sources = self.context_builder.sources(chunks)
        timings["context_assembly_time"] = round(time.perf_counter() - t0, 3)

        # 3. Handle empty / unsupported retrieval.
        if not context:
            answer = refusal_text(query_language)
            timings["total_time"] = round(time.perf_counter() - t_total_start, 3)
            logger.warning("Empty retrieval for question; returning grounded refusal.")
            self._record_event(
                question_id=question_id, kb_version=kb_version,
                understanding=understanding, answer=answer, sources=[],
                latency_ms=timings.get("total_time", 0.0),
                cache_hit=False, cache_entry_id=None,
                retrieval_meta=self._retrieval_metadata(
                    understanding, chunks, [],
                    strategy=strategy,
                    generation_route="refusal",
                    context_strategy="empty",
                    validation={"issues": ["empty_retrieval"]},
                    extra={"empty_retrieval": True},
                ),
            )
            self._record_cluster(
                question_id, kb_version, understanding, timings, cache_hit=False
            )
            self._log_perf(
                question, timings, extra="empty_retrieval", llm_used=False
            )
            return RAGResult(
                answer=answer,
                sources=[],
                retrieved_chunks=chunks,
                intent=intent,
                timings=timings,
                route=route.to_dict() if route else None,
                diagnostics=self._diagnostics(route),
                question_id=question_id,
                llm_used=False,
            )

        # 4. Generate (retry once if Ollama returns a blank response).
        user_prompt = build_rag_prompt(
            question, context, language=query_language, intent=intent,
            conversation=(conv_ctx.prompt_block if conv_ctx is not None and conv_ctx.active else None),
        )
        t0 = time.perf_counter()
        generation_options = self._generation_options(intent)
        answer = self.ollama.generate(SYSTEM_PROMPT, user_prompt, **generation_options)
        if not (answer or "").strip():
            logger.warning("Ollama returned an empty answer; retrying once.")
            answer = self.ollama.generate(
                SYSTEM_PROMPT, user_prompt, **generation_options
            )
        answer, stripped_reasoning = strip_reasoning_artifacts(answer or "")
        if stripped_reasoning:
            logger.warning("Stripped reasoning artifact from generated answer.")

        # 4b. Deterministic intent-relevance guard (no extra LLM call for
        # validation): if the LLM answered a clearly DIFFERENT topic than the
        # routed intent (e.g. the founding decree for a LOCATION question),
        # regenerate once, then fall through to the grounded list fallback /
        # refusal paths below.
        if (answer or "").strip() and not answer_relevance_ok(answer, intent):
            logger.warning(
                "Answer failed intent-relevance gate (intent=%s); regenerating once.",
                intent,
            )
            answer = self.ollama.generate(
                SYSTEM_PROMPT, user_prompt, **generation_options
            )
            answer, stripped_reasoning = strip_reasoning_artifacts(answer or "")
            if stripped_reasoning:
                logger.warning("Stripped reasoning artifact from regenerated answer.")
            if not answer_relevance_ok(answer, intent):
                answer = ""
        timings["ollama_request_time"] = round(time.perf_counter() - t0, 3)

        # Defensive grounded fallback for LIST intents only: if the LLM stalls
        # (qwen3-vl can over-think long Arabic lists on CPU), synthesize the
        # answer directly from the retrieved directory chunk. Non-list intents
        # must NOT use this (it could produce a misleading list).
        if not (answer or "").strip() and is_list_intent(intent):
            answer = self._fallback_list_answer(chunks, intent, query_language)
            logger.warning("Using grounded fallback list answer.")
            self._fallback_used = True

        # 5. Deterministic answer validation (no fabricated URLs, non-empty)
        # plus a soft quality score (verbosity / evidence / multi-intent
        # coverage). The hard gate still decides OK vs refusal.
        validation = validate_answer(
            answer, sources, chunks, question_language=query_language
        )
        validation["issues"].extend(
            completeness_issues(
                validation.get("cleaned") or answer,
                chunks,
                question=question,
                intent=intent,
                is_multi_intent=understanding.is_multi_intent,
            )
        )
        if self._should_repair_answer(validation):
            repair_prompt = build_repair_prompt(
                question,
                context,
                validation.get("cleaned") or answer,
                validation.get("issues") or [],
                language=query_language,
                intent=intent,
            )
            logger.warning(
                "Answer failed soft output-quality gate (%s); regenerating once.",
                validation.get("issues"),
            )
            repaired = self.ollama.generate(
                SYSTEM_PROMPT, repair_prompt, **generation_options
            )
            repair_validation = validate_answer(
                repaired, sources, chunks, question_language=query_language
            )
            repair_validation["issues"].extend(
                completeness_issues(
                    repair_validation.get("cleaned") or repaired,
                    chunks,
                    question=question,
                    intent=intent,
                    is_multi_intent=understanding.is_multi_intent,
                )
            )
            if (
                repair_validation["ok"]
                and not self._should_repair_answer(repair_validation)
                and (repair_validation.get("cleaned") or "").strip()
            ):
                answer = repair_validation["cleaned"]
                validation = repair_validation
                validation["issues"].append("regenerated_after_validation")
            else:
                validation["issues"].append("regeneration_failed_validation")

        # Qwen3 may exhaust its visible-answer budget on private reasoning or
        # leak meta-reasoning even though the retrieved evidence states a dean
        # explicitly. After the model has had its normal generation + repair
        # attempts, recover that narrow factual answer directly from the
        # retrieved evidence. No name is hardcoded and conflicting matches are
        # rejected.
        if (
            (intent or "").upper() == "PERSON"
            and (
                not validation.get("ok")
                or self._should_repair_answer(validation)
                or not (validation.get("cleaned") or "").strip()
            )
        ):
            person_fallback = person_evidence_answer(
                question, chunks, query_language
            )
            if person_fallback is not None:
                answer, person_chunks = person_fallback
                sources = self.context_builder.sources(person_chunks)
                validation = validate_answer(
                    answer, sources, person_chunks,
                    question_language=query_language,
                )
                validation["issues"].append("person_evidence_fallback")
                self._fallback_used = True
                logger.warning("Using grounded person evidence fallback.")
        # Recovery (Phase 4): remember whether the hard gate PASSED before the
        # refusal replacement below, so refusals / invalid answers are never
        # cached or written to retrieval memory (previously a refusal text was
        # stored at quality ~0.65 and served to similar questions, and its
        # sources were re-seeded forever -> "repeats old/wrong answers").
        valid_ok = bool(validation["ok"])
        if not validation["ok"]:
            answer = refusal_text(query_language)
            logger.warning("Answer invalidated (%s); returning refusal.",
                           validation["issues"])
        else:
            answer = validation["cleaned"]
        answer, formatter_issues = format_final_answer(answer)
        if formatter_issues:
            logger.warning("Final answer formatter applied: %s", formatter_issues)
        if not (answer or "").strip():
            answer = refusal_text(query_language)
            valid_ok = False
            logger.warning("Answer emptied by final formatter; returning refusal.")

        quality = {}
        if get_config().get("quality_validation_enabled", True):
            quality = evaluate_answer(answer, understanding, sources, chunks)
        quality_score = quality.get(
            "score", initial_quality_score(answer, understanding.is_multi_intent)
        )

        timings["total_time"] = round(time.perf_counter() - t_total_start, 3)
        self._record_event(
            question_id=question_id, kb_version=kb_version,
            understanding=understanding, answer=answer, sources=sources,
            latency_ms=timings.get("total_time", 0.0),
            cache_hit=False, cache_entry_id=None,
            retrieval_meta=self._retrieval_metadata(
                understanding, chunks, sources,
                strategy=strategy,
                generation_route="llm",
                context_strategy="compressed" if context_was_compressed else "standard",
                validation=validation,
                extra={
                    "quality_score": quality_score,
                    "quality_flags": quality.get("flags", []),
                    "formatter_issues": formatter_issues,
                },
            ),
        )
        # Only validated, non-refusal answers are persisted to the semantic
        # cache and retrieval memory (Phase 4 recovery). Refusals / empty /
        # fabricated-URL answers never contaminate future runs.
        answer_stripped = (answer or "").strip()
        grounded = (
            valid_ok
            and bool(answer_stripped)
            and answer_stripped not in (REFUSAL_EN, REFUSAL_AR)
        )
        if grounded and not conversation_request:
            self._cache_store(
                question, understanding, kb_version, answer, sources,
                quality_score, query_vector,
            )
            self._memory_remember(
                understanding, kb_version, sources, strategy,
                quality_score=quality_score,
            )
        self._record_cluster(
            question_id, kb_version, understanding, timings, cache_hit=False
        )
        self._log_perf(
            question, timings, extra=f"{strategy} score={quality_score}",
            llm_used=True,
        )

        return RAGResult(
            answer=answer,
            sources=sources,
            retrieved_chunks=chunks,
            intent=intent,
            timings=timings,
            route=route.to_dict() if route else None,
            diagnostics=self._diagnostics(
                route, {**validation, "formatter_issues": formatter_issues}
            ),
            question_id=question_id,
            llm_used=True,
        )

    # -- best-effort side effects (never break RAG) ------------------------------

    def _semantic_cache(self) -> SemanticCache:
        if self._semantic_cache_instance is None:
            self._semantic_cache_instance = SemanticCache(embedder=self.embedder)
        return self._semantic_cache_instance

    def _retrieval_memory(self) -> RetrievalMemory:
        if self._memory_instance is None:
            self._memory_instance = RetrievalMemory()
        return self._memory_instance

    def _feedback_enabled(self) -> bool:
        return bool(get_config().get("feedback_enabled", True))

    def _feedback_state(
        self, kb_version: str, understanding: QueryUnderstanding
    ) -> dict | None:
        if not self._feedback_enabled():
            return None
        try:
            return get_runtime_store().latest_feedback_for_query(
                kb_version=kb_version,
                normalized_question=understanding.normalized_question,
                semantic_group=self._semantic_group(understanding),
            )
        except Exception:  # pragma: no cover
            logger.exception("Feedback state lookup failed (ignored)")
            return None

    @staticmethod
    def _event_sources(event: dict | None) -> list[dict[str, Any]]:
        if not event:
            return []
        try:
            sources = json.loads(event.get("sources_json") or "[]")
        except (TypeError, ValueError):
            return []
        return sources if isinstance(sources, list) else []

    def _semantic_group(self, understanding: QueryUnderstanding) -> str:
        return get_runtime_store().semantic_group(
            intent=understanding.intent,
            language=understanding.language,
            category=understanding.category,
            faculty=understanding.faculty,
            topic=understanding.topic,
            subtopic=understanding.subtopic,
        )

    def _retrieval_metadata(
        self,
        understanding: QueryUnderstanding,
        chunks: list[RetrievedChunk],
        sources: list[dict[str, Any]],
        *,
        strategy: str,
        generation_route: str,
        context_strategy: str,
        validation: dict[str, Any] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Compact backend-only trace used by feedback learning.

        This deliberately stores retrieval/generation strategy signals, not
        free-form answer text. It lets feedback influence future routing,
        cache, source seeding and generation strategy without memorizing a
        corrected answer.
        """
        meta = dict(getattr(self.retriever, "last_meta", {}) or {})
        trace = dict(getattr(self.retriever, "last_trace", {}) or {})
        cfg = get_config()

        source_types = sorted({
            (c.content_type or "").strip()
            for c in chunks
            if (c.content_type or "").strip()
        })
        source_urls: list[str] = []
        seen_urls: set[str] = set()
        for item in sources:
            url = (item.get("url") or "").strip() if isinstance(item, dict) else ""
            if url and url not in seen_urls:
                seen_urls.add(url)
                source_urls.append(url)
        if not source_urls:
            for chunk in chunks:
                url = (chunk.source_url or "").strip()
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    source_urls.append(url)

        retrieval_scores = [
            {
                "chunk_id": c.chunk_id,
                "score": round(float(c.score or 0.0), 4),
                "dense": round(float(c.dense_score), 4) if c.dense_score is not None else None,
                "bm25": round(float(c.bm25_score), 4) if c.bm25_score is not None else None,
                "rerank": round(float(c.rerank_score), 4) if c.rerank_score is not None else None,
            }
            for c in chunks[:12]
        ]
        normalized = understanding.normalized_question
        semantic_group = self._semantic_group(understanding)
        validation_issues = list((validation or {}).get("issues") or [])
        retrieval_mode = "hybrid" if cfg.get("hybrid_enabled", True) else "dense"
        answer_format = "structured" if generation_route == "fast_path" else (
            "refusal" if generation_route == "refusal" else "freeform"
        )
        payload: dict[str, Any] = {
            "strategy": strategy,
            "retrieval_mode": retrieval_mode,
            "semantic_group": semantic_group,
            "question_fingerprint": get_runtime_store().question_fingerprint(normalized),
            "normalized_question": normalized,
            "intent": understanding.intent,
            "language": understanding.language,
            "category": understanding.category,
            "faculty": understanding.faculty,
            "topic": understanding.topic,
            "subtopic": understanding.subtopic,
            "query_variants": list(trace.get("query_variants") or []),
            "candidate_count": meta.get("candidate_count"),
            "reranked_count": meta.get("reranked_count"),
            "final_count": meta.get("final_count") or len(chunks),
            "top_k_used": meta.get("top_k_used"),
            "routed": bool(meta.get("routed")),
            "fallback_used": bool(meta.get("fallback_used")),
            "where": meta.get("where"),
            "feedback_diversified": bool(meta.get("feedback_diversified")),
            "feedback_strategy": meta.get("feedback_strategy"),
            "retrieved_chunk_ids": [c.chunk_id for c in chunks],
            "source_urls": source_urls[:20],
            "source_types": source_types,
            "retrieval_scores": retrieval_scores,
            "reranker_used": bool(getattr(self.retriever, "reranker_enabled", False)),
            "generation_route": generation_route,
            "context_strategy": context_strategy,
            "answer_format": answer_format,
            "validation_issues": validation_issues,
            "coverage": trace.get("coverage") or {},
        }
        if extra:
            payload.update(extra)
        signature_parts = {
            "strategy": payload.get("feedback_strategy") or payload.get("strategy"),
            "retrieval_mode": retrieval_mode,
            "reranker_used": payload["reranker_used"],
            "generation_route": generation_route,
            "context_strategy": context_strategy,
            "source_types": source_types,
        }
        raw = json.dumps(signature_parts, sort_keys=True, ensure_ascii=False)
        payload["strategy_signature"] = (
            "strategy_" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
        )
        return payload

    def _retrieve(
        self,
        question: str,
        understanding: QueryUnderstanding,
        kb_version: str,
        *,
        memory_seed: bool = True,
        candidate_factor: float | None = None,
        force_refresh: bool = False,
        force_diversify: bool = False,
    ) -> list[RetrievedChunk]:
        """Single-question retrieval with variants + retrieval-memory hints."""
        variants = retrieval_variants(understanding, question)
        seed_urls: list[str] = []
        diversify = force_diversify
        if memory_seed and get_config().get("retrieval_memory_enabled", True):
            try:
                max_seed = max(1, int(get_config().get("memory_seed_urls", 2) or 2))
                seed_urls = self._retrieval_memory().hint(
                    kb_version, understanding.normalized_question,
                    understanding=understanding,
                )[:max_seed]
                diversify = self._retrieval_memory().should_diversify(
                    kb_version, understanding.normalized_question, understanding
                )
                if seed_urls:
                    logger.info(
                        "[STRATEGY] matched_semantic_group=%s preferred_sources=%d",
                        self._semantic_group(understanding), len(seed_urls),
                    )
                if diversify:
                    logger.info(
                        "[STRATEGY] matched_semantic_group=%s previous_strategy_negative=True "
                        "next_strategy=hybrid_diversified_broad",
                        self._semantic_group(understanding),
                    )
            except Exception:  # pragma: no cover
                logger.exception("Retrieval memory hint failed (ignored)")
        primary = self.retriever.retrieve(
            question,
            intent=understanding.intent,
            route=None if diversify else understanding.route,
            query_variants=variants or None,
            memory_seed_urls=seed_urls or None,
            candidate_factor=candidate_factor,
            force_refresh=force_refresh,
        )
        if not diversify:
            coverage = (getattr(self.retriever, "last_trace", {}) or {}).get("coverage") or {}
            if (
                get_config().get("adaptive_retrieval_enabled", True)
                and coverage
                and not coverage.get("ok", True)
                # A named faculty is already a precise metadata scope. A
                # broad retry can replace its page with the university-wide
                # faculty directory, producing an unrelated list.
                and not (understanding.route and understanding.route.faculty)
            ):
                primary_meta = dict(getattr(self.retriever, "last_meta", {}) or {})
                primary_trace = dict(getattr(self.retriever, "last_trace", {}) or {})
                primary_timings = dict(getattr(self.retriever, "last_timings", {}) or {})
                logger.info(
                    "[ESCALATION] stage=broad_coverage_retry reason=missing_%s",
                    ",".join(coverage.get("missing") or []),
                )
                alt = self.retriever.retrieve(
                    question,
                    intent=understanding.intent,
                    route=None,
                    query_variants=variants or None,
                    memory_seed_urls=seed_urls or None,
                    candidate_factor=1.0,
                    force_refresh=True,
                )
                alt_coverage = (
                    (getattr(self.retriever, "last_trace", {}) or {}).get("coverage")
                    or {}
                )
                if alt_coverage.get("ok") or len(alt) >= len(primary):
                    self.retriever.last_meta["coverage_escalated"] = True
                    self.retriever.last_meta["coverage_escalation_reason"] = coverage.get("missing") or []
                    return alt
                self.retriever.last_meta = primary_meta
                self.retriever.last_trace = primary_trace
                self.retriever.last_timings = primary_timings
            return primary

        alt = self.retriever.retrieve(
            question,
            intent=understanding.intent,
            route=understanding.route,
            query_variants=variants or None,
            memory_seed_urls=None,
            candidate_factor=candidate_factor,
            force_refresh=True,
        )
        merged: dict[str, RetrievedChunk] = {c.chunk_id: c for c in primary}
        for c in alt:
            merged.setdefault(c.chunk_id, c)
        out = list(merged.values())
        self.retriever.last_meta["feedback_diversified"] = True
        self.retriever.last_meta["feedback_strategy"] = "hybrid_diversified_broad"
        top_k = int(getattr(self.retriever, "top_k", len(out)) or len(out))
        return out[:top_k]

    def _retrieve_multi(
        self,
        sub_questions: list[str],
        understanding: QueryUnderstanding,
        kb_version: str,
        *,
        force_refresh: bool = False,
        force_diversify: bool = False,
    ) -> list[RetrievedChunk]:
        """Retrieve each sub-question separately, then merge the evidence."""
        merged: dict[str, RetrievedChunk] = {}
        factor = float(get_config().get("adaptive_multi_factor", 0.6))
        for sq in sub_questions:
            sub_u = understand(sq)
            try:
                for chunk in self._retrieve(
                    sq, sub_u, kb_version,
                    memory_seed=False, candidate_factor=factor,
                    force_refresh=force_refresh,
                    force_diversify=force_diversify,
                ):
                    merged.setdefault(chunk.chunk_id, chunk)
            except Exception:  # pragma: no cover - never break the pipeline
                logger.exception("Sub-question retrieval failed: %r", sq)
        return list(merged.values())

    def _record_event(
        self, *, question_id, kb_version, understanding, answer, sources,
        latency_ms, cache_hit, cache_entry_id, retrieval_meta,
    ) -> None:
        if not self._feedback_enabled():
            return
        try:
            meta = dict(retrieval_meta or {})
            meta.setdefault("model", getattr(self.ollama, "model", "") or "auto")
            get_runtime_store().record_question_event(
                question_id=question_id, kb_version=kb_version,
                question=understanding.original_question,
                normalized_question=understanding.normalized_question,
                language=understanding.language,
                intent=understanding.intent, category=understanding.category,
                faculty=understanding.faculty,
                is_multi_intent=understanding.is_multi_intent,
                answer=answer, sources=sources, latency_ms=latency_ms,
                cache_hit=cache_hit, cache_entry_id=cache_entry_id,
                retrieval_meta=meta,
            )
        except Exception:  # pragma: no cover
            logger.exception("Failed to record question event (ignored)")

    def _record_cluster(
        self, question_id, kb_version, understanding, timings, *, cache_hit,
    ) -> None:
        if not self._feedback_enabled():
            return
        try:
            get_runtime_store().record_cluster(
                kb_version=kb_version,
                cluster_key=understanding.normalized_question,
                question_id=question_id,
                latency_ms=timings.get("total_time", 0.0),
                cache_hit=cache_hit,
            )
        except Exception:  # pragma: no cover
            logger.exception("Failed to record question cluster (ignored)")

    def _cache_store(
        self, question, understanding, kb_version, answer, sources,
        quality_score, query_vector,
    ) -> None:
        if not get_config().get("cache_enabled", True):
            return
        # Defense in depth (Phase 4 recovery): never cache an empty answer or a
        # refusal/insufficiency text — those would be served back to similar
        # questions ("says info doesn't exist when it does").
        if not (answer or "").strip():
            return
        if answer.strip() in (REFUSAL_EN, REFUSAL_AR):
            logger.info("Not caching a refusal answer (question: %r)",
                        (question or "")[:80])
            return
        try:
            self._semantic_cache().store(
                kb_version=kb_version, question=question,
                understanding=understanding, answer=answer, sources=sources,
                quality_score=quality_score, query_vector=query_vector,
            )
        except Exception:  # pragma: no cover
            logger.exception("Failed to store cache entry (ignored)")

    def _memory_remember(self, understanding, kb_version, sources, strategy,
                         quality_score: float | None = None) -> None:
        if not get_config().get("retrieval_memory_enabled", True):
            return
        try:
            # Recovery (Phase 4): optionally refuse to remember patterns for
            # low-quality answers so a wrong/refused answer's sources are never
            # re-seeded on future questions.
            gate = bool(get_config().get("memory_gate_on_quality", True))
            if gate:
                min_score = float(
                    get_config().get("memory_min_quality_score", 0.5) or 0.5
                )
                if quality_score is None or quality_score < min_score:
                    logger.info(
                        "Retrieval memory skipped (quality %.2f < %.2f)",
                        quality_score or 0.0, min_score,
                    )
                    return
            self._retrieval_memory().remember(
                kb_version=kb_version,
                normalized_question=understanding.normalized_question,
                intent=understanding.intent, category=understanding.category,
                faculty=understanding.faculty, sources=sources, strategy=strategy,
            )
        except Exception:  # pragma: no cover
            logger.exception("Failed to update retrieval memory (ignored)")

    def _log_perf(
        self, question: str, timings: dict[str, float], extra: str = "",
        *, llm_used: bool = False, cache_hit: bool = False,
    ) -> None:
        """Backend-only [PERF] instrumentation (never sent to the GUI)."""
        def _fmt(s: float) -> str:
            if s >= 60:
                return f"{s / 60:,.1f}m"
            if s >= 1:
                return f"{s:,.2f}s"
            return f"{s * 1000:,.0f}ms"

        retrieval = float(timings.get("retrieval_time", 0.0))
        reranking = float(timings.get("reranking_time", 0.0))
        bm25 = float(timings.get("bm25_time", 0.0))
        ollama = float(timings.get("ollama_request_time", 0.0))
        fast = float(timings.get("fast_path_time", 0.0))
        understand_ms = float(timings.get("query_understanding_ms", 0.0))
        cache_ms = float(timings.get("cache_lookup_ms", 0.0))
        total = float(timings.get("total_time", 0.0)) or sum(timings.values())
        meta = getattr(self.retriever, "last_meta", {}) or {}
        model = getattr(self.ollama, "model", "") or "auto"
        suffix = f" | {extra}" if extra else ""
        logger.info(
            "[PERF] %s | model=%s llm_used=%s cache_hit=%s candidates=%s "
            "reranked=%s final=%s | understand=%s cache=%s retrieval=%s "
            "bm25=%s rerank=%s context=%s fast_path=%s ollama_generation=%s "
            "total=%s%s",
            (question[:60] + "…") if len(question) > 60 else question,
            model, llm_used, cache_hit,
            meta.get("candidate_count", "-"), meta.get("reranked_count", "-"),
            meta.get("final_count", "-"),
            f"{understand_ms:.0f}ms", f"{cache_ms:.0f}ms",
            _fmt(retrieval), _fmt(bm25), _fmt(reranking),
            _fmt(float(timings.get("context_assembly_time", 0.0))),
            _fmt(fast), _fmt(ollama), _fmt(total), suffix,
        )

    def _intent_context_chunks(self, intent: str | None, question: str = "") -> int:
        """Chunk groups to send to the LLM for this intent (bounded)."""
        table = get_config().get("intent_context_chunks", {}) or {}
        cap = int(
            get_config().get("top_context_chunks")
            or get_config().get("max_context_chunks", 6)
            or 6
        )
        key = (intent or "FACT").lower()
        count = int(table.get(key, 4))
        q = (question or "").lower()
        broad_fee = key == "tuition" and any(
            marker in q
            for marker in (
                "all", "faculties", "faculty fees", "annual", "سنوي",
                "السنوية", "الكليات", "كل الكليات", "جميع",
            )
        )
        if broad_fee:
            count = max(count, int(table.get("list", 6)))
        return max(1, min(count, cap))

    def _generation_options(self, intent: str | None) -> dict[str, int]:
        """Bound Ollama output by answer type.

        The global ``OLLAMA_MAX_OUTPUT_TOKENS`` remains a hard ceiling. Simple
        factual questions get a smaller ``num_predict`` so local models cannot
        spend minutes producing analysis; list/comparison intents still get
        enough room for complete tables or numbered lists.
        """
        cfg = get_config()
        ceiling = int(
            cfg.get("ollama_max_output_tokens")
            or cfg.get("max_generation_tokens", 1200)
            or 1200
        )
        intent_upper = (intent or "FACT").upper()
        thinking_enabled = bool((cfg.get("ollama_options") or {}).get("think"))
        if intent_upper in {"LOCATION", "CONTACT", "PERSON", "FAQ", "FACT", "FACULTY"}:
            # Qwen3 spends from num_predict on its private thinking tokens too.
            # A 512-token factual budget can therefore finish the reasoning
            # with an empty visible content field. Keep the small fast budget
            # for direct mode, but leave enough room for a final answer when
            # OLLAMA_THINK=true.
            budget = 2500 if thinking_enabled else 512
        elif intent_upper in {"PROGRAM", "TUITION", "SCHOLARSHIP", "LIST"}:
            budget = 4000 if thinking_enabled else 1400
        elif intent_upper in {"COMPARISON", "REGULATION", "ADMINISTRATION"}:
            budget = 4000 if thinking_enabled else 1800
        else:
            budget = 3000 if thinking_enabled else 900
        return {
            "num_predict": max(128, min(budget, ceiling)),
            "stop": [
                "USER QUESTION:",
                "RETRIEVED EVIDENCE:",
                "END EVIDENCE",
                "TASK:",
                "Source 1",
                "Evidence item 1",
                "<think>",
                "</think>",
            ],
          }

    @staticmethod
    def _should_repair_answer(validation: dict[str, Any] | None) -> bool:
        """Whether a valid-but-messy answer deserves one controlled retry."""
        if not validation:
            return False
        issues = set(validation.get("issues") or [])
        prefixes = ("incomplete_evidence_coverage:", "fabricated_url:")
        hard_soft = {
            "reasoning_artifact_remaining",
            "source_or_context_leakage",
            "excessive_repetition",
            "language_mismatch",
            "emptied_after_cleanup",
        }
        return bool(issues & hard_soft) or any(
            any(issue.startswith(prefix) for prefix in prefixes)
            for issue in issues
        )

    def _diagnostics(
        self,
        route,
        validation: dict | None = None,
        extra: dict | None = None,
    ) -> dict:
        meta = dict(getattr(self.retriever, "last_meta", {}) or {})
        out = {
            "route": route.to_dict() if route is not None else None,
            "candidate_count": meta.get("candidate_count"),
            "reranked_count": meta.get("reranked_count"),
            "final_count": meta.get("final_count"),
            "top_k_used": meta.get("top_k_used"),
            "routed": meta.get("routed"),
            "fallback_used": meta.get("fallback_used", self._fallback_used),
            "cache_hit": meta.get("cache_hit", False),
            "where": meta.get("where"),
            "fast_path": False,
        }
        trace = getattr(self.retriever, "last_trace", None)
        if trace:
            out["coverage"] = trace.get("coverage")
            out["retrieval_trace_summary"] = {
                "stages": {
                    name: len(items)
                    for name, items in (trace.get("stages") or {}).items()
                },
                "removed": len(trace.get("removed") or []),
            }
        if validation is not None:
            out["validation_issues"] = validation["issues"]
        if extra:
            out.update(extra)
        return out

    def _fallback_list_answer(
        self, chunks: list[RetrievedChunk], intent: str, language: str = "en"
    ) -> str:
        """Build a grounded list answer straight from the retrieved directory.

        Used only when the LLM returned no usable text for a LIST intent. The
        best list-bearing chunk (preferring the question's language and the
        content type that matches the intent) is split into items and returned.
        Source URLs are NOT embedded in the answer text; they surface through
        the API ``sources`` field instead.
        """
        list_types = self.retriever.list_source_types or set()
        # Prefer content types that match the intent's category (e.g.
        # admission lists from an "admission" page, programs from "program").
        preferred_types: set[str] = set()
        intent_lower = (intent or "").lower()
        if intent_lower in {"faculty", "program", "list"}:
            preferred_types = {"program", "faculty", "about"}
        candidates = [
            c for c in chunks if (c.content_type or "").lower() in list_types
        ]
        if not candidates:
            return (
                "The available knowledge base does not contain enough information "
                "to answer this question."
            )
        def _items(text: str) -> list[str]:
            normalized = re.sub(r"\s*اعرف المزيد\s*", "|", text)
            return [
                part.strip()
                for part in re.split(r"[|\n]+", normalized)
                if part.strip() and len(part.strip()) > 1
            ]

        def _preference(c) -> tuple:
            content_type = (c.content_type or "").lower()
            type_rank = 3 if content_type in preferred_types else 1
            lang_rank = 1 if (c.language or "").lower() == "ar" else 0
            return type_rank, {"program": 4, "faculty": 3, "administration": 2}.get(
                content_type, 1
            ), lang_rank, c.score

        best = max(candidates, key=_preference)
        items = _items(best.text)
        if not items:
            return best.text
        bullet = "\n".join(f"- {item}" for item in items)
        if language == "ar":
            lead = "بحسب المصادر الرسمية لجامعة المنصورة الجديدة، فإن العناصر هي:"
        else:
            lead = "Based on the official NMU sources, the items are:"
        return f"{lead}\n{bullet}"

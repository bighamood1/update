"""Persistent semantic response cache.

Repeated or highly similar questions skip the full RAG pipeline. A cached
answer is only returned when ALL safety gates pass:

1. embedding similarity >= ``CACHE_SIMILARITY_THRESHOLD`` (and a stricter
   ``CACHE_GENERIC_SIMILARITY_THRESHOLD`` when either side is a generic intent),
2. the cache entry belongs to the same ``knowledge_base_version``,
3. the intent matches EXACTLY (a generic/uncertain intent never reuses a
   cached answer meant for a different intent),
4. the routing metadata is compatible (language, category, faculty),
5. the entry's feedback-derived ``quality_score`` >= ``CACHE_MIN_QUALITY_SCORE``.

Uncertain similarity, stale knowledge base or poor feedback all bypass the
cache and run full RAG. A cache failure NEVER breaks the pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import get_config
from ..embeddings.embedder import Embedder
from ..query.understanding import QueryUnderstanding
from ..utils.logging_utils import get_logger
from .store import RuntimeStore, get_runtime_store

logger = get_logger(__name__)

# Intents that carry little topic-specific signal (e.g. a plain "ما هي جامعة
# المنصورة الجديدة؟" routes to LIST). A cached answer stored under one of
# these is ONLY reusable for a question with the EXACT same intent AND a much
# higher similarity, because the entity name alone (جامعة المنصورة الجديدة)
# can make semantically different questions look near-identical to an embedding.
_GENERIC_INTENTS = {"FACT", "GENERAL", "FAQ", "UNKNOWN", "LIST"}


@dataclass
class CacheHit:
    entry_id: int
    question: str
    answer: str
    sources: list[dict]
    intent: str
    similarity: float


class SemanticCache:
    """Embedding-similarity response cache backed by the runtime SQLite store."""

    def __init__(
        self,
        embedder: Embedder | None = None,
        store: RuntimeStore | None = None,
    ) -> None:
        self.embedder = embedder or Embedder()
        # NOTE: stored as self._db (NOT self.store) so it cannot shadow the
        # public ``store()`` method below — previously the attribute shadowed
        # the method and every cache write raised ``TypeError`` which was
        # swallowed, so the semantic cache silently never persisted anything.
        self._db = store or get_runtime_store()
        self._entries: dict[str, list[dict]] = {}
        self._cache_key: str | None = None

    # -- helpers -------------------------------------------------------------

    def _load_entries(self, kb_version: str) -> list[dict]:
        if self._cache_key != kb_version:
            self._entries = {}
            self._cache_key = kb_version
        if kb_version not in self._entries:
            self._entries[kb_version] = self._db.find_cache_hits(kb_version)
            logger.info("Loaded %d cache entries (kb=%s)", len(self._entries[kb_version]), kb_version)
        return self._entries[kb_version]

    def _embed(self, text: str) -> np.ndarray:
        return np.asarray(self.embedder.embed_query(text), dtype=np.float32)

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        if denom <= 0.0:
            return 0.0
        return float(float(np.dot(a, b)) / denom)

    @staticmethod
    def _intent_compatible(entry_intent: str | None, current_intent: str) -> bool:
        """Exact-match intent gate for a semantic cache hit.

        A cached answer is only reused when the cached intent equals the
        current intent. A missing/unknown cached intent (legacy entry) is
        treated as unsafe — it can never be reused semantically, and a
        generic/uncertain intent never matches a different intent. This is the
        primary guard against serving one question's answer for another
        question that merely shares the same entity name.
        """
        e = (entry_intent or "").strip().upper()
        c = (current_intent or "").strip().upper()
        if not e or not c:
            return False
        return e == c

    @staticmethod
    def _metadata_compatible(
        row: dict, understanding: QueryUnderstanding
    ) -> tuple[bool, str]:
        """Routing-metadata gate (language / category / faculty).

        Returns ``(ok, reason)``. Incompatible metadata (or missing metadata
        on legacy entries where it is required) forces a CACHE MISS.
        """
        # Language: an Arabic answer must not be served for an English question.
        cached_lang = (row.get("language") or "").strip().lower()
        q_lang = (understanding.language or "").strip().lower()
        if cached_lang in ("ar", "en") and q_lang in ("ar", "en") and cached_lang != q_lang:
            return False, f"language {cached_lang} != {q_lang}"
        # Category: equal intents imply equal categories, but legacy rows may
        # carry a conflicting category — never reuse those.
        cached_cat = (row.get("category") or "").strip().upper()
        q_cat = (understanding.category or "").strip().upper()
        if cached_cat and q_cat and cached_cat != q_cat:
            return False, f"category {cached_cat} != {q_cat}"
        # Faculty: a question scoped to one faculty must never reuse a general
        # (or different-faculty) cached answer.
        cached_fac = (row.get("faculty") or "").strip().upper()
        q_fac = (understanding.faculty or "").strip().upper()
        if q_fac and (not cached_fac or cached_fac != q_fac):
            return False, f"faculty {cached_fac or '?'} != {q_fac}"
        return True, ""

    # -- lookup ----------------------------------------------------------------

    def lookup(
        self,
        question: str,
        understanding: QueryUnderstanding,
        kb_version: str,
    ) -> tuple[CacheHit | None, np.ndarray | None]:
        """Return ``(hit, query_vector)``. ``query_vector`` is reused by store().

        A miss returns ``(None, vector)`` so the pipeline does not embed the
        question twice. ``None`` vector means the cache is disabled/failed.
        """
        if not get_config().get("cache_enabled", True) or not self._db.enabled:
            return None, None
        try:
            query_vec = self._embed(understanding.normalized_question or question)
        except Exception:  # noqa: BLE001 - cache must never break RAG
            logger.exception("Cache embedding failed; bypassing cache")
            return None, None

        threshold = float(get_config().get("cache_similarity_threshold", 0.92))
        # Generic/uncertain intents need a much stronger match: the entity name
        # alone can drive similarity above the normal threshold for two
        # semantically different questions.
        generic_threshold = float(
            get_config().get("cache_generic_similarity_threshold", 0.97)
        )
        min_quality = float(get_config().get("cache_min_quality_score", 0.5))

        current_intent = (understanding.intent or "").strip().upper()

        best: CacheHit | None = None
        best_sim = -1.0
        try:
            for row in self._load_entries(kb_version):
                try:
                    vec = np.frombuffer(row["embedding"], dtype=np.float32)
                except Exception:  # noqa: BLE001 - skip malformed rows
                    continue
                if vec.size != query_vec.size:
                    continue
                sim = self._cosine(query_vec, vec)
                if sim < threshold:
                    continue  # too dissimilar to be a candidate
                feedback_status = (row.get("feedback_status") or "UNKNOWN").upper()
                if feedback_status in {"SOMEWHAT", "NOT_USEFUL"}:
                    logger.info(
                        "[SEMANTIC CACHE] decision=CACHE_MISS sim=%.3f entry=%d "
                        "reason=feedback_status_%s",
                        sim, row.get("id"), feedback_status,
                    )
                    continue
                cached_intent = (row.get("intent") or "").strip().upper()
                eff_threshold = (
                    generic_threshold
                    if cached_intent in _GENERIC_INTENTS
                    or current_intent in _GENERIC_INTENTS
                    else threshold
                )
                if sim < eff_threshold:
                    logger.info(
                        "[SEMANTIC CACHE] decision=CACHE_MISS sim=%.3f entry=%d "
                        "current_intent=%s cached_intent=%s intent_compatible=False "
                        "reason=generic_similarity_below_%.2f",
                        sim, row.get("id"), current_intent or "",
                        cached_intent or "", eff_threshold,
                    )
                    continue
                if sim < best_sim:
                    continue
                intent_ok = self._intent_compatible(cached_intent, current_intent)
                meta_ok, meta_reason = self._metadata_compatible(row, understanding)
                if not intent_ok or not meta_ok:
                    logger.info(
                        "[SEMANTIC CACHE] decision=CACHE_MISS sim=%.3f entry=%d "
                        "current_intent=%s cached_intent=%s intent_compatible=%s "
                        "metadata=%s",
                        sim, row.get("id"), current_intent or "",
                        cached_intent or "", intent_ok, meta_reason or "ok",
                    )
                    continue
                if float(row.get("quality_score", 0.0) or 0.0) < min_quality:
                    logger.info(
                        "[SEMANTIC CACHE] decision=CACHE_MISS sim=%.3f entry=%d "
                        "current_intent=%s cached_intent=%s intent_compatible=True "
                        "metadata=ok reason=quality_%.2f_below_min_%.2f",
                        sim, row.get("id"), current_intent or "",
                        cached_intent or "",
                        float(row.get("quality_score", 0.0) or 0.0), min_quality,
                    )
                    continue
                best = CacheHit(
                    entry_id=int(row["id"]),
                    question=row.get("question") or "",
                    answer=row.get("answer") or "",
                    sources=self._decode_sources(row.get("sources_json")),
                    intent=row.get("intent") or "",
                    similarity=sim,
                )
                best_sim = sim
        except Exception:  # noqa: BLE001 - cache must never break RAG
            logger.exception("Semantic cache lookup failed; bypassing")
            return None, query_vec

        if best is not None:
            logger.info(
                "[SEMANTIC CACHE] decision=CACHE_HIT sim=%.3f entry=%d "
                "current_intent=%s cached_intent=%s intent_compatible=True "
                "metadata=ok",
                best_sim, best.entry_id, current_intent or "", best.intent,
            )
            self._db.bump_cache_usage(best.entry_id)
            return best, query_vec
        return None, query_vec

    @staticmethod
    def _decode_sources(raw: str | None) -> list[dict]:
        if not raw:
            return []
        try:
            import json

            return json.loads(raw)
        except (ValueError, TypeError):  # pragma: no cover - defensive
            return []

    # -- store ----------------------------------------------------------------

    def store(
        self,
        *,
        kb_version: str,
        question: str,
        understanding: QueryUnderstanding,
        answer: str,
        sources: list[dict],
        quality_score: float,
        query_vector: np.ndarray | None = None,
    ) -> None:
        """Persist a new (generated, validated) answer into the cache."""
        if not get_config().get("cache_enabled", True) or not self._db.enabled:
            return
        # Recovery (Phase 4): never cache an empty / refusal / insufficiency
        # answer — caching those made the system "repeat old/wrong answers"
        # and say "info doesn't exist" for questions it actually knew.
        if not (answer or "").strip():
            return
        if any(
            (answer or "").strip() == text
            for text in (
                "I couldn't find enough information in the official NMU knowledge "
                "base to answer this reliably.",
                "لم أتمكن من العثور على معلومات كافية في قاعدة المعرفة الرسمية "
                "لجامعة المنصورة الجديدة للإجابة بشكل موثوق.",
            )
        ):
            logger.info("Not caching a refusal answer")
            return
        try:
            vector = query_vector if query_vector is not None else self._embed(
                understanding.normalized_question or question
            )
            if vector is None or np.asarray(vector).size == 0:
                return
            entry_id = self._db.upsert_cache_entry(
                kb_version=kb_version,
                embedding=vector,
                question=question,
                normalized_question=understanding.normalized_question,
                language=understanding.language,
                intent=understanding.intent,
                category=understanding.category,
                faculty=understanding.faculty,
                answer=answer,
                sources=sources,
                quality_score=max(0.0, min(1.0, quality_score)),
            )
            self._db.remember_kb_version(kb_version)
            if entry_id is not None:
                self._cache_key = None  # force reload on next lookup
        except Exception:  # noqa: BLE001 - cache must never break RAG
            logger.exception("Semantic cache store failed; continuing without cache")

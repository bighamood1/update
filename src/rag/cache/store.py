"""Persistent runtime store (SQLite) for the NMU assistant.

Isolated from ChromaDB (which holds the vector index). This database holds
everything that is *learned* over time:

- ``cache_entries``   semantic response cache (answer + embedding + gates)
- ``question_events`` every answered question (feedback target)
- ``feedback``        user ratings (useful / somewhat / not_useful)
- ``retrieval_memory`` successful retrieval patterns (source hints)
- ``question_clusters`` FAQ clustering / frequency analytics
- ``kb_versions``     version markers for cache invalidation

Every operation is best-effort: a storage failure is logged and NEVER breaks
the RAG pipeline (Phase 21).
"""

from __future__ import annotations

import json
import hashlib
import shutil
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from ..config import get_config
from ..utils.logging_utils import get_logger

logger = get_logger(__name__)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def new_id(prefix: str = "q") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def _strategy_confidence(success: int, partial: int, failure: int) -> float:
    total = max(1, success + partial + failure)
    raw = (success * 1.0 + partial * 0.3 - failure * 1.0) / total
    # Single feedback events are useful but not absolute.
    dampener = min(1.0, total / 3.0)
    return round(raw * dampener, 4)


def _failure_type(rating: str, reason: str | None, meta: dict) -> str | None:
    if rating == "useful":
        return None
    if reason:
        return reason
    validation = set(meta.get("validation_issues") or [])
    coverage = meta.get("coverage") or {}
    if coverage and not coverage.get("ok", True):
        return "retrieval_failure"
    if validation & {"reasoning_artifact_remaining", "source_or_context_leakage"}:
        return "generation_failure"
    if validation & {"excessive_repetition", "language_mismatch"}:
        return "formatting_failure"
    if meta.get("empty_retrieval"):
        return "knowledge_gap_or_retrieval_failure"
    return "partial_or_failed_strategy"


_SCHEMA = """
CREATE TABLE IF NOT EXISTS kb_versions (
    kb_version TEXT PRIMARY KEY,
    created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS cache_entries (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    kb_version       TEXT NOT NULL,
    embedding        BLOB NOT NULL,
    question         TEXT NOT NULL,
    normalized_question TEXT NOT NULL,
    language         TEXT,
    intent           TEXT,
    category         TEXT,
    faculty          TEXT,
    answer           TEXT NOT NULL,
    sources_json     TEXT NOT NULL,
    quality_score    REAL DEFAULT 0.5,
    rating_sum       INTEGER DEFAULT 0,
    rating_count     INTEGER DEFAULT 0,
    feedback_status  TEXT DEFAULT 'UNKNOWN',
    usage_count      INTEGER DEFAULT 1,
    last_used_at     TEXT,
    created_at       TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_cache_kb ON cache_entries(kb_version);

CREATE TABLE IF NOT EXISTS question_events (
    question_id        TEXT PRIMARY KEY,
    kb_version         TEXT,
    question           TEXT NOT NULL,
    normalized_question TEXT,
    language           TEXT,
    intent             TEXT,
    category           TEXT,
    faculty            TEXT,
    is_multi_intent    INTEGER DEFAULT 0,
    answer             TEXT,
    sources_json       TEXT,
    latency_ms         REAL,
    cache_hit          INTEGER DEFAULT 0,
    cache_entry_id     INTEGER,
    retrieval_meta_json TEXT,
    created_at         TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_events_kb ON question_events(kb_version);

CREATE TABLE IF NOT EXISTS feedback (
    feedback_id TEXT PRIMARY KEY,
    question_id TEXT NOT NULL,
    rating      TEXT NOT NULL CHECK (rating IN ('useful','somewhat','not_useful')),
    reason      TEXT,
    created_at  TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_feedback_q ON feedback(question_id);

CREATE TABLE IF NOT EXISTS retrieval_memory (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    kb_version           TEXT NOT NULL,
    normalized_question  TEXT NOT NULL,
    intent               TEXT,
    category             TEXT,
    faculty              TEXT,
    sources_json         TEXT,
    strategy             TEXT,
    good_feedback        INTEGER DEFAULT 0,
    bad_feedback         INTEGER DEFAULT 0,
    quality_score        REAL DEFAULT 0.5,
    usage_count          INTEGER DEFAULT 1,
    updated_at           TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_memory_q ON retrieval_memory(normalized_question, kb_version);

CREATE TABLE IF NOT EXISTS strategy_feedback (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    kb_version            TEXT NOT NULL,
    semantic_group        TEXT NOT NULL,
    question_fingerprint  TEXT NOT NULL,
    normalized_question   TEXT NOT NULL,
    intent                TEXT,
    language              TEXT,
    category              TEXT,
    faculty               TEXT,
    strategy_signature    TEXT NOT NULL,
    retrieval_strategy    TEXT,
    query_variants_json   TEXT,
    source_urls_json      TEXT,
    chunk_ids_json        TEXT,
    retrieval_scores_json TEXT,
    reranker_used         INTEGER DEFAULT 0,
    generation_route      TEXT,
    context_strategy      TEXT,
    answer_format         TEXT,
    failure_type          TEXT,
    feedback              TEXT NOT NULL CHECK (feedback IN ('useful','somewhat','not_useful')),
    success_count         INTEGER DEFAULT 0,
    partial_count         INTEGER DEFAULT 0,
    failure_count         INTEGER DEFAULT 0,
    confidence            REAL DEFAULT 0.0,
    created_at            TEXT NOT NULL,
    updated_at            TEXT NOT NULL
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_strategy_key
ON strategy_feedback(kb_version, semantic_group, strategy_signature);

CREATE INDEX IF NOT EXISTS idx_strategy_lookup
ON strategy_feedback(kb_version, semantic_group, confidence);

CREATE TABLE IF NOT EXISTS question_clusters (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    kb_version           TEXT NOT NULL,
    cluster_key          TEXT NOT NULL,
    member_questions_json TEXT,
    frequency            INTEGER DEFAULT 1,
    avg_rating           REAL,
    avg_latency_ms       REAL,
    success_rate         REAL,
    latest_used_at       TEXT,
    last_question_id     TEXT
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_clusters_key ON question_clusters(kb_version, cluster_key);
"""

# Runtime tables holding session/learned state. Every one of them is cleared by
# ``reset_runtime()``; the authoritative knowledge base is NEVER touched.
RUNTIME_TABLES = (
    "feedback",
    "question_events",
    "retrieval_memory",
    "strategy_feedback",
    "question_clusters",
    "cache_entries",
    "kb_versions",
)

# AUTOINCREMENT tables whose sqlite_sequence counter must also be reset.
_AUTOINCREMENT_TABLES = (
    "cache_entries", "question_clusters", "retrieval_memory", "strategy_feedback"
)


class RuntimeStore:
    """Thread-safe SQLite store; all public methods are best-effort."""

    def __init__(self, db_path: str | Path | None = None, enabled: bool = True) -> None:
        cfg = get_config()
        path = Path(db_path) if db_path is not None else cfg["runtime_db_path"]
        self._enabled = enabled
        if not enabled:
            self._conn = None
            self._lock = threading.Lock()
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(str(path), check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
            self._conn.executescript(_SCHEMA)
            self._migrate()
            self._conn.commit()
            self._lock = threading.Lock()
        except Exception:  # noqa: BLE001 - storage must never crash the app
            logger.exception("Failed to open runtime DB at %s", path)
            self._conn = None
            self._lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        return self._enabled and self._conn is not None

    # -- generic helpers -----------------------------------------------------

    def _execute(self, sql: str, params: tuple = ()) -> Any:
        if not self.enabled:
            return None
        try:
            with self._lock:
                cur = self._conn.execute(sql, params)
                self._conn.commit()
                return cur
        except Exception:  # noqa: BLE001 - best-effort storage
            logger.exception("Runtime DB query failed: %.120s", sql)
            return None

    def _fetchone(self, sql: str, params: tuple = ()) -> dict | None:
        if not self.enabled:
            return None
        try:
            with self._lock:
                row = self._conn.execute(sql, params).fetchone()
            return dict(row) if row is not None else None
        except Exception:  # noqa: BLE001 - best-effort storage
            logger.exception("Runtime DB read failed: %.120s", sql)
            return None

    def _fetchall(self, sql: str, params: tuple = ()) -> list[dict]:
        if not self.enabled:
            return []
        try:
            with self._lock:
                rows = self._conn.execute(sql, params).fetchall()
            return [dict(r) for r in rows]
        except Exception:  # noqa: BLE001 - best-effort storage
            logger.exception("Runtime DB read failed: %.120s", sql)
            return []

    def _blob_encode(self, vector: np.ndarray) -> bytes:
        return np.asarray(vector, dtype=np.float32).tobytes()

    def _blob_decode(self, blob: bytes) -> np.ndarray:
        if not blob:
            return np.zeros(0, dtype=np.float32)
        return np.frombuffer(blob, dtype=np.float32)

    def _migrate(self) -> None:
        """Best-effort runtime DB migrations for feedback learning.

        Runtime DBs are local state and may have been created by older builds.
        Keep migrations small, explicit and idempotent.
        """
        if self._conn is None:
            return
        cols = {
            row["name"]
            for row in self._conn.execute("PRAGMA table_info(cache_entries)").fetchall()
        }
        if "feedback_status" not in cols:
            self._conn.execute(
                "ALTER TABLE cache_entries ADD COLUMN feedback_status TEXT DEFAULT 'UNKNOWN'"
            )

        feedback_sql = self._conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='feedback'"
        ).fetchone()
        if feedback_sql and "medium" in (feedback_sql["sql"] or ""):
            self._conn.executescript(
                """
                ALTER TABLE feedback RENAME TO feedback_old;
                CREATE TABLE feedback (
                    feedback_id TEXT PRIMARY KEY,
                    question_id TEXT NOT NULL,
                    rating      TEXT NOT NULL CHECK (rating IN ('useful','somewhat','not_useful')),
                    reason      TEXT,
                    created_at  TEXT NOT NULL
                );
                INSERT INTO feedback (feedback_id, question_id, rating, reason, created_at)
                SELECT feedback_id, question_id,
                       CASE WHEN rating='medium' THEN 'somewhat' ELSE rating END,
                       reason, created_at
                FROM feedback_old;
                DROP TABLE feedback_old;
                CREATE INDEX IF NOT EXISTS idx_feedback_q ON feedback(question_id);
                """
            )

    @staticmethod
    def semantic_group(*, intent: str, language: str, category: str, faculty: str | None,
                       topic: str | None = None, subtopic: str | None = None) -> str:
        parts = [
            (intent or "").upper() or "FACT",
            (language or "").lower() or "unknown",
            (category or "").lower() or "general",
            (faculty or "general").lower(),
            (topic or "").lower(),
            (subtopic or "").lower(),
        ]
        return "|".join(parts)

    @staticmethod
    def question_fingerprint(normalized_question: str) -> str:
        norm = " ".join((normalized_question or "").lower().split())
        return hashlib.sha256(norm.encode("utf-8")).hexdigest()[:16]

    # -- kb versions ----------------------------------------------------------

    def remember_kb_version(self, kb_version: str) -> None:
        self._execute(
            "INSERT OR IGNORE INTO kb_versions (kb_version, created_at) VALUES (?, ?)",
            (kb_version, _now_iso()),
        )

    # -- question events ------------------------------------------------------

    def record_question_event(
        self,
        question_id: str,
        *,
        kb_version: str,
        question: str,
        normalized_question: str,
        language: str,
        intent: str,
        category: str,
        faculty: str | None,
        is_multi_intent: bool,
        answer: str,
        sources: list[dict],
        latency_ms: float | None,
        cache_hit: bool,
        cache_entry_id: int | None,
        retrieval_meta: dict | None,
    ) -> None:
        self._execute(
            """INSERT OR REPLACE INTO question_events (
                 question_id, kb_version, question, normalized_question,
                 language, intent, category, faculty, is_multi_intent,
                 answer, sources_json, latency_ms, cache_hit, cache_entry_id,
                 retrieval_meta_json, created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                question_id,
                kb_version,
                question,
                normalized_question,
                language,
                intent,
                category,
                faculty,
                1 if is_multi_intent else 0,
                answer,
                json.dumps(sources, ensure_ascii=False),
                float(latency_ms) if latency_ms is not None else None,
                1 if cache_hit else 0,
                cache_entry_id,
                json.dumps(retrieval_meta or {}, ensure_ascii=False),
                _now_iso(),
            ),
        )

    def get_question_event(self, question_id: str) -> dict | None:
        return self._fetchone(
            "SELECT * FROM question_events WHERE question_id = ?", (question_id,)
        )

    # -- feedback -------------------------------------------------------------

    def add_feedback(self, question_id: str, rating: str, reason: str | None = None) -> str | None:
        rating = "somewhat" if rating == "medium" else rating
        event = self.get_question_event(question_id)
        if event is None:
            logger.warning("Feedback for unknown question_id=%s", question_id)
            return None
        feedback_id = new_id("fb")
        self._execute(
            "INSERT OR REPLACE INTO feedback (feedback_id, question_id, rating, reason, created_at) "
            "VALUES (?,?,?,?,?)",
            (feedback_id, question_id, rating, reason, _now_iso()),
        )
        # Propagate the rating into cache + memory quality stats.
        self._apply_rating(event, rating)
        self._record_strategy_feedback(event, rating, reason)
        return feedback_id

    def _apply_rating(self, event: dict, rating: str) -> None:
        score = {"useful": 1.0, "somewhat": 0.3, "not_useful": 0.0}.get(rating, 0.5)
        status = rating.upper() if rating in {"useful", "somewhat", "not_useful"} else "UNKNOWN"
        cache_entry_id = event.get("cache_entry_id")
        if cache_entry_id is not None:
            self._execute(
                "UPDATE cache_entries SET rating_sum = rating_sum + ?, rating_count = rating_count + 1, "
                "quality_score = ?, feedback_status = ? WHERE id = ?",
                (1 if rating == "useful" else (-1 if rating == "not_useful" else 0),
                 score, status, cache_entry_id),
            )
        self._execute(
            """UPDATE retrieval_memory
               SET good_feedback = good_feedback + ?, bad_feedback = bad_feedback + ?,
                   quality_score = ?
               WHERE normalized_question = ? AND kb_version = ?""",
            (1 if rating == "useful" else 0, 1 if rating in ("not_useful", "somewhat") else 0,
             score, event.get("normalized_question") or "", event.get("kb_version") or ""),
        )
        logger.info(
            "[FEEDBACK] response_id=%s feedback=%s cache_entry=%s",
            event.get("question_id"), rating, cache_entry_id,
        )

    def _record_strategy_feedback(self, event: dict, rating: str, reason: str | None) -> None:
        try:
            meta = json.loads(event.get("retrieval_meta_json") or "{}")
        except (TypeError, ValueError):
            meta = {}
        try:
            sources = json.loads(event.get("sources_json") or "[]")
        except (TypeError, ValueError):
            sources = []

        normalized = event.get("normalized_question") or ""
        intent = event.get("intent") or ""
        language = event.get("language") or ""
        category = event.get("category") or ""
        faculty = event.get("faculty")
        semantic_group = meta.get("semantic_group") or self.semantic_group(
            intent=intent, language=language, category=category, faculty=faculty,
            topic=meta.get("topic"), subtopic=meta.get("subtopic"),
        )
        strategy_signature = meta.get("strategy_signature") or self._strategy_signature(meta)
        qfp = self.question_fingerprint(normalized)
        now = _now_iso()
        row = self._fetchone(
            """SELECT id, success_count, partial_count, failure_count
               FROM strategy_feedback
               WHERE kb_version=? AND semantic_group=? AND strategy_signature=?""",
            (event.get("kb_version") or "", semantic_group, strategy_signature),
        )
        success_inc = 1 if rating == "useful" else 0
        partial_inc = 1 if rating == "somewhat" else 0
        failure_inc = 1 if rating == "not_useful" else 0
        source_urls = [s.get("url") for s in sources if isinstance(s, dict) and s.get("url")]
        chunk_ids = meta.get("retrieved_chunk_ids") or [
            s.get("chunk_id") for s in sources if isinstance(s, dict) and s.get("chunk_id")
        ]
        if row:
            success = int(row.get("success_count") or 0) + success_inc
            partial = int(row.get("partial_count") or 0) + partial_inc
            failure = int(row.get("failure_count") or 0) + failure_inc
            confidence = _strategy_confidence(success, partial, failure)
            self._execute(
                """UPDATE strategy_feedback
                   SET feedback=?, success_count=?, partial_count=?, failure_count=?,
                       confidence=?, failure_type=?, updated_at=?
                   WHERE id=?""",
                (rating, success, partial, failure, confidence,
                 _failure_type(rating, reason, meta), now, row["id"]),
            )
        else:
            confidence = _strategy_confidence(success_inc, partial_inc, failure_inc)
            self._execute(
                """INSERT INTO strategy_feedback (
                    kb_version, semantic_group, question_fingerprint,
                    normalized_question, intent, language, category, faculty,
                    strategy_signature, retrieval_strategy, query_variants_json,
                    source_urls_json, chunk_ids_json, retrieval_scores_json,
                    reranker_used, generation_route, context_strategy,
                    answer_format, failure_type, feedback, success_count,
                    partial_count, failure_count, confidence, created_at, updated_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    event.get("kb_version") or "", semantic_group, qfp,
                    normalized, intent, language, category, faculty,
                    strategy_signature, meta.get("strategy"),
                    json.dumps(meta.get("query_variants") or [], ensure_ascii=False),
                    json.dumps(source_urls[:12], ensure_ascii=False),
                    json.dumps([c for c in chunk_ids if c][:20], ensure_ascii=False),
                    json.dumps(meta.get("retrieval_scores") or [], ensure_ascii=False),
                    1 if meta.get("reranker_used") else 0,
                    meta.get("generation_route"),
                    meta.get("context_strategy"),
                    meta.get("answer_format"),
                    _failure_type(rating, reason, meta),
                    rating, success_inc, partial_inc, failure_inc,
                    confidence, now, now,
                ),
            )
        logger.info(
            "[FEEDBACK_LEARNING] response_id=%s intent=%s semantic_group=%s "
            "strategy=%s result=%s strategy_score=%.2f",
            event.get("question_id"), intent, semantic_group,
            strategy_signature, rating, confidence,
        )

    @staticmethod
    def _strategy_signature(meta: dict) -> str:
        parts = {
            "strategy": meta.get("strategy"),
            "retrieval_mode": meta.get("retrieval_mode", "hybrid"),
            "reranker_used": bool(meta.get("reranker_used")),
            "generation_route": meta.get("generation_route"),
            "context_strategy": meta.get("context_strategy"),
            "source_types": meta.get("source_types") or [],
        }
        raw = json.dumps(parts, sort_keys=True, ensure_ascii=False)
        return "strategy_" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

    def get_feedback(self, question_id: str) -> dict | None:
        return self._fetchone(
            "SELECT * FROM feedback WHERE question_id = ? ORDER BY created_at DESC LIMIT 1",
            (question_id,),
        )

    def latest_feedback_for_query(
        self,
        *,
        kb_version: str,
        normalized_question: str,
        semantic_group: str | None = None,
    ) -> dict | None:
        """Latest exact-query feedback for the current KB version.

        This is intentionally exact on normalized question + KB version. It may
        approve an exact runtime optimization, or force regeneration after
        ``somewhat`` / ``not_useful``. Similar questions are handled by
        strategy_feedback, not by replaying old answer text.
        """
        if not normalized_question:
            return None
        rows = self._fetchall(
            """SELECT q.*, f.feedback_id, f.rating, f.reason, f.created_at AS feedback_at
               FROM feedback f
               JOIN question_events q ON q.question_id = f.question_id
               WHERE q.kb_version = ?
                 AND q.normalized_question = ?
               ORDER BY f.created_at DESC
               LIMIT 8""",
            (kb_version, normalized_question),
        )
        if not rows:
            return None
        if semantic_group:
            for row in rows:
                try:
                    meta = json.loads(row.get("retrieval_meta_json") or "{}")
                except (TypeError, ValueError):
                    meta = {}
                if meta.get("semantic_group") == semantic_group:
                    return row
        return rows[0]

    # -- semantic cache ---------------------------------------------------------

    def find_cache_hits(
        self, kb_version: str, limit: int = 2000
    ) -> list[dict]:
        return self._fetchall(
            "SELECT id, embedding, question, normalized_question, language, intent, "
            "category, faculty, answer, sources_json, quality_score, usage_count, "
            "feedback_status "
            "FROM cache_entries WHERE kb_version = ? LIMIT ?",
            (kb_version, limit),
        )

    def upsert_cache_entry(
        self,
        *,
        kb_version: str,
        embedding: np.ndarray,
        question: str,
        normalized_question: str,
        language: str,
        intent: str,
        category: str,
        faculty: str | None,
        answer: str,
        sources: list[dict],
        quality_score: float,
    ) -> int | None:
        """Insert a new cache entry; returns its row id (or None)."""
        cur = self._execute(
            """INSERT INTO cache_entries (
                 kb_version, embedding, question, normalized_question, language,
                 intent, category, faculty, answer, sources_json,
                 quality_score, usage_count, created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,1,?)""",
            (
                kb_version,
                sqlite3.Binary(self._blob_encode(embedding)),
                question,
                normalized_question,
                language,
                intent,
                category,
                faculty,
                answer,
                json.dumps(sources, ensure_ascii=False),
                quality_score,
                _now_iso(),
            ),
        )
        if cur is None or cur.lastrowid is None:
            return None
        self._prune_cache(kb_version)
        return int(cur.lastrowid)

    def _prune_cache(self, kb_version: str) -> None:
        max_entries = int(get_config().get("cache_max_entries", 2000) or 2000)
        if max_entries <= 0:
            return
        self._execute(
            """DELETE FROM cache_entries WHERE id IN (
                 SELECT id FROM cache_entries WHERE kb_version = ?
                 ORDER BY usage_count DESC, quality_score DESC
                 LIMIT -1 OFFSET ?)""",
            (kb_version, max_entries),
        )

    def bump_cache_usage(self, entry_id: int) -> None:
        self._execute(
            "UPDATE cache_entries SET usage_count = usage_count + 1, last_used_at = ? WHERE id = ?",
            (_now_iso(), entry_id),
        )

    # -- retrieval memory -------------------------------------------------------

    def upsert_retrieval_memory(
        self,
        *,
        kb_version: str,
        normalized_question: str,
        intent: str,
        category: str,
        faculty: str | None,
        sources: list[dict],
        strategy: str,
    ) -> None:
        if not normalized_question:
            return
        existing = self._fetchone(
            "SELECT id FROM retrieval_memory WHERE kb_version = ? AND normalized_question = ?",
            (kb_version, normalized_question),
        )
        now = _now_iso()
        sources_json = json.dumps(sources[:8], ensure_ascii=False)
        if existing:
            self._execute(
                """UPDATE retrieval_memory SET intent=?, category=?, faculty=?,
                     sources_json=?, strategy=?, usage_count = usage_count + 1,
                     updated_at=?
                   WHERE id = ?""",
                (intent, category, faculty, sources_json, strategy, now, existing["id"]),
            )
        else:
            self._execute(
                """INSERT INTO retrieval_memory (
                     kb_version, normalized_question, intent, category, faculty,
                     sources_json, strategy, usage_count, updated_at)
                   VALUES (?,?,?,?,?,?,?,1,?)""",
                (kb_version, normalized_question, intent, category, faculty,
                 sources_json, strategy, now),
            )

    def get_memory_hint(self, kb_version: str, normalized_question: str) -> dict | None:
        return self._fetchone(
            """SELECT normalized_question, intent, category, faculty, sources_json,
                      strategy, quality_score, usage_count
               FROM retrieval_memory
               WHERE kb_version = ? AND normalized_question = ?
               ORDER BY quality_score DESC LIMIT 1""",
            (kb_version, normalized_question),
        )

    def get_strategy_hints(
        self,
        *,
        kb_version: str,
        normalized_question: str,
        intent: str,
        language: str,
        category: str,
        faculty: str | None,
        topic: str | None = None,
        subtopic: str | None = None,
        limit: int = 5,
    ) -> list[dict]:
        semantic_group = self.semantic_group(
            intent=intent, language=language, category=category, faculty=faculty,
            topic=topic, subtopic=subtopic,
        )
        qfp = self.question_fingerprint(normalized_question)
        return self._fetchall(
            """SELECT *
               FROM strategy_feedback
               WHERE kb_version = ?
                 AND semantic_group = ?
                 AND (question_fingerprint = ?
                      OR success_count > 0
                      OR partial_count > 0
                      OR failure_count > 0
                      OR ABS(confidence) >= 0.60
                      OR success_count + partial_count + failure_count >= 2)
               ORDER BY confidence DESC, updated_at DESC
               LIMIT ?""",
            (kb_version, semantic_group, qfp, limit),
        )

    def invalidate_cache_for_response(self, question_id: str, rating: str) -> None:
        event = self.get_question_event(question_id)
        if not event or event.get("cache_entry_id") is None:
            return
        status = "USEFUL" if rating == "useful" else (
            "SOMEWHAT" if rating == "somewhat" else "NOT_USEFUL"
        )
        score = {"USEFUL": 1.0, "SOMEWHAT": 0.3, "NOT_USEFUL": 0.0}[status]
        self._execute(
            "UPDATE cache_entries SET feedback_status=?, quality_score=? WHERE id=?",
            (status, score, event.get("cache_entry_id")),
        )

    # -- question clusters (FAQ analytics) ---------------------------------------

    def record_cluster(
        self,
        *,
        kb_version: str,
        cluster_key: str,
        question_id: str,
        latency_ms: float | None,
        cache_hit: bool,
    ) -> None:
        if not cluster_key:
            return
        existing = self._fetchone(
            "SELECT id, member_questions_json, frequency FROM question_clusters "
            "WHERE kb_version = ? AND cluster_key = ?",
            (kb_version, cluster_key),
        )
        if existing:
            try:
                members = json.loads(existing.get("member_questions_json") or "[]")
            except json.JSONDecodeError:
                members = []
            if question_id not in members:
                members = (members + [question_id])[-100:]
            self._execute(
                """UPDATE question_clusters
                   SET member_questions_json = ?, frequency = frequency + 1,
                       avg_latency_ms = ?, latest_used_at = ?, last_question_id = ?
                   WHERE id = ?""",
                (json.dumps(members, ensure_ascii=False),
                 latency_ms, _now_iso(), question_id, existing["id"]),
            )
        else:
            self._execute(
                """INSERT INTO question_clusters (
                     kb_version, cluster_key, member_questions_json, frequency,
                     avg_latency_ms, latest_used_at, last_question_id)
                   VALUES (?,?,?,1,?,?,?)""",
                (kb_version, cluster_key, json.dumps([question_id], ensure_ascii=False),
                 latency_ms, _now_iso(), question_id),
            )

    # -- analytics / export -------------------------------------------------------

    def top_faqs(self, limit: int = 10) -> list[dict]:
        rows = self._fetchall(
            """SELECT cluster_key, frequency, avg_rating, avg_latency_ms,
                      last_question_id
               FROM question_clusters ORDER BY frequency DESC LIMIT ?""",
            (limit,),
        )
        return rows

    def failed_questions(self, limit: int = 20) -> list[dict]:
        return self._fetchall(
            """SELECT q.question, q.normalized_question, q.intent, q.category,
                      f.rating, f.reason
               FROM feedback f JOIN question_events q ON q.question_id = f.question_id
               WHERE f.rating = 'not_useful'
               ORDER BY f.created_at DESC LIMIT ?""",
            (limit,),
        )

    def export_training_rows(self) -> list[dict]:
        """Phase 28: clean rows for future LoRA / DPO / reranker training."""
        rows = self._fetchall(
            """SELECT q.question, q.answer, q.sources_json, q.intent, q.category,
                      q.retrieval_meta_json, f.rating
               FROM question_events q
               LEFT JOIN feedback f ON f.question_id = q.question_id
               ORDER BY q.created_at"""
        )
        out: list[dict] = []
        for r in rows:
            try:
                sources = json.loads(r.get("sources_json") or "[]")
            except json.JSONDecodeError:
                sources = []
            try:
                meta = json.loads(r.get("retrieval_meta_json") or "{}")
            except json.JSONDecodeError:
                meta = {}
            answer = (r.get("answer") or "").strip()
            if not answer:
                continue
            row = {
                "question": r.get("question"),
                "answer": answer,
                "rating": r.get("rating"),
                "sources": sources,
                "intent": r.get("intent"),
                "retrieval_context": meta,
            }
            out.append(row)
        return out

    def stats(self) -> dict:
        n_questions = self._fetchone("SELECT COUNT(*) AS c FROM question_events")
        n_feedback = self._fetchone("SELECT COUNT(*) AS c FROM feedback")
        n_cache = self._fetchone("SELECT COUNT(*) AS c FROM cache_entries")
        cache_hits = self._fetchone(
            "SELECT COUNT(*) AS c FROM question_events WHERE cache_hit = 1"
        )
        ratings = self._fetchall(
            "SELECT rating, COUNT(*) AS c FROM feedback GROUP BY rating"
        )
        return {
            "questions": (n_questions or {}).get("c", 0),
            "feedback": (n_feedback or {}).get("c", 0),
            "cache_entries": (n_cache or {}).get("c", 0),
            "cache_hit_events": (cache_hits or {}).get("c", 0),
            "ratings": {r["rating"]: r["c"] for r in ratings},
        }


_store: RuntimeStore | None = None
_store_lock = threading.Lock()


def get_runtime_store() -> RuntimeStore:
    """Process-wide singleton store (lazy, best-effort)."""
    global _store
    with _store_lock:
        if _store is None:
            cfg = get_config()
            _store = RuntimeStore(enabled=cfg.get("feedback_enabled", True))
        return _store


def reset_runtime(db_path: str | Path | None = None, *, backup: bool = True) -> dict:
    """Delete ALL runtime state (cache, feedback, events, memory, clusters).

    The knowledge base / vector index / source documents are never touched:
    this only clears the session-learned tables in ``nmu_runtime.db``.

    - optionally backs the database up to a timestamped sibling file,
    - deletes every row from every runtime table,
    - resets the ``sqlite_sequence`` counters (AUTOINCREMENT),
    - VACUUMS to reclaim space,
    - invalidates the process singleton so the next ``get_runtime_store()``
      reopens a clean (empty) store.

    Returns a report dict:
    ``{"deleted": {table: n}, "verified": {table: n}, "verified_all_empty": bool,
    "backup_path": str | None, "error": str | None}``.

    Best-effort: never raises for the pipeline.
    """
    global _store
    cfg = get_config()
    path = Path(db_path) if db_path is not None else Path(cfg["runtime_db_path"])
    report: dict[str, Any] = {
        "deleted": {}, "verified": {}, "verified_all_empty": False,
        "backup_path": None, "error": None,
    }
    try:
        backup_path: Path | None = None
        if backup and path.exists():
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = path.parent / f"nmu_runtime_backup_{stamp}.db"
            shutil.copy2(path, backup_path)
            report["backup_path"] = str(backup_path)

        path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(path))
        try:
            # Ensure the schema exists even if the file was deleted (matches the
            # app's own startup behavior, so a fresh DB starts empty).
            conn.executescript(_SCHEMA)
            for table in RUNTIME_TABLES:
                cur = conn.execute(f"DELETE FROM \"{table}\"")
                report["deleted"][table] = cur.rowcount
            for table in _AUTOINCREMENT_TABLES:
                conn.execute("DELETE FROM sqlite_sequence WHERE name = ?", (table,))
            conn.commit()
            conn.execute("VACUUM")
        finally:
            conn.close()
    except Exception:  # noqa: BLE001 - reset must never crash tooling
        logger.exception("Runtime reset failed at %s", path)
        report["error"] = str(path)
        return report

    # Invalidate the process singleton so the next call reopens an empty store
    # (and the in-memory semantic-cache entry list is dropped).
    with _store_lock:
        if _store is not None:
            try:
                if _store._conn is not None:
                    _store._conn.close()
            except Exception:  # noqa: BLE001
                pass
        _store = None

    all_empty = True
    vconn = sqlite3.connect(str(path))
    try:
        for table in RUNTIME_TABLES:
            n = vconn.execute(f"SELECT COUNT(*) FROM \"{table}\"").fetchone()[0]
            report["verified"][table] = n
            if n != 0:
                all_empty = False
    finally:
        vconn.close()
    report["verified_all_empty"] = all_empty
    return report

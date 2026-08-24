"""Bounded LRU/TTL caches for embeddings and retrieval results.

Caches are invalidated implicitly by a *signature* derived from the relevant
configuration keys and the index manifest (dataset hash + embedding model).
When any of those change, lookups miss automatically — no stale data is ever
served after a rebuild or a config change.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field

from ..config import get_config
from ..utils.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class _LRUEntry:
    value: object
    expires_at: float = 0.0  # 0 = never expires


class LRUCache:
    """Thread-safe bounded LRU cache with optional TTL."""

    def __init__(self, maxsize: int = 256, ttl: float = 0.0) -> None:
        self.maxsize = max(1, maxsize)
        self.ttl = ttl
        self._data: OrderedDict[str, _LRUEntry] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: str) -> object | None:
        with self._lock:
            entry = self._data.get(key)
            if entry is None:
                return None
            if entry.expires_at and time.time() > entry.expires_at:
                del self._data[key]
                return None
            self._data.move_to_end(key)
            return entry.value

    def put(self, key: str, value: object) -> None:
        with self._lock:
            if key in self._data:
                self._data.move_to_end(key)
            self._data[key] = _LRUEntry(
                value=value,
                expires_at=(time.time() + self.ttl) if self.ttl > 0 else 0.0,
            )
            while len(self._data) > self.maxsize:
                self._data.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)


def retrieval_signature() -> str:
    """A stable string capturing every input that affects retrieval results."""
    cfg = get_config()
    keys = [
        "embedding_model", "similarity_threshold", "rerank_candidates",
        "top_k", "hybrid_enabled", "rrf_k", "query_expansion_enabled",
        "max_query_expansion_terms", "router_enabled",
        "router_confidence_threshold", "router_filter_language",
        "router_filter_category", "router_filter_faculty", "hybrid_fusion",
        "dense_weight", "bm25_weight",
        "candidate_k", "max_retrieval_variants", "adaptive_retrieval_enabled",
        "adaptive_narrow_factor", "adaptive_multi_factor",
        "adaptive_min_candidates", "retrieval_memory_enabled",
        "memory_seed_urls",
    ]
    parts = [f"{k}={cfg.get(k)}" for k in keys]
    # Index identity: dataset hash + embedding model + dimension.
    try:
        from ..vectorstore.store import VectorStore

        manifest = VectorStore().load_manifest()
        if manifest:
            parts.append(f"dataset={manifest.get('dataset_hash', '')}")
            parts.append(f"build={manifest.get('build_timestamp', '')}")
    except Exception:  # noqa: BLE001 - signature must never crash routing
        pass
    return "|".join(parts)


class CacheRegistry:
    """Shared caches plus an index/config signature for invalidation."""

    def __init__(self) -> None:
        cfg = get_config()
        self.embeddings = LRUCache(maxsize=cfg.get("cache_embedding_size", 512))
        self.retrieval = LRUCache(
            maxsize=cfg.get("cache_retrieval_size", 256),
            ttl=cfg.get("cache_retrieval_ttl", 3600),
        )
        self._signature = retrieval_signature()

    def current_signature(self) -> str:
        return retrieval_signature()

    def invalidated(self) -> bool:
        return retrieval_signature() != self._signature

    def refresh(self) -> None:
        self._signature = retrieval_signature()
        if self.invalidated():
            self.embeddings.clear()
            self.retrieval.clear()
            self._signature = retrieval_signature()


# Process-wide registry (singleton). Models/caches live once per process.
_registry: CacheRegistry | None = None
_registry_lock = threading.Lock()


def get_cache_registry() -> CacheRegistry:
    global _registry
    with _registry_lock:
        if _registry is None or _registry.invalidated():
            _registry = CacheRegistry()
        return _registry
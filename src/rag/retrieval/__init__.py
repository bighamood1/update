"""Retrieval: dense retriever and optional reranker.

The heavy model-backed classes are imported lazily so that importing any
submodule of ``rag.retrieval`` (e.g. ``query_normalizer``) never pulls in
sentence-transformers / scipy. ``from rag.retrieval import Retriever`` still
works unchanged via ``__getattr__``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - import-time hints only
    from .reranker import Reranker
    from .retriever import Retriever

__all__ = ["Retriever", "Reranker"]


def __getattr__(name: str) -> Any:
    if name == "Retriever":
        from .retriever import Retriever

        return Retriever
    if name == "Reranker":
        from .reranker import Reranker

        return Reranker
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

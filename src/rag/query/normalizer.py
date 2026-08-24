"""Query normalization (canonical package path).

Re-exports the implementation from :mod:`rag.retrieval.query_normalizer` so
callers can import from the spec-mandated location ``rag.query.normalizer``.
"""

from __future__ import annotations

from ..retrieval.query_normalizer import (  # noqa: F401
    expand_query,
    is_arabic,
    normalize_query,
)

__all__ = ["normalize_query", "expand_query", "is_arabic"]
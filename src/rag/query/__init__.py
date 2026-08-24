"""Query preprocessing sub-package (canonical paths per spec)."""

from __future__ import annotations

from .normalizer import expand_query, is_arabic, normalize_query  # noqa: F401

__all__ = ["normalize_query", "expand_query", "is_arabic"]
"""Robust streaming JSONL loader for the NMU dataset.

Streams records one line at a time, normalizes each into a typed
:class:`RawDocument`, and exposes helpers to separate text-bearing
records from gallery/image records.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from ..config import get_config
from ..schemas.documents import RawDocument
from ..utils.logging_utils import get_logger
from .text_filter import TextFilter

logger = get_logger(__name__)

# Content types that represent galleries / image collections.
# These records may legitimately have no meaningful text and are excluded
# from the text RAG index (while staying untouched in documents.jsonl).
_GALLERY_TYPES = {"gallery", "gallery_album", "image", "images"}

# Content types that are not useful for text retrieval even if they have text.
_EXCLUDED_TYPES = {"other"}


class JsonlLoader:
    """Stream records from a JSONL file into RawDocument objects."""

    def __init__(self, path: str | Path | None = None) -> None:
        cfg = get_config()
        self.path = Path(path) if path is not None else cfg["data_path"]
        self._filter = TextFilter()

    @property
    def text_filter(self) -> TextFilter:
        return self._filter

    def iter_raw(self) -> Iterator[RawDocument]:
        """Yield every record as a RawDocument, skipping blank lines."""
        if not self.path.exists():
            raise FileNotFoundError(
                f"Dataset not found: {self.path}. Expected data/documents.jsonl."
            )
        with self.path.open("r", encoding="utf-8") as fh:
            for line_no, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Malformed JSON at {self.path.name}:{line_no}: {exc}"
                    ) from exc
                raw = RawDocument.model_validate(obj)
                # Preserve any PHASE-1 fields the schema does not model
                # explicitly (never silently drop metadata).
                known = set(RawDocument.model_fields)
                raw.extra = {k: v for k, v in obj.items() if k not in known}
                yield raw

    def iter_textual(self) -> Iterator[RawDocument]:
        """Yield only records that should participate in text RAG.

        Deduplicates by document ID (the raw dataset contains exact-duplicate
        records from PHASE 1 scraping; documents.jsonl is not modified).
        """
        seen_ids: set[str] = set()
        for doc in self.iter_raw():
            if doc.id in seen_ids:
                logger.debug("Skipping duplicate record id=%s", doc.id)
                continue
            if self._is_textual(doc):
                seen_ids.add(doc.id)
                doc.text = self._filter.clean(doc.text)
                yield doc

    def iter_textual_deduplicated(self) -> Iterator[RawDocument]:
        """Alias for :meth:`iter_textual` (deduplicated by design)."""
        yield from self.iter_textual()

    @staticmethod
    def _is_textual(doc: RawDocument) -> bool:
        content_type = (doc.content_type or "").strip().lower()
        if content_type in _GALLERY_TYPES:
            return False
        if content_type in _EXCLUDED_TYPES:
            return False
        text = (doc.text or "").strip()
        return len(text) > 0

    def count_all(self) -> int:
        """Total number of records in the dataset."""
        return sum(1 for _ in self.iter_raw())

    def count_textual(self) -> int:
        """Number of text-bearing records usable for RAG."""
        return sum(1 for _ in self.iter_textual())

"""Shared fixtures for NMU RAG unit tests.

Test data is constructed inline (no model downloads, no Ollama).
"""

from __future__ import annotations

import json

import pytest

from rag.chunking.chunker import Chunker
from rag.schemas.documents import NormalizedDocument


@pytest.fixture
def chunker() -> Chunker:
    return Chunker(chunk_size=200, chunk_overlap=50)


def sample_document(
    content_type: str = "home",
    language: str = "en",
    text: str | None = None,
    url: str | None = None,
    **overrides,
) -> NormalizedDocument:
    data = {
        "id": "doc-001",
        "text": text or (
            "Welcome to New Mansoura University. "
            "The university is considered one of the generation of smart universities. "
            "It adopts various programs that take into account the labor market."
        ),
        "title": "New Mansoura University",
        "language": language,
        "content_type": content_type,
        "url": url or "https://nmu.edu.eg/en",
        "source_domain": "nmu.edu.eg",
        "section": None,
        "faculty": None,
        "program": None,
        "department": None,
        "academic_year": None,
        "published_at": None,
        "scraped_at": None,
        "content_hash": "abc123",
    }
    data.update(overrides)
    return NormalizedDocument(**data)


def jsonl_records(records: list[dict]) -> str:
    return "\n".join(json.dumps(r, ensure_ascii=False) for r in records)
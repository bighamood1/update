"""Shared helpers for index-building scripts (hash, manifest payload)."""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

from rag.config import get_config
from rag.schemas.documents import NormalizedDocument

cfg = get_config()


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Compute SHA-256 of a file (streamed, memory-safe)."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            block = fh.read(chunk_size)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def load_documents(loader) -> list[NormalizedDocument]:
    """Load and normalize textual documents, deduplicating by ID.

    Metadata enrichment (faculty / faculty_id from URL) is applied here so the
    enriched values flow into the chunk metadata at index time.
    """
    from rag.ingestion.metadata import enrich

    docs = []
    for raw in loader.iter_textual():
        doc = NormalizedDocument.model_validate(raw.model_dump())
        enrich(doc)
        docs.append(doc)
    return docs


def manifest_payload(
    dataset_path: Path,
    dataset_hash: str,
    doc_count: int,
    text_doc_count: int,
    chunk_count: int,
    embedding_model: str,
    embedding_dim: int,
    vector_db: str,
    collection: str,
) -> dict:
    """Build the standard index manifest payload."""
    return {
        "dataset_path": str(dataset_path),
        "dataset_hash": dataset_hash,
        "document_count": doc_count,
        "text_document_count": text_doc_count,
        "chunk_count": chunk_count,
        "embedding_model": embedding_model,
        "embedding_dimension": embedding_dim,
        "vector_database": vector_db,
        "chroma_collection": collection,
        "chunk_size": cfg["chunk_size"],
        "chunk_overlap": cfg["chunk_overlap"],
        # Versioning: bump when chunking/metadata/retrieval behavior changes.
        "chunking_version": 2,
        "metadata_schema_version": 2,
        "retrieval_version": 2,
        "build_timestamp": None,  # filled by VectorStore.write_manifest
        "python_version": platform.python_version(),
        "platform": sys.platform,
    }
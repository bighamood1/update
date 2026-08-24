"""Persistent ChromaDB-backed vector store.

Stores chunk embeddings, text, and metadata under ``vectorstore/``.
The database persists across executions; chunk IDs are stable so running
the indexer again does not duplicate entries.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import chromadb

from ..config import get_config
from ..schemas.documents import DocumentChunk, RetrievedChunk
from ..utils.logging_utils import get_logger

logger = get_logger(__name__)

_MANIFEST_NAME = "index_manifest.json"


class VectorStore:
    """Thin wrapper around a persistent Chroma collection."""

    def __init__(self, path: str | Path | None = None, collection_name: str | None = None) -> None:
        cfg = get_config()
        self.path = Path(path) if path is not None else cfg["vector_db_path"]
        self.collection_name = collection_name or cfg["chroma_collection"]
        self.path.mkdir(parents=True, exist_ok=True)

        self._client = chromadb.PersistentClient(path=str(self.path))
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    # -- collection helpers ------------------------------------------------

    @property
    def collection(self):
        return self._collection

    def count(self) -> int:
        return self._collection.count()

    def get_all(self) -> list[RetrievedChunk]:
        """Return every indexed chunk (used to build the BM25 lexical index)."""
        limit = self.count()
        if limit <= 0:
            return []
        result = self._collection.get(
            limit=limit,
            include=["documents", "metadatas"],
        )
        return self._to_retrieved_get(result)

    def upsert_chunks(self, chunks: list[DocumentChunk], embeddings) -> None:
        """Insert or update chunks with their precomputed embeddings."""
        if not chunks:
            return
        ids = [c.chunk_id for c in chunks]
        documents = [c.text for c in chunks]
        metadatas = [c.metadata for c in chunks]
        self._collection.upsert(
            ids=ids,
            embeddings=embeddings.tolist(),
            documents=documents,
            metadatas=metadatas,
        )
        logger.info("Upserted %d chunks into '%s' (total=%d)", len(chunks), self.collection_name, self.count())

    def query(
        self,
        query_embedding,
        top_k: int,
        where: dict | None = None,
    ) -> list[RetrievedChunk]:
        """Query the collection; returns RetrievedChunk list sorted by score desc."""
        top_k = max(1, top_k)
        result = self._collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        return self._to_retrieved(result)

    def get_by_document(self, document_id: str) -> list[RetrievedChunk]:
        """Return all chunks belonging to one document (ordered by chunk_index).

        Used for parent/section expansion: after a relevant chunk is found,
        the complete parent document section can be recovered.
        """
        result = self._collection.get(
            where={"document_id": document_id},
            include=["documents", "metadatas"],
        )
        chunks = self._to_retrieved_get(result)
        chunks.sort(key=lambda c: c.chunk_index or 0)
        return chunks

    def get_by_url(self, source_url: str) -> list[RetrievedChunk]:
        """Return all chunks indexed for one source page URL.

        Used to seed canonical directory pages (complete lists) into the
        candidate pool for list intents.
        """
        result = self._collection.get(
            where={"source_url": {"$eq": source_url}},
            include=["documents", "metadatas"],
        )
        chunks = self._to_retrieved_get(result)
        chunks.sort(key=lambda c: c.chunk_index or 0)
        return chunks

    def get_source_document_by_url(self, source_url: str) -> list[RetrievedChunk]:
        """Return full source-document text for a URL from the indexed dataset.

        This is a read-only hydration path used after Chroma has identified an
        authoritative source URL. It does not mutate the vector index; it simply
        restores table rows that may have been split or dropped by chunking so
        collection evidence can be assessed at source-document granularity.
        """
        if not source_url:
            return []
        data_path = get_config()["data_path"]
        if not data_path.exists():
            return []

        wanted = {source_url.rstrip("/"), _canonical_url(source_url)}
        hydrated: list[RetrievedChunk] = []
        try:
            with data_path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    try:
                        raw = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    url = (raw.get("url") or "").rstrip("/")
                    if url not in wanted and _canonical_url(url) not in wanted:
                        continue
                    text = raw.get("text") or ""
                    if not text.strip():
                        continue
                    doc_id = raw.get("id") or url
                    hydrated.append(
                        RetrievedChunk(
                            chunk_id=f"source-document:{doc_id}",
                            document_id=doc_id,
                            text=_clean_source_text(text),
                            score=1.0,
                            dense_score=None,
                            title=raw.get("title"),
                            source_url=url,
                            language=raw.get("language"),
                            content_type=raw.get("content_type"),
                            section=raw.get("section"),
                            faculty=raw.get("faculty"),
                            faculty_id=raw.get("faculty_id"),
                            program=raw.get("program"),
                            department=raw.get("department"),
                            academic_year=raw.get("academic_year"),
                            published_at=raw.get("published_at"),
                            document_hash=raw.get("content_hash"),
                            chunk_index=0,
                            chunk_count=1,
                        )
                    )
        except OSError:
            logger.exception("Could not hydrate source document for %s", source_url)
        return hydrated

    def get_by_section(self, document_id: str, section_id: str) -> list[RetrievedChunk]:
        """Return all chunks of a document belonging to one logical section."""
        result = self._collection.get(
            where={"$and": [{"document_id": document_id}, {"section_id": section_id}]},
            include=["documents", "metadatas"],
        )
        chunks = self._to_retrieved_get(result)
        chunks.sort(key=lambda c: c.chunk_index or 0)
        return chunks

    @staticmethod
    def _to_retrieved(result) -> list[RetrievedChunk]:
        chunks: list[RetrievedChunk] = []
        ids = (result.get("ids") or [[]])[0]
        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]

        for cid, text, meta, dist in zip(ids, documents, metadatas, distances):
            meta = meta or {}
            # Chroma returns distances; cosine similarity = 1 - distance (when using
            # cosine space). Chroma stores cosine distance in [0,2].
            score = 1.0 - float(dist)
            chunks.append(
                RetrievedChunk(
                    chunk_id=cid,
                    document_id=meta.get("document_id") or "",
                    text=text,
                    score=round(score, 4),
                    dense_score=round(score, 4),
                    title=meta.get("title"),
                    source_url=meta.get("source_url"),
                    language=meta.get("language"),
                    content_type=meta.get("content_type"),
                    section=meta.get("section"),
                    faculty=meta.get("faculty"),
                    faculty_id=meta.get("faculty_id"),
                    program=meta.get("program"),
                    department=meta.get("department"),
                    academic_year=meta.get("academic_year"),
                    published_at=meta.get("published_at"),
                    document_hash=meta.get("document_hash"),
                    section_id=meta.get("section_id"),
                    section_index=meta.get("section_index"),
                    chunk_index=meta.get("chunk_index"),
                    chunk_count=meta.get("chunk_count"),
                )
            )
        return chunks

    def _to_retrieved_get(self, result) -> list[RetrievedChunk]:
        """Map a ``get()`` result (no distances) into RetrievedChunk objects.

        Retrieved chunks fetched for expansion carry no similarity score, so
        both ``score`` and ``dense_score`` are set to 0.0 (they are re-ranked
        before being presented to the user when reranking is enabled).
        """
        chunks: list[RetrievedChunk] = []
        ids = result.get("ids") or []
        documents = result.get("documents") or []
        metadatas = result.get("metadatas") or []

        for cid, text, meta in zip(ids, documents, metadatas):
            meta = meta or {}
            chunks.append(
                RetrievedChunk(
                    chunk_id=cid,
                    document_id=meta.get("document_id") or "",
                    text=text,
                    score=0.0,
                    dense_score=0.0,
                    title=meta.get("title"),
                    source_url=meta.get("source_url"),
                    language=meta.get("language"),
                    content_type=meta.get("content_type"),
                    section=meta.get("section"),
                    faculty=meta.get("faculty"),
                    faculty_id=meta.get("faculty_id"),
                    program=meta.get("program"),
                    department=meta.get("department"),
                    academic_year=meta.get("academic_year"),
                    published_at=meta.get("published_at"),
                    document_hash=meta.get("document_hash"),
                    section_id=meta.get("section_id"),
                    section_index=meta.get("section_index"),
                    chunk_index=meta.get("chunk_index"),
                    chunk_count=meta.get("chunk_count"),
                )
            )
        return chunks

    def reset(self) -> None:
        """Delete all data in the collection (used by full rebuild)."""
        self._client.delete_collection(self.collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("Reset collection '%s'", self.collection_name)

    # -- manifest ------------------------------------------------------------

    def manifest_path(self) -> Path:
        return self.path / _MANIFEST_NAME

    def write_manifest(self, payload: dict) -> None:
        payload["build_timestamp"] = datetime.now(timezone.utc).isoformat()
        self.manifest_path().write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info("Wrote manifest to %s", self.manifest_path())

    def load_manifest(self) -> dict | None:
        if not self.manifest_path().exists():
            return None
        return json.loads(self.manifest_path().read_text(encoding="utf-8"))

    def kb_version(self) -> str:
        """Stable identifier of the CURRENT index content.

        Any change that makes cached answers potentially stale (dataset hash,
        build timestamp, embedding model, chunking schema) produces a new
        version, which automatically invalidates old semantic-cache / feedback
        entries. Never raises; falls back to a stable "no-manifest" value.
        """
        try:
            manifest = self.load_manifest() or {}
            identity = "|".join(
                str(manifest.get(k, ""))
                for k in (
                    "dataset_hash",
                    "build_timestamp",
                    "embedding_model",
                    "chunking_version",
                    "metadata_schema_version",
                    "retrieval_version",
                )
            )
        except Exception:  # pragma: no cover - defensive
            return "no-manifest"
        return hashlib.sha1(identity.encode("utf-8")).hexdigest()[:16] if identity else "no-manifest"

    def is_built(self) -> bool:
        return self.manifest_path().exists() and self.count() > 0

    def compatibility_errors(self) -> list[str]:
        """Return a list of fatal index/config mismatches (empty = compatible).

        Fields that are missing in an old manifest (pre-versioning) are
        treated as warnings, not errors, so an existing index keeps working.
        """
        manifest = self.load_manifest()
        if not manifest:
            return ["no manifest — index must be rebuilt"]
        cfg = get_config()
        errors: list[str] = []
        stored_model = manifest.get("embedding_model")
        if stored_model and stored_model != cfg["embedding_model"]:
            errors.append(
                f"INDEX OUT OF DATE: index built with '{stored_model}' but config "
                f"expects '{cfg['embedding_model']}'. Rebuild: python scripts/build_index.py"
            )
        return errors

    def compatibility_warnings(self) -> list[str]:
        manifest = self.load_manifest() or {}
        warnings: list[str] = []
        for field in ("chunking_version", "metadata_schema_version", "retrieval_version"):
            if field not in manifest:
                warnings.append(
                    f"index manifest predates '{field}' — rebuild recommended "
                    f"(python scripts/build_index.py)"
                )
        return warnings


def _canonical_url(url: str) -> str:
    u = (url or "").strip().rstrip("/").lower()
    u = u.replace("https://www.", "https://").replace("http://www.", "http://")
    return u


def _clean_source_text(text: str) -> str:
    """Trim site chrome while preserving table rows and headings."""
    lines = [line.strip() for line in (text or "").splitlines()]
    drop_exact = {
        "تابعنا على", "العربية", "English", "الرئيسية", "Home", "Students",
        "FOLLOW US", "FOLLOW US ON", "Quick Links", "Get in Touch",
        "روابط سريعة", "ابقى على تواصل", "تواصل معنا", "المساعد الذكي (NMU)",
    }
    kept: list[str] = []
    for line in lines:
        if not line or line in drop_exact:
            continue
        if "الزوار" in line or "Visitors" in line or "Subscribe to our newsletter" in line:
            continue
        if line.startswith("جميع الحقوق محفوظة") or line.startswith("All Rights Reserved"):
            continue
        kept.append(line)
    return "\n".join(kept)

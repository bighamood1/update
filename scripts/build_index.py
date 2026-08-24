"""Build the persistent vector index from data/documents.jsonl.

Pipeline:
    documents.jsonl -> validate -> load textual docs -> chunk -> embed
    -> store in Chroma -> write index manifest -> write indexing report

Stable chunk IDs mean repeated runs never duplicate entries.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tqdm import tqdm

from rag.chunking.chunker import Chunker
from rag.config import get_config
from rag.embeddings.embedder import Embedder
from rag.ingestion.loader import JsonlLoader
from rag.ingestion.validator import DatasetValidator
from rag.utils.logging_utils import setup_logging
from rag.vectorstore.store import VectorStore

from common import file_sha256, load_documents, manifest_payload

setup_logging()


def main() -> None:
    cfg = get_config()
    data_path = Path(cfg["data_path"])
    if not data_path.exists():
        print(f"[ERROR] Dataset not found: {data_path}")
        sys.exit(1)

    t_start = time.time()

    # --- validate ---
    print("Validating dataset ...")
    validator = DatasetValidator(data_path)
    vresult = validator.validate()
    if vresult.malformed_lines:
        print(f"[ERROR] {len(vresult.malformed_lines)} malformed lines; cannot build index.")
        sys.exit(1)
    print(f"  records={vresult.total_records}  errors={len(vresult.errors)}  warnings={len(vresult.warnings)}")

    # --- load ---
    print("Loading textual documents ...")
    loader = JsonlLoader(data_path)
    docs = load_documents(loader)
    print(f"  text documents (deduplicated): {len(docs)}")

    # --- chunk ---
    print("Chunking ...")
    chunker = Chunker()
    all_chunks = []
    for doc in tqdm(docs, desc="chunking", leave=False):
        all_chunks.extend(chunker.chunk_document(doc))
    print(f"  chunks: {len(all_chunks)}")

    # --- embed ---
    print(f"Embedding with {cfg['embedding_model']} on {cfg['embedding_device']} ...")
    embedder = Embedder()
    texts = [c.text for c in all_chunks]
    embeddings = embedder.embed_batch(texts)
    print(f"  embeddings: {embeddings.shape}")

    # --- store ---
    store = VectorStore()
    if cfg["force_rebuild"]:
        store.reset()
    store.upsert_chunks(all_chunks, embeddings)

    # --- manifest ---
    dataset_hash = file_sha256(data_path)
    payload = manifest_payload(
        dataset_path=data_path,
        dataset_hash=dataset_hash,
        doc_count=vresult.total_records,
        text_doc_count=len(docs),
        chunk_count=len(all_chunks),
        embedding_model=cfg["embedding_model"],
        embedding_dim=int(embeddings.shape[1]),
        vector_db="chromadb",
        collection=cfg["chroma_collection"],
    )
    store.write_manifest(payload)

    # --- report ---
    report = {
        **payload,
        "duration_seconds": round(time.time() - t_start, 1),
        "gallery_records_preserved": True,
        "dataset_untouched": True,
    }
    report_path = Path(cfg["vector_db_path"]) / "indexing_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print()
    print("=" * 56)
    print("INDEX BUILD COMPLETE")
    print("=" * 56)
    print(f"Records (raw)          : {vresult.total_records}")
    print(f"Text documents (dedup) : {len(docs)}")
    print(f"Chunks                 : {len(all_chunks)}")
    print(f"Embedding dim          : {embeddings.shape[1]}")
    print(f"Vector store           : {store.path}")
    print(f"Manifest               : {store.manifest_path()}")
    print(f"Duration               : {report['duration_seconds']}s")


if __name__ == "__main__":
    main()
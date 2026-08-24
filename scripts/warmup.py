"""Warm up the RAG stack so the first real question is fast.

Loads the vector index, embeddings, reranker and BM25 lexical index, then
sends one tiny generation request to Ollama to force the model into RAM and
report cold-start timings.

Usage::

    python scripts/warmup.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rag.embeddings.embedder import Embedder
from rag.retrieval.retriever import Retriever
from rag.utils.logging_utils import setup_logging
from rag.vectorstore.store import VectorStore

setup_logging()


def main() -> None:
    t0 = time.time()
    store = VectorStore()
    if not store.is_built():
        print("[ERROR] Index not built. Run: python scripts/build_index.py")
        sys.exit(1)
    print(f"Vector store ready ({store.count()} chunks) after {time.time()-t0:.1f}s")

    t1 = time.time()
    embedder = Embedder()
    print(f"Embedder ready after {time.time()-t1:.1f}s")

    t2 = time.time()
    retriever = Retriever(vectorstore=store, embedder=embedder)
    # Force the BM25 lexical index to build now, not on first question.
    _ = retriever._bm25_index()
    if retriever.reranker_enabled and retriever.reranker is not None:
        _ = retriever.reranker.model
    print(f"Retriever (BM25 + reranker) ready after {time.time()-t2:.1f}s")

    t3 = time.time()
    try:
        from rag.generation.ollama_client import OllamaClient
        from rag.generation.prompts import SYSTEM_PROMPT

        client = OllamaClient()
        client._verify_connection()
        client.generate(SYSTEM_PROMPT, "Say 'ready' only.")
        print(f"Ollama warm (model '{client.model}') after {time.time()-t3:.1f}s")
    except Exception as exc:  # noqa: BLE001 - warmup must not crash
        print(f"[WARN] Ollama warm-up failed: {exc}")

    print(f"\nWarm-up complete in {time.time()-t0:.1f}s total.")


if __name__ == "__main__":
    main()
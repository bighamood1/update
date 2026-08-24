"""Completely rebuild the vector database from data/documents.jsonl.

Use when: dataset changes, embedding model changes, chunking changes, or
vector DB schema changes. Does NOT scrape anything.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rag.config import get_config
from rag.utils.logging_utils import setup_logging
from rag.vectorstore.store import VectorStore

setup_logging()


def main() -> None:
    cfg = get_config()
    store = VectorStore()

    print(f"Resetting collection '{cfg['chroma_collection']}' at {store.path} ...")
    store.reset()

    # Drop stale manifest so build_index regenerates everything.
    manifest = store.manifest_path()
    if manifest.exists():
        manifest.unlink()
        print(f"Removed stale manifest: {manifest}")

    # Re-run the standard build with FORCE_REBUILD implied.
    cfg_dict = get_config()
    cfg_dict["force_rebuild"] = True

    from build_index import main as build_main

    build_main()


if __name__ == "__main__":
    main()
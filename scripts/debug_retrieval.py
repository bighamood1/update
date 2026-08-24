"""Debug the NMU retrieval/context pipeline without calling Ollama.

Prints the internal retrieval stages for one or more queries:

    python scripts/debug_retrieval.py
    python scripts/debug_retrieval.py -q "كم تبلغ رسوم كلية الطب؟"
    python scripts/debug_retrieval.py --full-text -q "What faculties does NMU have?"

The output answers three different questions:
- Was the correct evidence missing from ChromaDB?
- Was it retrieved and then removed?
- Did it reach the final context?
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")

from rag.context.builder import ContextBuilder
from rag.config import get_config
from rag.embeddings.embedder import Embedder
from rag.query.expansion import retrieval_variants
from rag.query.understanding import understand
from rag.retrieval.reranker import Reranker
from rag.retrieval.retriever import Retriever
from rag.utils.logging_utils import setup_logging
from rag.vectorstore.store import VectorStore


DEFAULT_QUERIES = [
    "ما هي كليات جامعة المنصورة الجديدة؟",
    "What faculties does NMU have?",
    "أين تقع جامعة المنصورة الجديدة؟",
    "كم تبلغ رسوم كلية الطب؟",
    "كم تبلغ رسوم الكليات السنوية؟",
    "ما شروط القبول؟",
    "ما نظام الدراسة؟",
    "ما شروط التحويل؟",
]


def _print_rows(title: str, rows: list[dict], *, limit: int, full_text: bool) -> None:
    print(f"\n{title} ({len(rows)})")
    print("-" * 88)
    for row in rows[:limit]:
        rank = row.get("rank") or "-"
        print(
            f"#{rank} score={row.get('score')} dense={row.get('dense_score')} "
            f"bm25={row.get('bm25_score')} rerank={row.get('rerank_score')} "
            f"type={row.get('type')} lang={row.get('language')} faculty={row.get('faculty')}"
        )
        print(f"    title: {row.get('title')}")
        print(f"    url:   {row.get('url')}")
        print(f"    id:    {row.get('chunk_id')}")
        print(f"    text:  {row.get('text_preview')}")
    if full_text and len(rows) > limit:
        print(f"    ... {len(rows) - limit} more")


def _print_removed(rows: list[dict], *, limit: int) -> None:
    print(f"\nREMOVED ({len(rows)})")
    print("-" * 88)
    for row in rows[:limit]:
        print(
            f"- reason={row.get('reason')} detail={row.get('detail')} "
            f"score={row.get('score')} type={row.get('type')} url={row.get('url')}"
        )
        print(f"  text: {row.get('text_preview')}")
    if len(rows) > limit:
        print(f"  ... {len(rows) - limit} more")


def debug_query(retriever: Retriever, question: str, *, limit: int, full_text: bool) -> None:
    understanding = understand(question)
    variants = retrieval_variants(understanding, question)
    print("\n" + "=" * 100)
    print(f"QUERY: {question}")
    print(
        f"UNDERSTANDING: lang={understanding.language} intent={understanding.intent} "
        f"category={understanding.category} faculty={understanding.faculty} "
        f"confidence={understanding.confidence} multi={understanding.is_multi_intent}"
    )
    print("VARIANTS:")
    for v in variants:
        print(f"  - {v}")

    chunks = retriever.retrieve(
        question,
        intent=understanding.intent,
        route=understanding.route,
        query_variants=variants,
    )
    trace = retriever.last_trace
    stages = trace.get("stages", {})
    for stage in (
        "dense", "bm25", "fused", "broad_fallback_merged", "candidate_cap",
        "deduped", "list_expanded", "fee_expanded", "threshold_filtered",
        "scholarship_expanded", "reranked", "source_diverse", "final",
    ):
        rows = stages.get(stage)
        if rows is not None:
            _print_rows(stage.upper(), rows, limit=limit, full_text=full_text)
    _print_removed(trace.get("removed") or [], limit=limit)

    coverage = trace.get("coverage") or {}
    print("\nCOVERAGE")
    print("-" * 88)
    print(coverage)

    cfg = get_config()
    max_chunks = int((cfg.get("intent_context_chunks") or {}).get(
        (understanding.intent or "FACT").lower(), cfg.get("final_context_chunks", 4)
    ))
    max_chars = None
    if understanding.intent == "TUITION" and any(
        marker in question.lower()
        for marker in ("all", "faculties", "annual", "سنوي", "السنوية", "الكليات", "جميع")
    ):
        max_chunks = max(max_chunks, int((cfg.get("intent_context_chunks") or {}).get("list", 6)))
        max_chars = int(cfg.get("context_max_chars_list", 8000))
    context = ContextBuilder().build(chunks, max_chunks=max_chunks, max_chars=max_chars)
    print("\nFINAL CONTEXT")
    print("-" * 88)
    print(f"chunks={len(chunks)} chars={len(context)}")
    print(context if full_text else context[:3000])


def main() -> None:
    parser = argparse.ArgumentParser(description="Debug NMU retrieval stages")
    parser.add_argument("-q", "--query", action="append", help="Query to inspect")
    parser.add_argument("--limit", type=int, default=8, help="Rows per stage")
    parser.add_argument("--full-text", action="store_true", help="Print full final context")
    parser.add_argument(
        "--no-reranker",
        action="store_true",
        help="Disable reranker for faster local debugging",
    )
    args = parser.parse_args()

    setup_logging()
    cfg = get_config()
    store = VectorStore()
    if not store.is_built():
        raise SystemExit("Vector index is not built. Run: python scripts/build_index.py")
    print(f"collection={store.collection_name} count={store.count()} kb={store.kb_version()}")
    print(f"embedding_model={cfg['embedding_model']} collection_path={store.path}")

    embedder = Embedder()
    reranker = None if args.no_reranker else Reranker()
    retriever = Retriever(vectorstore=store, embedder=embedder, reranker=reranker)
    for query in args.query or DEFAULT_QUERIES:
        debug_query(retriever, query, limit=max(1, args.limit), full_text=args.full_text)


if __name__ == "__main__":
    main()

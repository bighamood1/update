"""Retrieval benchmark (Phase 24): per-scenario retrieval latency + quality.

Runs retrieval-only (no Ollama) over the benchmark question set (EN/AR/mixed +
a no-answer-in-KB query) and reports P50 / P90 / P95 latency, recall@5,
recall@8, MRR, source-hit rate and context size per scenario.

Usage::

    python evaluation/benchmark/benchmark_retrieval.py [--questions benchmark_questions.jsonl]
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from rag.embeddings.embedder import Embedder
from rag.pipeline.rag import ContextBuilder
from rag.retrieval.retriever import Retriever
from rag.routing.router import QueryRouter
from rag.utils.logging_utils import setup_logging
from rag.vectorstore.store import VectorStore

setup_logging()

QUESTIONS = ROOT / "evaluation" / "benchmark_questions.jsonl"
RESULTS = ROOT / "evaluation" / "results"


def load_questions(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def pct(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    return float(statistics.quantiles(sorted(values), n=100, method="inclusive")[q - 1])


def normalize_url(url: str) -> str:
    return (url or "").replace("https://www.", "https://").replace("http://www.", "http://")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", type=Path, default=QUESTIONS)
    ap.add_argument("--reranker", action="store_true",
                    help="enable the cross-encoder reranker (slow on CPU)")
    args = ap.parse_args()

    questions = load_questions(args.questions)
    store = VectorStore()
    embedder = Embedder()
    router = QueryRouter()
    builder = ContextBuilder()

    from rag.retrieval.reranker import Reranker

    reranker = Reranker() if args.reranker else None
    retriever = Retriever(vectorstore=store, embedder=embedder, reranker=reranker)

    rows: list[dict] = []
    latencies: list[float] = []
    for q in questions:
        qid = q["id"]
        start = time.perf_counter()
        route = router.route(q["question"])
        chunks = retriever.retrieve(q["question"], intent=route.intent, route=route)
        elapsed = round(time.perf_counter() - start, 3)
        latencies.append(elapsed)

        context = builder.build(chunks)
        sources = builder.sources(chunks)
        gold_urls = [normalize_url(u) for u in q.get("expected_sources") or []]
        retrieved_urls = [normalize_url(c.source_url or "") for c in chunks]
        retrieved_set = set(retrieved_urls)
        gold_hits = [u for u in retrieved_set if u in set(gold_urls)]

        topic_hits = [t for t in q.get("expected_topics") or []
                      if t.lower() in (context or "").lower()]
        recall5 = len(gold_hits) / len(gold_urls) if gold_urls else 0.0
        mrr = 0.0
        for i, u in enumerate(retrieved_urls[:8], start=1):
            if u in set(gold_urls):
                mrr = 1.0 / i
                break
        rows.append({
            "id": qid,
            "question": q["question"],
            "lang": q["lang"],
            "category": q["category"],
            "route_intent": route.intent,
            "route_confidence": route.confidence,
            "retrieval_time_s": elapsed,
            "candidate_count": retriever.last_meta.get("candidate_count"),
            "final_count": len(chunks),
            "top_k_used": retriever.last_meta.get("top_k_used"),
            "routed": retriever.last_meta.get("routed", False),
            "fallback_used": retriever.last_meta.get("fallback_used", False),
            "cache_hit": retriever.last_meta.get("cache_hit", False),
            "context_chars": len(context),
            "n_sources": len(sources),
            "topic_hit": bool(topic_hits),
            "recall@5": recall5 if gold_urls else None,
            "mrr": mrr if gold_urls else None,
            "source_hit": bool(gold_hits),
            "has_gold": bool(gold_urls),
            "expected_sources": gold_urls,
        })

    scoped = [r for r in rows if r["has_gold"]]
    report = {
        "mode": "retrieval",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_questions": len(rows),
        "latency_s": {
            "mean": round(statistics.mean(latencies), 3),
            "p50": round(pct(latencies, 50), 3),
            "p90": round(pct(latencies, 90), 3),
            "p95": round(pct(latencies, 95), 3),
        },
        "quality": {
            "topic_hit_rate": round(
                sum(1 for r in rows if r["topic_hit"]) / len(rows), 4),
            "source_hit_rate": round(
                sum(1 for r in scoped if r["source_hit"]) / len(scoped), 4)
            if scoped else None,
            "mean_recall@5": round(
                statistics.mean([r["recall@5"] for r in scoped]), 4)
            if scoped else None,
            "mean_mrr": round(
                statistics.mean([r["mrr"] for r in scoped]), 4)
            if scoped else None,
            "mean_context_chars": round(
                statistics.mean([r["context_chars"] for r in rows]), 1),
        },
        "results": rows,
    }
    out = RESULTS / f"benchmark_retrieval_{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
    RESULTS.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Wrote", out)
    print(json.dumps(report["latency_s"], indent=2))
    print(json.dumps(report["quality"], indent=2))
    for r in rows:
        print("%s | %-30s | t=%.2fs | ctx=%d | routed=%s | fail=%s" % (
            r["id"], r["question"][:30], r["retrieval_time_s"],
            r["context_chars"], r["routed"], r["fallback_used"]))


if __name__ == "__main__":
    main()
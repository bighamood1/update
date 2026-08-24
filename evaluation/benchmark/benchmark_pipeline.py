"""Pipeline benchmark (Phase 24): end-to-end RAG latency + answer quality.

Runs full RAG (retrieval + Ollama generation) over the benchmark question set
and reports P50 / P90 / P95 for retrieval, generation and total latency, plus
answer rate, fabricated-URL (hallucination) rate and source counts.

The LLM calls make this slow (CPU + qwen3-vl:8b); use ``--limit`` for a quick
smoke run. Warm up first with ``python scripts/warmup.py``.

Usage::

    python evaluation/benchmark/benchmark_pipeline.py [--limit 5]
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

from rag.pipeline.rag import RAGPipeline
from rag.utils.logging_utils import setup_logging

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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", type=Path, default=QUESTIONS)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    questions = load_questions(args.questions)
    if args.limit:
        questions = questions[: args.limit]

    pipe = RAGPipeline()

    rows: list[dict] = []
    retr_times: list[float] = []
    gen_times: list[float] = []
    total_times: list[float] = []
    for q in questions:
        qid = q["id"]
        t_start = time.perf_counter()
        try:
            result = pipe.ask(q["question"])
        except Exception as exc:  # noqa: BLE001
            rows.append({"id": qid, "question": q["question"], "error": str(exc)})
            continue
        elapsed = round(time.perf_counter() - t_start, 3)
        total_times.append(elapsed)
        retr_times.append(float(result.timings.get("retrieval_time", 0.0)))
        gen_times.append(float(result.timings.get("ollama_request_time", 0.0)))

        import re as _re

        urls = _re.findall(r"https?://[^\s)\]}>'\"]+", result.answer)
        fabricated = [
            u for u in urls
            if u.rstrip("/").lower()
            not in {(c.source_url or "").rstrip("/").lower() for c in result.retrieved_chunks}
        ]
        topic_hits = [t for t in q.get("expected_topics") or []
                      if t.lower() in result.answer.lower()]
        rows.append({
            "id": qid,
            "question": q["question"],
            "lang": q["lang"],
            "category": q["category"],
            "retrieval_time_s": result.timings.get("retrieval_time"),
            "ollama_time_s": result.timings.get("ollama_request_time"),
            "total_time_s": elapsed,
            "answer_chars": len(result.answer),
            "n_sources": len(result.sources),
            "topic_hit": bool(topic_hits),
            "fabricated_urls": fabricated,
            "intent": result.intent,
            "diagnostics": result.diagnostics,
        })

    scoped = [r for r in rows if "error" not in r]
    report = {
        "mode": "pipeline",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_questions": len(rows),
        "latency_s": {
            "retrieval": {"mean": round(statistics.mean(retr_times), 3),
                          "p50": round(pct(retr_times, 50), 3),
                          "p90": round(pct(retr_times, 90), 3),
                          "p95": round(pct(retr_times, 95), 3)},
            "generation": {"mean": round(statistics.mean(gen_times), 3),
                           "p50": round(pct(gen_times, 50), 3),
                           "p90": round(pct(gen_times, 90), 3),
                           "p95": round(pct(gen_times, 95), 3)},
            "total": {"mean": round(statistics.mean(total_times), 3),
                      "p50": round(pct(total_times, 50), 3),
                      "p90": round(pct(total_times, 90), 3),
                      "p95": round(pct(total_times, 95), 3)},
        },
        "quality": {
            "answer_rate": round(
                sum(1 for r in scoped if r["answer_chars"] > 0) / len(scoped), 4)
            if scoped else None,
            "topic_hit_rate": round(
                sum(1 for r in scoped if r["topic_hit"]) / len(scoped), 4)
            if scoped else None,
            "hallucination_rate": round(
                sum(1 for r in scoped if r["fabricated_urls"]) / len(scoped), 4)
            if scoped else None,
            "mean_sources": round(
                statistics.mean([r["n_sources"] for r in scoped]), 2)
            if scoped else None,
        },
        "results": rows,
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / f"benchmark_pipeline_{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Wrote", out)
    print(json.dumps(report["latency_s"], indent=2))
    print(json.dumps(report["quality"], indent=2))
    for r in rows:
        print("%s | %-30s | gen=%.1fs | total=%.1fs | src=%d | hallu=%s" % (
            r["id"], r["question"][:30],
            r.get("ollama_time_s") or 0.0, r.get("total_time_s") or 0.0,
            r.get("n_sources") or 0, bool(r.get("fabricated_urls"))))


if __name__ == "__main__":
    main()
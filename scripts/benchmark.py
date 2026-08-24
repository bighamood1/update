"""Benchmark retrieval and/or full RAG answers over an AR/EN question set.

Unlike ``evaluation/evaluate.py`` (which reports the aggregate pipeline), this
benchmark is designed for comparing optimisation runs: it records per-question
retrieval + generation timings, exact context size, and source counts, and can
diff against a previous baseline report.

Usage::

    python scripts/benchmark.py --mode retrieval            # no Ollama
    python scripts/benchmark.py --mode full                 # calls Ollama
    python scripts/benchmark.py --mode both --baseline <previous-report.json>
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rag.config import get_config
from rag.embeddings.embedder import Embedder
from rag.pipeline.rag import ContextBuilder, RAGPipeline
from rag.retrieval.reranker import Reranker
from rag.retrieval.retriever import Retriever
from rag.utils.logging_utils import setup_logging
from rag.vectorstore.store import VectorStore

setup_logging()

DEFAULT_QUESTIONS = ROOT / "evaluation" / "benchmark_questions.jsonl"


def load_questions(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def token_hits(text: str, tokens: list[str]) -> list[str]:
    low = (text or "").lower()
    return [t for t in tokens if t.lower() in low]


def baseline_report(args) -> dict | None:
    if not args.baseline:
        return None
    with Path(args.baseline).open("r", encoding="utf-8") as fh:
        return json.load(fh)


def compare_row(row: dict, base: dict) -> dict:
    b = base.get(row["id"]) or {}
    for key in ("retrieval_time", "ollama_time", "total_time", "context_chars"):
        if key in b and key in row and b[key]:
            delta = (row[key] - b[key]) / b[key]
            row[f"{key}_delta_pct"] = round(delta * 100, 1)
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark the NMU RAG system")
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--mode", choices=["retrieval", "full", "both"], default="both")
    parser.add_argument("--baseline", type=Path, help="Previous report JSON to diff")
    parser.add_argument("--out", type=Path, default=ROOT / "evaluation" / "results")
    args = parser.parse_args()

    questions = load_questions(args.questions)
    if len(questions) < 10:
        print(f"[WARN] Only {len(questions)} questions (< 10 recommended).")
    cfg = get_config()
    base = baseline_report(args)
    base_map = {r["id"]: r for r in base.get("rows", [])} if base else {}

    store = VectorStore()
    if not store.is_built():
        sys.exit("[ERROR] Index not built. Run: python scripts/build_index.py")

    embedder = Embedder()
    reranker = Reranker() if cfg["reranker_enabled"] else None
    retriever = Retriever(vectorstore=store, embedder=embedder, reranker=reranker)
    builder = ContextBuilder()
    pipeline = None
    if args.mode in ("full", "both"):
        pipeline = RAGPipeline(vectorstore=store, embedder=embedder, retriever=retriever)

    rows = []
    for q in questions:
        row = {
            "id": q["id"],
            "question": q["question"],
            "lang": q.get("lang", ""),
            "category": q.get("category", ""),
        }
        start = time.time()
        chunks = retriever.retrieve(q["question"], intent=q.get("intent"))
        row["retrieval_time"] = round(time.time() - start, 3)
        row["n_retrieved"] = len(chunks)
        hits = []
        for c in chunks:
            hits.extend(token_hits(c.text, q.get("expected_topics", [])))
        row["topic_hits"] = sorted(set(hits))
        row["top1_hit"] = bool(
            token_hits(chunks[0].text, q.get("expected_topics", [])) if chunks else []
        )
        row["context_chars"] = len(builder.build(chunks))

        if pipeline is not None:
            start = time.time()
            try:
                result = pipeline.ask(q["question"], skip_cache=True)
                elapsed = time.time() - start
                row["answer"] = result.answer
                row["ok"] = bool(result.answer.strip()) and bool(result.sources)
                row["n_sources"] = len(result.sources)
                row["total_time"] = round(elapsed, 3)
                row["ollama_time"] = result.timings.get("ollama_request_time", 0.0)
                row["retrieval_time"] = result.timings.get("retrieval_time", row["retrieval_time"])
            except Exception as exc:  # noqa: BLE001 - keep the run going
                row["ok"] = False
                row["error"] = str(exc)
                row["total_time"] = round(time.time() - start, 3)
                row["ollama_time"] = 0.0
        rows.append(compare_row(row, base_map))

    # -- summary -------------------------------------------------------------
    retrieval_times = [r["retrieval_time"] for r in rows]
    summary = {
        "mode": args.mode,
        "n_questions": len(rows),
        "n_en": sum(1 for r in rows if r["lang"] == "en"),
        "n_ar": sum(1 for r in rows if r["lang"] == "ar"),
        "n_mixed": sum(1 for r in rows if r["lang"] == "mixed"),
        "config": {
            "rerank_candidates": cfg["rerank_candidates"],
            "top_k": cfg["top_k"],
            "final_context_chunks": cfg["final_context_chunks"],
            "context_max_chars": cfg["context_max_chars"],
            "hybrid_enabled": cfg["hybrid_enabled"],
            "query_expansion_enabled": cfg["query_expansion_enabled"],
            "ollama_max_output_tokens": cfg["ollama_max_output_tokens"],
            "reranker_enabled": cfg["reranker_enabled"],
        },
        "avg_retrieval_time_s": round(sum(retrieval_times) / len(retrieval_times), 3),
        "topic_hit_rate": round(
            sum(1 for r in rows if r["topic_hits"]) / len(rows), 4
        ),
        "top1_hit_rate": round(sum(1 for r in rows if r["top1_hit"]) / len(rows), 4),
        "avg_n_retrieved": round(sum(r["n_retrieved"] for r in rows) / len(rows), 2),
        "avg_context_chars": round(
            sum(r["context_chars"] for r in rows) / len(rows), 1
        ),
    }
    if pipeline is not None:
        full = [r for r in rows if "total_time" in r]
        summary.update(
            {
                "answer_rate": round(sum(1 for r in full if r["ok"]) / len(full), 4),
                "avg_total_time_s": round(sum(r["total_time"] for r in full) / len(full), 3),
                "avg_ollama_time_s": round(sum(r["ollama_time"] for r in full) / len(full), 3),
                "avg_n_sources": round(sum(r["n_sources"] for r in full) / len(full), 2),
            }
        )

    if base:
        for key in ("avg_retrieval_time_s", "avg_total_time_s", "avg_ollama_time_s",
                    "topic_hit_rate", "answer_rate"):
            if key in summary and key in base:
                b = base[key]
                if b:
                    summary[f"{key}_delta_pct"] = round((summary[key] - b) / b * 100, 1)

    report = {"summary": summary, "rows": rows}
    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    json_path = out_dir / f"benchmark_{stamp}.json"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = ["# NMU RAG Benchmark", "", f"- Mode: `{args.mode}`", f"- Timestamp (UTC): {stamp}", ""]
    for k, v in summary.items():
        lines.append(f"- {k}: {v}")
    lines += ["", "## Per-question", "", "| id | lang | hits | top1 | n_retr | ctx_chars | retr_s | total_s | ok |", "|---|------|------|------|--------|-----------|--------|---------|----|"]
    for r in rows:
        lines.append(
            f"| {r['id']} | {r['lang']} | {len(r['topic_hits'])} | "
            f"{'Y' if r['top1_hit'] else 'N'} | {r['n_retrieved']} | {r['context_chars']} | "
            f"{r['retrieval_time']:.2f} | {r.get('total_time', 0):.1f} | "
            f"{'PASS' if r.get('ok', False) else '-'} |"
        )
    md_path = out_dir / f"benchmark_{stamp}.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"\nReport written:\n  {json_path}\n  {md_path}")
    for k, v in summary.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
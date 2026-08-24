"""Evaluate retrieval (and optionally full RAG answers) against a question set.

Usage::

    python evaluation/evaluate.py                      # retrieval-only (fast)
    python evaluation/evaluate.py --mode full          # full RAG answers (slow, calls Ollama)
    python evaluation/evaluate.py --limit 5            # first N questions
    python evaluation/evaluate.py --questions questions.jsonl

Outputs JSON + Markdown reports into ``evaluation/results/``.

Retrieval metrics:
- hit_rate / top1_hit_rate: topic-token overlap in the final set / top-1.
- recall@5, recall@8, precision@5, precision@8, mrr, source_hit_rate:
  URL-level metrics computed against each question's ``expected_sources``
  (normalized, ``www.``-agnostic).
- duplicate_rate: fraction of final chunks whose text is a verbatim repeat of
  an earlier chunk (context redundancy).
- average_latency_s: mean end-to-end retrieval time (embedding + search +
  rerank).

Out-of-scope questions (empty ``expected_topics``) are excluded from scoped
metrics and reported separately: a good system retrieves nothing / refuses.
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
from rag.pipeline.rag import RAGPipeline
from rag.retrieval.intents import classify_intent
from rag.retrieval.reranker import Reranker
from rag.retrieval.retriever import Retriever
from rag.utils.logging_utils import setup_logging
from rag.vectorstore.store import VectorStore

setup_logging()


def load_questions(path: Path) -> list[dict]:
    questions = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            questions.append(json.loads(line))
    return questions


def token_hits(text: str, tokens: list[str]) -> list[str]:
    """Return the expected tokens found (case-insensitive substring match)."""
    if not tokens:
        return []
    low = text.lower()
    return [t for t in tokens if t.lower() in low]


def normalize_url(url: str) -> str:
    """Strip protocol/host variants so www vs non-www pages match."""
    return (url or "").replace("https://www.", "https://").replace("http://www.", "http://")


def url_metrics(
    retrieved: list, gold: list[str], k_values: tuple[int, ...] = (5, 8)
) -> dict:
    """Compute Recall@K / Precision@K / MRR / source hit rate over URLs."""
    gold_norm = {normalize_url(u) for u in gold}
    ordered = []
    seen = set()
    for c in retrieved:
        u = normalize_url(c.source_url or "")
        if u and u not in seen:
            seen.add(u)
            ordered.append(u)

    out = {}
    for k in k_values:
        prefix = ordered[:k]
        hits = len([u for u in prefix if u in gold_norm])
        out[f"recall@{k}"] = round(hits / len(gold_norm), 4) if gold_norm else 0.0
        out[f"precision@{k}"] = (
            round(hits / len(prefix), 4) if prefix and gold_norm else 0.0
        )

    rank = next((i + 1 for i, u in enumerate(ordered) if u in gold_norm), 0)
    out["mrr"] = round(1.0 / rank, 4) if rank else 0.0
    out["source_hit_rate"] = 1.0 if (gold_norm & set(ordered)) else 0.0
    out["_has_gold"] = bool(gold_norm)
    return out


def duplicate_rate(chunks) -> float:
    """Fraction of chunks that repeat the text of an earlier chunk."""
    if not chunks:
        return 0.0
    seen = set()
    dupes = 0
    for c in chunks:
        sig = " ".join((c.text or "").split())[:200]
        if sig in seen:
            dupes += 1
        else:
            seen.add(sig)
    return round(dupes / len(chunks), 4)


def evaluate_retrieval(retriever: Retriever, questions: list[dict]) -> dict:
    results = []
    latencies = []
    for q in questions:
        qid = q["id"]
        expected = q.get("expected_topics", []) or []
        gold = q.get("expected_sources", []) or []
        intent = classify_intent(q["question"])

        t0 = time.time()
        chunks = retriever.retrieve(q["question"], intent=intent)
        elapsed = time.time() - t0
        latencies.append(elapsed)
        top1 = chunks[0] if chunks else None

        hits = []
        for c in chunks:
            hits.extend(token_hits(c.text, expected))
        top1_hits = token_hits(top1.text, expected) if top1 else []

        out_of_scope = not expected
        um = url_metrics(chunks, gold) if not out_of_scope else {}
        record = {
            "id": qid,
            "question": q["question"],
            "lang": q.get("lang", ""),
            "category": q.get("category", ""),
            "intent": intent,
            "out_of_scope": out_of_scope,
            "num_results": len(chunks),
            "top1_score": round(top1.score, 4) if top1 else None,
            "top1_title": top1.title if top1 else None,
            "top1_hits": top1_hits,
            "total_hits": hits,
            "duplicate_rate": duplicate_rate(chunks),
            "hit": bool(hits) or (out_of_scope and not chunks),
            "pass": bool(hits) if not out_of_scope else (not chunks),
            **um,
        }
        results.append(record)

    scoped = [r for r in results if not r["out_of_scope"]]
    oos = [r for r in results if r["out_of_scope"]]
    hit_rate = sum(1 for r in scoped if r["hit"]) / len(scoped) if scoped else 0.0
    top1_hit_rate = (
        sum(1 for r in scoped if r["top1_hits"]) / len(scoped) if scoped else 0.0
    )
    scores = [r["top1_score"] for r in scoped if r["top1_score"] is not None]
    mean_score = sum(scores) / len(scores) if scores else 0.0
    oos_ok = sum(1 for r in oos if r["pass"]) / len(oos) if oos else 0.0
    with_gold = [r for r in scoped if r.get("_has_gold")]

    def _mean(records, key):
        vals = [r[key] for r in records if r.get(key) is not None]
        return round(sum(vals) / len(vals), 4) if vals else 0.0

    return {
        "mode": "retrieval",
        "n_questions": len(questions),
        "n_scoped": len(scoped),
        "n_out_of_scope": len(oos),
        "hit_rate": round(hit_rate, 4),
        "top1_hit_rate": round(top1_hit_rate, 4),
        "mean_top1_score": round(mean_score, 4),
        "recall@5": _mean(with_gold, "recall@5"),
        "recall@8": _mean(with_gold, "recall@8"),
        "precision@5": _mean(with_gold, "precision@5"),
        "precision@8": _mean(with_gold, "precision@8"),
        "mrr": _mean(with_gold, "mrr"),
        "source_hit_rate": _mean(with_gold, "source_hit_rate"),
        "mean_duplicate_rate": _mean(scoped, "duplicate_rate"),
        "average_latency_s": round(sum(latencies) / len(latencies), 3) if latencies else 0.0,
        "out_of_scope_refusal_rate": round(oos_ok, 4),
        "results": results,
    }


def evaluate_full(pipeline: RAGPipeline, questions: list[dict]) -> dict:
    results = []
    for q in questions:
        qid = q["id"]
        expected = q.get("expected_topics", []) or []
        out_of_scope = not expected
        start = time.time()
        try:
            result = pipeline.ask(q["question"])
        except Exception as exc:  # noqa: BLE001 - keep the run going
            elapsed = time.time() - start
            print(f"[{qid}] ERROR ({elapsed:.0f}s) {q['question'][:50]}: {exc}")
            results.append(
                {
                    "id": qid,
                    "question": q["question"],
                    "lang": q.get("lang", ""),
                    "category": q.get("category", ""),
                    "out_of_scope": out_of_scope,
                    "elapsed_seconds": round(elapsed, 1),
                    "answer": "",
                    "hits": [],
                    "has_sources": False,
                    "refused": False,
                    "n_sources": 0,
                    "error": str(exc),
                    "pass": False,
                }
            )
            continue
        elapsed = time.time() - start

        hits = token_hits(result.answer, expected)
        has_sources = bool(result.sources)
        refused = (
            "not contain enough information" in result.answer
            or "does not contain any information" in result.answer
            or "not contain any information" in result.answer
            or "لا يحتوي" in result.answer
            or "لا يتوفر" in result.answer
        )
        pass_ = (bool(hits) and has_sources) if not out_of_scope else refused
        results.append(
            {
                "id": qid,
                "question": q["question"],
                "lang": q.get("lang", ""),
                "category": q.get("category", ""),
                "out_of_scope": out_of_scope,
                "elapsed_seconds": round(elapsed, 1),
                "answer": result.answer,
                "hits": hits,
                "has_sources": has_sources,
                "refused": refused,
                "n_sources": len(result.sources),
                "intent": result.intent,
                "timings": result.timings,
                "pass": pass_,
            }
        )
        print(
            f"[{qid}] {('PASS' if pass_ else 'FAIL')} "
            f"({elapsed:.0f}s) {q['question'][:50]}"
        )

    scoped = [r for r in results if not r["out_of_scope"]]
    oos = [r for r in results if r["out_of_scope"]]
    answer_rate = sum(1 for r in scoped if r["pass"]) / len(scoped) if scoped else 0.0
    source_rate = (
        sum(1 for r in scoped if r["has_sources"]) / len(scoped) if scoped else 0.0
    )
    oos_ok = sum(1 for r in oos if r["pass"]) / len(oos) if oos else 0.0
    total_time = sum(r["elapsed_seconds"] for r in results)

    timings = [r.get("timings") or {} for r in results if r.get("timings")]
    avg_retrieval = (
        sum(t.get("retrieval_time", 0.0) for t in timings) / len(timings)
        if timings
        else 0.0
    )
    avg_embed = (
        sum(t.get("embedding_time", 0.0) for t in timings) / len(timings)
        if timings
        else 0.0
    )
    avg_rerank = (
        sum(t.get("reranking_time", 0.0) for t in timings) / len(timings)
        if timings
        else 0.0
    )
    avg_ollama = (
        sum(t.get("ollama_request_time", 0.0) for t in timings) / len(timings)
        if timings
        else 0.0
    )

    return {
        "mode": "full",
        "n_questions": len(questions),
        "n_scoped": len(scoped),
        "n_out_of_scope": len(oos),
        "answer_rate": round(answer_rate, 4),
        "source_citation_rate": round(source_rate, 4),
        "out_of_scope_refusal_rate": round(oos_ok, 4),
        "total_time_seconds": round(total_time, 1),
        "mean_time_seconds": round(total_time / len(results), 1) if results else 0.0,
        "mean_embedding_time_s": round(avg_embed, 3),
        "mean_retrieval_time_s": round(avg_retrieval, 3),
        "mean_reranking_time_s": round(avg_rerank, 3),
        "mean_ollama_time_s": round(avg_ollama, 3),
        "results": results,
    }


def write_report(report: dict, out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    json_path = out_dir / f"report_{stamp}.json"
    md_path = out_dir / f"report_{stamp}.md"

    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    cfg = get_config()
    lines = [
        "# NMU RAG Evaluation Report",
        "",
        f"- Mode: `{report['mode']}`",
        f"- Timestamp (UTC): {stamp}",
        f"- Embedding model: `{cfg['embedding_model']}`",
        f"- TOP_K: {cfg['top_k']}  Candidate_K: {cfg['candidate_k']}  Threshold: {cfg['similarity_threshold']}",
        f"- Reranker: {'enabled' if cfg['reranker_enabled'] else 'disabled'} "
        f"(`{cfg['reranker_model']}`)",
        f"- Questions: {report['n_questions']} "
        f"(scoped {report['n_scoped']}, out-of-scope {report['n_out_of_scope']})",
        "",
        "## Metrics",
        "",
    ]
    for key, val in report.items():
        if isinstance(val, list):
            continue
        lines.append(f"- {key}: {val}")
    lines += ["", "## Per-question results", "", "| id | lang | cat | intent | pass | top1 | dup | hits |", "|---|------|-----|--------|------|------|-----|------|"]
    for r in report["results"]:
        top1 = r.get("top1_score")
        top1_str = f"{top1:.3f}" if top1 is not None else "-"
        hits = ", ".join(r.get("top1_hits") or r.get("hits") or []) or "-"
        dup = r.get("duplicate_rate")
        dup_str = f"{dup:.2f}" if dup is not None else "-"
        lines.append(
            f"| {r['id']} | {r.get('lang','')} | {r.get('category','')} "
            f"| {r.get('intent','-')} | {'PASS' if r.get('pass') else 'FAIL'} "
            f"| {top1_str} | {dup_str} | {hits} |"
        )
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path, md_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the NMU RAG system")
    parser.add_argument(
        "--questions",
        type=Path,
        default=ROOT / "evaluation" / "questions.jsonl",
    )
    parser.add_argument("--mode", choices=["retrieval", "full"], default="retrieval")
    parser.add_argument("--limit", type=int, default=0, help="Run only first N questions")
    parser.add_argument("--out", type=Path, default=ROOT / "evaluation" / "results")
    args = parser.parse_args()

    if not args.questions.exists():
        sys.exit(f"[ERROR] Questions file not found: {args.questions}")

    questions = load_questions(args.questions)
    if args.limit:
        questions = questions[: args.limit]
    print(f"Loaded {len(questions)} questions (mode={args.mode}).")

    store = VectorStore()
    if not store.is_built():
        sys.exit("[ERROR] Index not built. Run: python scripts/build_index.py")

    if args.mode == "retrieval":
        embedder = Embedder()
        reranker = Reranker() if get_config().get("reranker_enabled", True) else None
        retriever = Retriever(
            vectorstore=store,
            embedder=embedder,
            reranker=reranker,
        )
        report = evaluate_retrieval(retriever, questions)
    else:
        pipeline = RAGPipeline(vectorstore=store)
        report = evaluate_full(pipeline, questions)

    json_path, md_path = write_report(report, args.out)
    print(f"\nReport written:\n  {json_path}\n  {md_path}")

    metrics = {k: v for k, v in report.items() if not isinstance(v, list)}
    for k, v in metrics.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
"""Benchmark report (Phase 25): render a benchmark JSON into a Markdown report.

Optionally diffs against a baseline JSON (e.g. ``evaluation/results/baseline.json``)
and flags regressions.

Usage::

    python evaluation/benchmark/benchmark_report.py <benchmark.json> [--baseline baseline.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

REQUIRED_COMMANDS = (
    "recall@5", "recall@8", "mrr", "source_hit_rate", "hit_rate",
    "top1_hit_rate", "answer_rate", "hallucination_rate",
)


def fmt(v) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def render(report: dict, baseline: dict | None = None) -> str:
    lines: list[str] = []
    lines.append("# Benchmark Report")
    lines.append("")
    lines.append(f"- mode: `{report.get('mode')}`")
    lines.append(f"- generated_at: `{report.get('generated_at')}`")
    lines.append(f"- n_questions: {report.get('n_questions')}")
    same_set = (
        baseline is not None
        and report.get("n_questions") == baseline.get("n_questions")
    )
    if not same_set and baseline:
        lines.append(
            f"- NOTE: baseline has {baseline.get('n_questions')} questions — "
            "deltas below compare different question sets and are indicative only."
        )
    lines.append("")

    quality = report.get("quality") or {}
    base_quality = (baseline or {}).get("quality") or {}
    if quality:
        lines.append("## Quality")
        lines.append("")
        lines.append("| metric | after | baseline | delta |")
        lines.append("|---|---:|---:|---:|")
        for key, val in quality.items():
            b = base_quality.get(key)
            delta = ""
            if same_set and isinstance(val, (int, float)) and isinstance(b, (int, float)):
                d = val - b
                delta = f"{d:+.4f}"
                if key in ("hallucination_rate",) and d > 0.0:
                    delta += " (regression!)"
                elif key not in ("hallucination_rate",) and d < 0.0:
                    delta += " (regression!)"
            lines.append(f"| {key} | {fmt(val)} | {fmt(b)} | {delta} |")
        lines.append("")

    latency = report.get("latency_s") or {}
    base_lat = (baseline or {}).get("latency_s") or {}
    if isinstance(latency, dict) and "total" in latency:
        for group in ("retrieval", "generation", "total"):
            g = latency.get(group)
            if not g:
                continue
            lines.append(f"### Latency — {group} (s)")
            lines.append("")
            lines.append("| stat | after | baseline | delta |")
            lines.append("|---|---:|---:|---:|")
            for key in ("mean", "p50", "p90", "p95"):
                b = (base_lat.get(group) or {}).get(key)
                d = ""
                if isinstance(g.get(key), (int, float)) and isinstance(b, (int, float)):
                    d = f"{g[key] - b:+.3f}"
                lines.append(f"| {key} | {fmt(g.get(key))} | {fmt(b)} | {d} |")
            lines.append("")
    elif isinstance(latency, dict) and any(
        k in latency for k in ("mean", "p50", "p90", "p95")
    ):
        lines.append("### Latency — retrieval (s)")
        lines.append("")
        lines.append("| stat | after | baseline | delta |")
        lines.append("|---|---:|---:|---:|")
        for key in ("mean", "p50", "p90", "p95"):
            b = base_lat.get(key) if isinstance(base_lat, dict) else None
            d = ""
            if isinstance(latency.get(key), (int, float)) and isinstance(b, (int, float)):
                d = f"{latency[key] - b:+.3f}"
            lines.append(f"| {key} | {fmt(latency.get(key))} | {fmt(b)} | {d} |")
        lines.append("")

    results = report.get("results") or []
    if results:
        lines.append("## Per-question")
        lines.append("")
        lines.append("| id | question | intent | final | ctx | retrieval_s | gen_s | total_s | topic | hallu |")
        lines.append("|---|---|---|---:|---:|---:|---:|---:|---|---|")
        for r in results:
            lines.append(
                "| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
                    r.get("id", ""), (r.get("question") or "")[:40].replace("|", "/"),
                    r.get("route_intent") or r.get("intent") or "",
                    r.get("final_count", ""), r.get("context_chars", ""),
                    fmt(r.get("retrieval_time_s")), fmt(r.get("ollama_time_s")),
                    fmt(r.get("total_time_s")),
                    "Y" if r.get("topic_hit") else "n",
                    "Y" if r.get("fabricated_urls") else "n",
                )
            )
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("report", type=Path)
    ap.add_argument("--baseline", type=Path)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    with args.report.open("r", encoding="utf-8") as fh:
        report = json.load(fh)
    baseline = None
    if args.baseline:
        with args.baseline.open("r", encoding="utf-8") as fh:
            baseline = json.load(fh)

    md = render(report, baseline)
    out = args.out or (args.report.with_suffix(".md"))
    out.write_text(md, encoding="utf-8")
    print("Wrote", out)
    print(md)


if __name__ == "__main__":
    main()
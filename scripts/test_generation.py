"""Test LLM generation quality (requires Ollama) on a small question set.

Purpose: verify generation independently of retrieval metrics — every
question must return a non-empty answer with at least one grounded source,
and the retrieval stage is reported separately so regressions can be
attributed to generation vs retrieval.

Usage::

    python scripts/test_generation.py [--limit N] [--questions file]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rag.pipeline.rag import RAGPipeline
from rag.utils.logging_utils import setup_logging

setup_logging()

DEFAULT_QUESTIONS = [
    "ما هي كليات جامعة المنصورة الجديدة؟",
    "Where is New Mansoura University located?",
    "ايه برامج Faculty of Artificial Intelligence؟",
    "How much is the tuition fees?",
    "شروط القبول في جامعة المنصورة الجديدة؟",
    "What programs does the Faculty of Engineering offer?",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Test RAG generation against Ollama")
    parser.add_argument("--questions", type=Path)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--out", type=Path, default=ROOT / "evaluation" / "results")
    args = parser.parse_args()

    questions = (
        [q["question"] for q in _load_jsonl(args.questions)]
        if args.questions
        else DEFAULT_QUESTIONS
    )
    if args.limit:
        questions = questions[: args.limit]

    pipeline = RAGPipeline()
    print("=" * 70)
    print("GENERATION TEST  (model=%s)" % pipeline.ollama.model)
    print("=" * 70)

    rows = []
    failed = 0
    for i, q in enumerate(questions, start=1):
        start = time.time()
        print(f"\n[{i}/{len(questions)}] Q: {q}")
        try:
            result = pipeline.ask(q)
        except Exception as exc:  # noqa: BLE001 - report and keep going
            print(f"    ERROR: {exc}")
            failed += 1
            rows.append({"question": q, "ok": False, "error": str(exc)})
            continue
        elapsed = time.time() - start
        ok = bool(result.answer.strip()) and bool(result.sources)
        if not ok:
            failed += 1
        print(f"    A: {result.answer[:160].replace(chr(10), ' ')}")
        print(
            f"    ok={ok} sources={len(result.sources)} elapsed={elapsed:.1f}s "
            f"ollama={result.timings.get('ollama_request_time', 0):.1f}s"
        )
        rows.append(
            {
                "question": q,
                "ok": ok,
                "answer_length": len(result.answer),
                "n_sources": len(result.sources),
                "elapsed_s": round(elapsed, 1),
                "timings": result.timings,
            }
        )

    print("\n" + "=" * 70)
    print(f"RESULT: {len(questions) - failed}/{len(questions)} questions passed")

    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        out = args.out / "generation_test.json"
        out.write_text(
            json.dumps({"passed": len(questions) - failed, "total": len(questions), "rows": rows},
                       ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"Wrote: {out}")

    sys.exit(1 if failed else 0)


def _load_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


if __name__ == "__main__":
    main()
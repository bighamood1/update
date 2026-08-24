"""End-to-end smoke test: retrieval + generation + sources (requires Ollama).

Small enough to run in a few minutes. Verifies the full pipeline produces a
grounded answer with source URLs for a balanced EN/AR set.

Usage::

    python scripts/test_e2e.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rag.pipeline.rag import RAGPipeline
from rag.retrieval.intents import classify_intent
from rag.utils.logging_utils import setup_logging

setup_logging()

E2E_QUESTIONS = [
    ("ما هي كليات جامعة المنصورة الجديدة؟", "ar", ["كلية", "كليات"]),
    ("Where is New Mansoura University located?", "en", ["Dakahlia", "New Mansoura"]),
    ("What faculties does New Mansoura University have?", "en", ["Faculty", "faculties"]),
    ("شروط القبول في جامعة المنصورة الجديدة؟", "ar", ["قبول", "شروط"]),
]


def main() -> None:
    pipeline = RAGPipeline()
    print("=" * 70)
    print("END-TO-END TEST")
    print("=" * 70)

    passed = 0
    for question, lang, expected in E2E_QUESTIONS:
        start = time.time()
        print(f"\nQ: {question}")
        try:
            result = pipeline.ask(question)
        except Exception as exc:  # noqa: BLE001 - keep running
            print(f"    ERROR: {exc}")
            continue
        elapsed = time.time() - start
        answer = result.answer
        hits = [t for t in expected if t.lower() in answer.lower()]
        ok = bool(answer.strip()) and bool(result.sources) and bool(hits)
        if ok:
            passed += 1
        print(f"    intent={classify_intent(question)} sources={len(result.sources)} hits={hits}")
        print(f"    A: {answer[:140].replace(chr(10), ' ')}")
        print(f"    {'PASS' if ok else 'FAIL'}  ({elapsed:.1f}s)")

    print("\n" + "=" * 70)
    print(f"E2E RESULT: {passed}/{len(E2E_QUESTIONS)} passed")
    sys.exit(0 if passed == len(E2E_QUESTIONS) else 1)


if __name__ == "__main__":
    main()
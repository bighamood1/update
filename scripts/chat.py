"""Interactive chat with the NMU AI Assistant (RAG pipeline).

Usage::

    python scripts/chat.py                # interactive chat
    python scripts/chat.py --debug        # show retrieved chunks per question
    python scripts/chat.py -q "question"  # single question, non-interactive

Commands in interactive mode: ``exit`` / ``quit`` to leave.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rag.pipeline.rag import RAGPipeline
from rag.utils.logging_utils import setup_logging

EXIT_COMMANDS = {"exit", "quit", "خروج", "وداعا"}


def print_sources(result) -> None:
    print("\nSources:")
    for i, s in enumerate(result.sources, start=1):
        line = f"  [{i}] {s.get('title') or 'Untitled'}"
        if s.get("url"):
            line += f"\n      {s['url']}"
        if s.get("score") is not None:
            line += f"  (score {s['score']:.3f})"
        print(line)


def print_debug(result) -> None:
    print("\n[DEBUG] Intent:", result.intent)
    if result.timings:
        t = result.timings
        parts = []
        for key in (
            "embedding_time",
            "bm25_time",
            "retrieval_time",
            "reranking_time",
            "context_assembly_time",
            "ollama_request_time",
            "total_time",
        ):
            if key in t:
                parts.append(f"{key}={t[key]:.3f}s")
        print("[DEBUG] Timings:", "  ".join(parts))
    print("[DEBUG] Retrieved chunks:")
    for i, c in enumerate(result.retrieved_chunks, start=1):
        dense = f"{c.dense_score:.4f}" if c.dense_score is not None else "-"
        bm25 = f"{c.bm25_score:.4f}" if c.bm25_score is not None else "-"
        rerank = f"{c.rerank_score:.4f}" if c.rerank_score is not None else "-"
        print(
            f"  #{i} score={c.score:.4f} dense={dense} bm25={bm25} "
            f"rerank={rerank} type={c.content_type} lang={c.language}"
        )
        print(f"     title: {(c.title or '')[:70]}")
        print(f"     url:   {c.source_url}")
        text = c.text[:160].replace("\n", " ")
        print(f"     text:  {text}...")
    print_sources(result)


def print_context(result) -> None:
    print("\n[CONTEXT] Chunks sent to the LLM:")
    for i, c in enumerate(result.retrieved_chunks, start=1):
        print(f"\n----- Source {i} -----")
        print(c.to_context_block(i))


def run_once(pipeline: RAGPipeline, question: str, debug: bool, context: bool) -> None:
    print(f"\nQ: {question}")
    start = time.time()
    result = pipeline.ask(question)
    elapsed = time.time() - start
    print(f"\nA: {result.answer}")
    print(f"(elapsed {elapsed:.1f}s)")
    if debug:
        print_debug(result)
    elif context:
        print_context(result)
    elif result.sources:
        print_sources(result)


def interactive(pipeline: RAGPipeline, debug: bool, context: bool) -> None:
    print("=" * 64)
    print("NMU AI ASSISTANT  (RAG, local embeddings + local LLM)")
    print("Ask about New Mansoura University. Type 'exit' to quit.")
    print("=" * 64)
    while True:
        try:
            question = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break
        if not question:
            continue
        if question.lower() in EXIT_COMMANDS:
            print("Bye.")
            break
        run_once(pipeline, question, debug, context)


def main() -> None:
    parser = argparse.ArgumentParser(description="Chat with the NMU AI Assistant")
    parser.add_argument("-q", "--question", help="Ask a single question and exit")
    parser.add_argument("--debug", action="store_true", help="Show retrieved chunks")
    parser.add_argument(
        "--context",
        action="store_true",
        help="Show the exact context blocks sent to the LLM",
    )
    args = parser.parse_args()

    setup_logging()

    if args.debug and args.question:
        print("[debug] single-question mode")

    pipeline = RAGPipeline()
    if args.question:
        run_once(pipeline, args.question, args.debug, args.context)
    else:
        interactive(pipeline, args.debug, args.context)


if __name__ == "__main__":
    main()

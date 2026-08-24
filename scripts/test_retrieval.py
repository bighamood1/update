"""Test retrieval quality across Arabic, English, and mixed-language queries.

Purpose: verify retrieval BEFORE blaming the LLM. For every question, print
the top results, scores, and sources.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rag.config import get_config
from rag.embeddings.embedder import Embedder
from rag.retrieval.retriever import Retriever
from rag.utils.logging_utils import setup_logging
from rag.vectorstore.store import VectorStore

setup_logging()

QUESTIONS = [
    # university identity
    ("What is New Mansoura University?", "identity"),
    ("ما هي جامعة المنصورة الجديدة؟", "identity-ar"),
    # faculties
    ("What faculties does New Mansoura University have?", "faculties"),
    ("ما هي كليات جامعة المنصورة الجديدة؟", "faculties-ar"),
    ("Which faculties are available?", "faculties-en2"),
    # programs
    ("What programs does the Faculty of Engineering offer?", "programs"),
    ("ايه برامج Faculty of Artificial Intelligence؟", "programs-mixed"),
    # admission
    ("What are the admission requirements for New Mansoura University?", "admission"),
    ("شروط القبول في جامعة المنصورة الجديدة؟", "admission-ar"),
    # tuition
    ("How much is the tuition fees?", "tuition"),
    ("مصروفات الجامعة؟", "tuition-ar"),
    # scholarships
    ("What scholarships are available?", "scholarship"),
    # president
    ("Who is the president of New Mansoura University?", "president"),
    ("من هو رئيس جامعة المنصورة الجديدة؟", "president-ar"),
    # facilities
    ("What facilities does the university have?", "facilities"),
    # FAQ
    ("What are the faculties and programs of the university?", "faq"),
    # news
    ("What is the latest news about the university?", "news"),
    # regulations
    ("What are the university regulations?", "regulation"),
    # contact
    ("How can I contact the university?", "contact"),
]


def main() -> None:
    cfg = get_config()
    store = VectorStore()
    embedder = Embedder()
    retriever = Retriever(vectorstore=store, embedder=embedder)

    if not store.is_built():
        print("[ERROR] Index not built. Run: python scripts/build_index.py")
        sys.exit(1)

    print("=" * 70)
    print("RETRIEVAL TEST  (TOP_K=%d  THRESHOLD=%.2f)" % (cfg["top_k"], cfg["similarity_threshold"]))
    print("=" * 70)

    results = []
    for question, tag in QUESTIONS:
        chunks = retriever.retrieve(question)
        print(f"\n--- [{tag}] Q: {question}")
        if not chunks:
            print("    NO RESULTS above threshold")
            results.append((question, tag, []))
            continue
        for i, c in enumerate(chunks, start=1):
            print(f"  #{i} score={c.score:.3f} type={c.content_type} lang={c.language}")
            print(f"     title: {(c.title or '')[:80]}")
            print(f"     url:   {c.source_url}")
            print(f"     text:  {c.text[:120]}...")
        results.append((question, tag, chunks))

    # Summary
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for question, tag, chunks in results:
        avg = sum(c.score for c in chunks) / len(chunks) if chunks else 0.0
        print(f"  {tag:16s} n={len(chunks)} avg_score={avg:.3f}")


if __name__ == "__main__":
    main()
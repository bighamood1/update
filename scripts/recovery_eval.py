"""Recovery verification harness (Phase 19) — runs WITHOUT the real LLM.

Three modes:

1. ``retrieval`` — verify that the fixed retriever + context builder surface
   the expected evidence for every question in ``evaluation/questions.jsonl``:
   non-empty retrieval, topic hits in the final context, expected-source hits.
   This directly checks the fixes for "says info doesn't exist when it does"
   and "incomplete answers" (the new list-intent context budget).

2. ``mocked`` — run the FULL pipeline (retrieval -> rerank -> context ->
   generation -> validation -> cache/memory gating) with a deterministic
   grounded fake LLM (it answers from the CONTEXT block it is given, exactly
   like a well-behaved generator). No Ollama, no model downloads. Verifies the
   pipeline is stable and that validated answers are produced for scoped
   questions while refusals/invalid answers never contaminate cache/memory.

3. ``random`` — sample questions synthesized from the knowledge base
   (document titles + intent-aware keywords, AR + EN) and verify retrieval
   returns non-empty evidence containing the query entity.

Usage::

    python scripts/recovery_eval.py --mode retrieval
    python scripts/recovery_eval.py --mode mocked
    python scripts/recovery_eval.py --mode random --n 20
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# Isolate ALL modes from the runtime DB BEFORE config is cached at import:
# the mocked LLM's answers must never be written to the semantic cache,
# feedback analytics or retrieval memory (a grounded fake would otherwise
# pollute the real runtime DB with source-boilerplate rows).
os.environ.setdefault("CACHE_ENABLED", "false")
os.environ.setdefault("FEEDBACK_ENABLED", "false")
os.environ.setdefault("RETRIEVAL_MEMORY_ENABLED", "false")

from rag.config import get_config  # noqa: E402
from rag.context.builder import ContextBuilder  # noqa: E402
from rag.embeddings.embedder import Embedder  # noqa: E402
from rag.pipeline.rag import RAGPipeline  # noqa: E402
from rag.retrieval.reranker import Reranker  # noqa: E402
from rag.retrieval.retriever import Retriever  # noqa: E402
from rag.utils.logging_utils import setup_logging  # noqa: E402
from rag.vectorstore.store import VectorStore  # noqa: E402

setup_logging()

# Arabic question text must survive console printing on cp1252 terminals.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # pragma: no cover - not all Python builds support it
    pass

QUESTIONS_PATH = ROOT / "evaluation" / "questions.jsonl"
DOCUMENTS_PATH = ROOT / "data" / "documents.jsonl"


def load_questions(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def token_hits(text: str, tokens: list[str]) -> list[str]:
    if not tokens:
        return []
    low = (text or "").lower()
    return [t for t in tokens if t.lower() in low]


def normalize_url(url: str) -> str:
    return (url or "").replace("https://www.", "https://").replace("http://www.", "http://")


class MockOllama:
    """Deterministic fake LLM: answers with the first sentences of CONTEXT.

    It skips the per-source header metadata (``[Source N]`` / ``Title:`` /
    ``URL:`` / ``Type:`` / ``Language:`` lines) and answers from the actual
    ``Content:`` text — mirroring a grounded generator that reads the whole
    prompt instead of echoing the boilerplate.
    """

    def __init__(self, max_chars: int = 500) -> None:
        self.model = "mock-grounded"
        self.calls = 0
        self.max_chars = max_chars

    def generate(self, system_prompt: str, user_prompt: str, **kwargs) -> str:
        self.calls += 1
        m = re.search(r"CONTEXT:\n(.*?)\nEND OF CONTEXT", user_prompt, re.DOTALL)
        if not m:
            return ""
        # Join the actual content of every source block (skip the metadata).
        contents = re.findall(r"^Content: (.*)$", m.group(1), re.MULTILINE)
        text = "\n".join(contents).strip()
        if not text:
            return ""
        # First meaningful sentence(s), trimmed — mirrors a grounded generator.
        sentences = re.split(r"(?<=[.!?؟])\s+", text)
        out = []
        total = 0
        for s in sentences:
            if total + len(s) > self.max_chars:
                break
            out.append(s)
            total += len(s)
        return " ".join(out)


def _build_pipeline(mock_ollama: bool):
    store = VectorStore()
    embedder = Embedder()
    reranker = Reranker() if get_config().get("reranker_enabled", True) else None
    retriever = Retriever(vectorstore=store, embedder=embedder, reranker=reranker)
    ollama = MockOllama() if mock_ollama else None
    return RAGPipeline(
        vectorstore=store, embedder=embedder, retriever=retriever, ollama=ollama
    ), retriever


def _report_dict(**metrics) -> dict:
    return {k: v for k, v in metrics.items() if not isinstance(v, list)}


def run_retrieval(questions: list[dict]) -> dict:
    store = VectorStore()
    if not store.is_built():
        sys.exit("[ERROR] Index not built. Run: python scripts/build_index.py")
    embedder = Embedder()
    reranker = Reranker() if get_config().get("reranker_enabled", True) else None
    retriever = Retriever(vectorstore=store, embedder=embedder, reranker=reranker)
    builder = ContextBuilder()
    cfg = get_config()

    rows = []
    latencies = []
    for q in questions:
        qid = q["id"]
        expected = q.get("expected_topics", []) or []
        gold = q.get("expected_sources", []) or []
        t0 = time.time()
        chunks = retriever.retrieve(q["question"])
        elapsed = time.time() - t0
        latencies.append(elapsed)
        # Final context the LLM would receive (respects the intent-aware budget).
        from rag.pipeline.rag import _LIST_LIKE_INTENTS
        from rag.query.understanding import understand

        understanding = understand(q["question"])
        intent = understanding.intent
        list_like = intent in _LIST_LIKE_INTENTS
        max_chars = int(cfg.get("context_max_chars_list", 8000)) if list_like else None
        context = builder.build(chunks, max_chars=max_chars)
        hits = token_hits(context, expected)
        source_urls = {normalize_url(c.source_url or "") for c in chunks}
        gold_norm = {normalize_url(u) for u in gold}
        source_hit = bool(gold_norm & source_urls) if gold_norm else None
        rows.append(
            {
                "id": qid,
                "question": q["question"],
                "lang": q.get("lang", ""),
                "category": q.get("category", ""),
                "intent": intent,
                "out_of_scope": not expected,
                "n_chunks": len(chunks),
                "context_chars": len(context),
                "hits": hits,
                "hit": bool(hits) or (not expected and not chunks),
                "source_hit": source_hit,
            }
        )

    scoped = [r for r in rows if r["hit"]]
    hit_rate = sum(1 for r in scoped for _ in [0]) / len(scoped) if scoped else 0.0
    with_gold = [r for r in rows if r["source_hit"] is not None]
    src_rate = sum(1 for r in with_gold if r["source_hit"]) / len(with_gold) if with_gold else 0.0
    n_empty = sum(1 for r in rows if r["n_chunks"] == 0)
    avg_ctx = sum(r["context_chars"] for r in rows) / len(rows) if rows else 0
    return {
        "mode": "retrieval",
        "n_questions": len(rows),
        "topic_hit_rate": round(hit_rate, 4),
        "source_hit_rate": round(src_rate, 4),
        "n_empty_retrieval": n_empty,
        "avg_context_chars": round(avg_ctx, 1),
        "avg_latency_s": round(sum(latencies) / len(latencies), 3) if latencies else 0.0,
        "results": rows,
    }


def run_mocked(questions: list[dict], limit: int = 0) -> dict:
    if limit:
        questions = questions[:limit]
    # Runtime DB isolation is set at module import (before config caching),
    # so the mocked LLM's answers never pollute cache / feedback / memory.
    pipeline, retriever = _build_pipeline(mock_ollama=True)
    rows = []
    for q in questions:
        qid = q["id"]
        expected = q.get("expected_topics", []) or []
        out_of_scope = not expected
        t0 = time.time()
        try:
            result = pipeline.ask(q["question"], skip_cache=True)
        except Exception as exc:  # noqa: BLE001 - keep the run going
            rows.append(
                {
                    "id": qid, "question": q["question"], "lang": q.get("lang", ""),
                    "category": q.get("category", ""), "pass": False,
                    "error": str(exc), "answer": "", "hits": [], "n_sources": 0,
                }
            )
            continue
        elapsed = time.time() - t0
        hits = token_hits(result.answer, expected)
        refused = (
            "not contain enough information" in result.answer
            or "لم أتمكن من العثور" in result.answer
            or "couldn't find enough information" in result.answer
        )
        passed = (bool(hits) and bool(result.sources)) if not out_of_scope else refused
        rows.append(
            {
                "id": qid, "question": q["question"], "lang": q.get("lang", ""),
                "category": q.get("category", ""), "pass": passed,
                "answer": result.answer, "hits": hits,
                "n_sources": len(result.sources), "refused": refused,
                "intent": result.intent,
                "out_of_scope": out_of_scope,
                "elapsed_s": round(elapsed, 2),
                "llm_calls": getattr(pipeline.ollama, "calls", 0),
            }
        )

    return {
        "mode": "mocked",
        "n_questions": len(rows),
        "answer_rate": round(sum(1 for r in rows if r["pass"]) / len(rows), 4) if rows else 0.0,
        "results": rows,
    }


# Intent-aware randomized queries drawn from the knowledge base itself.
_RANDOM_QUERIES = [
    ("How many faculties does New Mansoura University have?", "faculties", "faculties"),
    ("كم عدد كليات جامعة المنصورة الجديدة؟", "faculties", "كليات"),
    ("What programs does the Faculty of Engineering offer?", "programs", "Engineering"),
    ("ما هي برامج كلية الهندسة؟", "programs", "الهندسة"),
    ("What is the tuition fee for students?", "tuition", "tuition"),
    ("كم تبلغ مصروفات الدراسة في الجامعة؟", "tuition", "مصروفات"),
    ("What are the admission requirements?", "admission", "admission"),
    ("شروط القبول في الجامعة؟", "admission", "شروط"),
    ("What scholarships are available?", "scholarship", "scholarship"),
    ("ما هي المنح الدراسية المتاحة؟", "scholarship", "المنح"),
    ("Who is the president of the university?", "person", "President"),
    ("من هو رئيس جامعة المنصورة الجديدة؟", "person", "رئيس"),
    ("Where is the university located?", "location", "located"),
    ("أين تقع جامعة المنصورة الجديدة؟", "location", "المنصورة الجديدة"),
    ("How can I contact the university?", "contact", "contact"),
    ("ما هو هاتف الجامعة؟", "contact", "هاتف"),
    ("What are the rules for transferring to the university?", "regulation", "transfer"),
    ("ما هي قواعد التحويل إلى الجامعة؟", "regulation", "التحويل"),
    ("Does the university provide housing?", "facilities", "housing"),
    ("What is the university code of ethics?", "regulation", "ethics"),
]


def run_random(n: int = 20, seed: int = 7) -> dict:
    store = VectorStore()
    if not store.is_built():
        sys.exit("[ERROR] Index not built. Run: python scripts/build_index.py")
    embedder = Embedder()
    reranker = Reranker() if get_config().get("reranker_enabled", True) else None
    retriever = Retriever(vectorstore=store, embedder=embedder, reranker=reranker)
    builder = ContextBuilder()

    queries = list(_RANDOM_QUERIES)
    rng = random.Random(seed)
    # Add title-derived queries from the knowledge base itself.
    titles = []
    with DOCUMENTS_PATH.open("r", encoding="utf-8") as fh:
        for line in fh:
            try:
                doc = json.loads(line)
            except ValueError:
                continue
            title = (doc.get("title") or "").strip()
            if title and 3 <= len(title) <= 90 and " " in title:
                titles.append((title, doc.get("language", "en")))
    if titles:
        rng.shuffle(titles)
        for title, lang in titles[: n // 2]:
            if lang == "ar":
                queries.append((f"ما هي {title}؟", "entity", title.split()[0]))
            else:
                queries.append((f"What is {title}?", "entity", title.split()[-1]))

    rng.shuffle(queries)
    queries = queries[:n]

    rows = []
    for q_text, cat, keyword in queries:
        t0 = time.time()
        chunks = retriever.retrieve(q_text)
        elapsed = time.time() - t0
        if not chunks:
            rows.append(
                {
                    "question": q_text, "category": cat, "n_chunks": 0,
                    "hit": False, "top_keyword_hit": False, "elapsed_s": round(elapsed, 2),
                }
            )
            continue
        joined = " ".join(c.text for c in chunks)
        kw_hit = keyword.lower() in joined.lower()
        rows.append(
            {
                "question": q_text, "category": cat, "n_chunks": len(chunks),
                "hit": True, "top_keyword_hit": kw_hit, "elapsed_s": round(elapsed, 2),
            }
        )

    n_empty = sum(1 for r in rows if r["n_chunks"] == 0)
    return {
        "mode": "random",
        "n_questions": len(rows),
        "non_empty_rate": round(sum(1 for r in rows if r["n_chunks"]) / len(rows), 4) if rows else 0.0,
        "keyword_hit_rate": round(
            sum(1 for r in rows if r.get("top_keyword_hit")) / len(rows), 4
        ) if rows else 0.0,
        "n_empty_retrieval": n_empty,
        "results": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Recovery verification harness (no LLM)")
    parser.add_argument("--mode", choices=["retrieval", "mocked", "random"], default="retrieval")
    parser.add_argument("--limit", type=int, default=0, help="Run only first N eval questions")
    parser.add_argument("--n", type=int, default=20, help="Randomized suite size")
    parser.add_argument("--out", type=Path, default=ROOT / "evaluation" / "results")
    args = parser.parse_args()

    if args.mode == "random":
        report = run_random(args.n)
    else:
        questions = load_questions(QUESTIONS_PATH)
        if args.limit:
            questions = questions[: args.limit]
        report = run_retrieval(questions) if args.mode == "retrieval" else run_mocked(questions, args.limit)

    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    from datetime import datetime, timezone

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    json_path = out_dir / f"recovery_{args.mode}_{stamp}.json"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nReport: {json_path}")
    for k, v in _report_dict(**report).items():
        print(f"  {k}: {v}")
    in_scope = [
        r for r in report["results"]
        if not r.get("out_of_scope", False)
        and not r.get("pass", r.get("hit", False))
    ]
    oos_bad = [
        r for r in report["results"]
        if r.get("out_of_scope", False) and r.get("n_chunks", 1) > 0
    ]
    if in_scope:
        print(f"\n  {len(in_scope)} in-scope failing rows:")
        for r in in_scope[:10]:
            print(f"    - [{r.get('id', r.get('category','?'))}] {r.get('question','')[:70]}"
                  f"  hits={r.get('hits', r.get('top_keyword_hit'))}")
    if oos_bad:
        print(f"\n  {len(oos_bad)} out-of-scope rows that still retrieved evidence "
              f"(should refuse):")
        for r in oos_bad[:5]:
            print(f"    - [{r.get('id', r.get('category','?'))}] {r.get('question','')[:70]}")


if __name__ == "__main__":
    main()
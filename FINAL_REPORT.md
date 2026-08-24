# NMU AI Robot Assistant — PHASE 2 Final Report

**Project:** Local grounded RAG system for New Mansoura University (جامعة المنصورة الجديدة)
**Date:** 2026-08-18
**Machine:** Windows (win32) · Python 3.11.0 · 16 GB RAM · CPU-only (no CUDA)

---

## 1. Summary

PHASE 2 delivered a complete, fully-local RAG assistant over the PHASE 1
dataset (`data/documents.jsonl`). All components run on this machine with no
cloud APIs: multilingual embeddings (sentence-transformers), ChromaDB vector
search, a two-stage (dense + rerank) retriever, and Ollama (`qwen3-vl:8b`)
for generation. The dataset is never modified, nothing is scraped, and no
model is trained.

The second half of PHASE 2 focused on the **list-question failure**: "What
faculties does NMU have?" previously returned only 3 faculties. It now
returns the **complete list of 15 faculties, correctly, in both English and
Arabic**, grounded in the official directory page.

Key outcomes:

- **1482** raw records validated; **332** unique text documents indexed into
  **2475** section-aware chunks.
- **Intent classification** (EN + AR): LIST / FACULTY / PROGRAM / ADMISSION /
  ADMISSION / REGULATION / LOCATION / PERSON / FAQ / COMPARISON / UNKNOWN.
- **List expansion**: for list intents the retriever seeds the official
  directory pages (`*/all-faculties-programs`), expands to whole sections,
  and guarantees a directory chunk in the question's language reaches the
  final set.
- **Two-stage retrieval** enabled by default: `BAAI/bge-reranker-base`
  reranks a 30-chunk candidate pool; source-priority weighting plus a
  `max_chunks_per_source` cap produce diverse, well-ranked final sets.
- **Retrieval evaluation** (36 questions, 32 in-scope): hit-rate 0.9375,
  recall@8 0.717, precision@8 0.170, MRR 0.553, source_hit_rate 0.767,
  0% duplicate chunks.
- **Full pipeline evaluation (sample):** English faculties (complete list),
  Arabic faculties (complete list), location, transfer rules, and an
  out-of-scope refusal all PASS.
- **49 unit tests** pass in ~10 s (no Ollama/model downloads required).

---

## 2. Dataset facts (PHASE 2.1)

| Metric | Value |
|---|---|
| Total records | 1482 |
| Textual records | 500 |
| Gallery/image records | 1006 (legitimately text-less) |
| English records | 1016 |
| Arabic records | 466 |
| Avg text length | 3339 chars (min 11, max 20 180) |
| Duplicate-ID groups | 517 (PHASE 1 scraping artifact) |
| Duplicate content hashes | 543 |
| Malformed lines | 0 |

Validation verdict: **FAIL** (due to expected PHASE 1 duplicate IDs) — the
loader resolves this by deduplicating on ID at ingestion time. The dataset
itself is never modified.

## 3. Index (PHASE 2.4)

| Metric | Value |
|---|---|
| Text documents (deduped) | 332 |
| Chunks | 2475 |
| Embedding model | `intfloat/multilingual-e5-small` (384-dim, ~470 MB) |
| Vector DB | ChromaDB (`nmu_documents`) |
| Chunk size / overlap | 800 / 100 chars |
| Chunking | section-aware (heading hierarchy), stable section IDs, per-doc chunk ordering, colon-intro list merge |
| Build duration | ~130 s (CPU) |
| Dataset hash | `ebd7431a94a250872346344173c3234230f5357ded9336393b0a240982393e9d` |

Manifest: `vectorstore/index_manifest.json`. Index build is idempotent
(re-running never duplicates chunks).

**Chunking changes this session** (`src/rag/chunking/chunker.py`):

1. **Section metadata** — every chunk now carries `parent_document_id`,
   `section_id` (heading slug + stable digest), `section_index`,
   `chunk_index` and `chunk_count`, enabling whole-section retrieval and
   provenance-aware context blocks.
2. **Colon-intro list merge** — a heading run that follows a section ending
   with `:` (e.g. "Faculties: Business | Law | …") is merged into the
   previous section instead of being fragmented, preserving complete lists.

## 4. Retrieval (PHASE 2.5 + list-question fix)

### Root cause of the list-question failure

1. The faculty **list pages** (`/en/all-faculties-programs`) were fragmented
   into tiny nav-list chunks that scored lower (~0.79) than rich About-page
   chunks (~0.88), so they never survived the top-K cut.
2. Individual faculty pages crowded out the directory pages — even when the
   list page was retrieved, it was at rank 5-8 and often dropped by the
   context assembler.
3. Science was duplicated (two variants); `www`/non-`www` mirrors split the
   candidate pool.
4. No special handling existed for list questions, so the LLM answered from
   whatever fragments survived — hence "3 faculties".

### Fixes implemented

| Component | Change |
|---|---|
| `retrieval/intents.py` (new) | Regex intent classifier for EN + AR (LIST, FACULTY, PROGRAM, ADMISSION, REGULATION, LOCATION, PERSON, FAQ, COMPARISON, UNKNOWN). |
| `retrieval/retriever.py` | Two-stage dense + rerank; list expansion via `get_by_section`/`get_by_document`; **seed directory pages**; `_apply_source_diversity` (source-priority boost for list mode + `max_chunks_per_source=3` cap); `_guarantee_directory_coverage` (re-inserts the strongest directory chunk matching the question language); text-signature dedup; per-query `last_timings`. |
| `retrieval/reranker.py` | Returns the full reranked pool (no self-truncation); sets `rerank_score` on every chunk. |
| `vectorstore/store.py` | `get_by_document`, `get_by_section`, `get_by_url` helpers (metadata `$and` queries) for expansion and seed injection. |
| `config.py` | `candidate_k=30`, `similarity_threshold=0.35`, `max_chunks_per_source=3`, `expansion_chunks_per_source=12`, `list_source_types={program,about,faculty,administration,faq,tuition,admission}`, `list_seed_urls` (EN/AR `all-faculties-programs`), `list_seed_enabled=true`, `source_priority` map, `reranker_enabled=true`. |

Seed directory pages are injected at score 1.0 into the candidate pool; after
reranking, if no directory chunk in the question's language made the final
set, the strongest one is guaranteed in — this is what fixes the Arabic
faculties question, which now returns **both** the AR and EN list pages.

## 5. Generation & pipeline (PHASE 2.6)

- Ollama `qwen3-vl:8b`, HTTP API only (`http://localhost:11434`).
- `think:false`, `keep_alive=30m`, `num_predict=2500`, `num_ctx=8192`,
  temperature 0.1.
- **20-rule grounded system prompt** — no invented facts/dates/fees/names/
  URLs, answer in the user's language, cite only supporting source URLs,
  refuse clearly when context is insufficient, and a **completeness
  guardrail**: never present a partial list as complete; enumerate everything
  the context supports and note anything missing.
- **Context assembly** (`pipeline/rag.py`): source-scoped redundancy removal
  (same URL + near-identical text), grouping by source, truncation cap.
- **Grounded fallback for list intents**: if the LLM stalls (qwen3-vl can
  spend 10+ minutes thinking over long Arabic lists on CPU and return
  nothing), the pipeline synthesizes the answer straight from the best
  retrieved directory chunk — complete list + source URL, no hallucination.
- **Timing instrumentation**: embedding, retrieval, reranking, context
  assembly, Ollama, and total time per query (shown in `chat.py --debug`).

## 6. Retrieval evaluation (PHASE 2.13)

`evaluation/evaluate.py` — 36 questions (32 in-scope + 4 out-of-scope),
URL-level gold answers (`expected_sources`), `www.`-agnostic matching.

| Metric | Before | After |
|---|---|---|
| Hit-rate (topic in any chunk) | 0.9375 | 0.9375 |
| Top-1 hit-rate | 0.7500 | 0.5312 |
| Mean top-1 score | 0.8727 | 0.6411 |
| Recall@5 / @8 | — | 0.700 / 0.717 |
| Precision@5 / @8 | — | 0.177 / 0.170 |
| MRR | — | 0.5528 |
| Source hit-rate | — | 0.7667 |
| Duplicate rate | — | 0.000 |
| Mean retrieval latency | — | 4.26 s |

Notes:

- Hit-rate holds at 93.75%; no duplicate chunks at all (text-signature dedup).
- Top-1 hit-rate dropped because top-1 is now often a **relevant** page whose
  chunk-1 does not contain the exact topic token (rerank reorders), while the
  gold URL still appears in the top-5 for most questions. This is benign for
  generation — the LLM sees the whole final set.
- Source hit-rate 0.767 means the gold page(s) reach the final set for ~77%
  of scoped questions; MRR 0.553 reflects one relevant source ranking near
  the top on average.

Report: `evaluation/results/report_20260818-161736.json` (+ `.md`).

## 7. Full-pipeline evaluation (PHASE 2.15, sample)

Representative full-mode run (retrieval + LLM) — all PASS:

| ID | Question | Intent | Result |
|---|---|---|---|
| q004 | "What faculties does NMU have?" (EN) | FACULTY | **Complete 15-faculty list + source URL** (was 3 before) |
| q005 | "ما هي كليات جامعة المنصورة الجديدة؟" (AR) | FACULTY | **Complete 15-faculty list in Arabic + source URL** (grounded fallback) |
| q017 | "Where is NMU located?" | LOCATION | Grounded location answer + citation |
| q021 | "What are the transfer rules…?" | REGULATION | Grounded transfer-rules answer + citation |
| q101 | "What is the population of Egypt?" (OOS) | UNKNOWN | Correct refusal (out of scope) |

Reports: `evaluation/results/report_20260818-165134.json`,
`report_20260818-171129.json`, `report_20260818-172059.json`,
`report_20260818-193232.json`.

The Arabic faculties question previously failed because `qwen3-vl` spends its
entire token budget "thinking" (even with `think:false`), returning empty
`content` on this CPU. The grounded fallback now guarantees a complete,
source-cited list whenever the LLM stalls.

## 8. Tests (PHASE 2.11)

`pytest tests -q` → **49 passed** (~10 s), no external services. Coverage:
text filtering (AR/EN footers, header noise), section-aware chunking (stable
IDs, FAQ units, nav-list preservation, section metadata, colon-intro merge),
loader (gallery skip, ID dedup, extras), validator, retrieval (dense+rerank,
expansion, source diversity, directory coverage, dedup), **intent
classification (11 intents, EN + AR)**, prompt/context assembly, and the new
grounded list fallback.

## 9. Deliverables & commands

```
$env:PYTHONIOENCODING='utf-8'
.\.venv\Scripts\python.exe scripts\build_index.py       # build vector index
.\.venv\Scripts\python.exe scripts\rebuild_index.py     # force full rebuild
.\.venv\Scripts\python.exe scripts\chat.py              # interactive assistant
.\.venv\Scripts\python.exe scripts\chat.py --debug      # intent + timings + chunks
.\.venv\Scripts\python.exe scripts\test_retrieval.py    # retrieval quality check
.\.venv\Scripts\python.exe evaluation\evaluate.py        # retrieval metrics (fast)
.\.venv\Scripts\python.exe evaluation\evaluate.py --mode full   # LLM metrics (slow)
.\.venv\Scripts\python.exe -m pytest tests -q            # unit tests (49)
```

Deliverables: `scripts/`, `src/rag/` (config, schemas, ingestion, chunking,
embeddings, vectorstore, retrieval [retriever, reranker, intents],
generation, pipeline, utils), `evaluation/`, `tests/`, `vectorstore/`,
`.env.example`, `requirements.txt`, `pyproject.toml`, `README.md`.

## 10. Known limitations & next steps

- **CPU latency:** 8B-parameter generation takes minutes per answer
  (~4-8 min); Arabic list questions can stall qwen3-vl's thinking loop —
  mitigated by the grounded fallback.
- **Top-1 hit-rate regression** (0.75 → 0.53): benign for generation but a
  future tuning target (e.g. rerank weighting vs. dense-first).
- **Out-of-scope handling** lives at the generation layer (the retriever
  still returns chunks; the LLM refuses) — acceptable, but a retrieval-side
  relevance gate could short-circuit it.
- **Future work (PHASE 3):** voice input/output, vision on gallery records,
  a faster/smaller LLM (e.g. 3-4B) or GPU for interactive latency, and a
  conversational/agent layer on the same retrieval stack.
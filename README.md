# NMU AI Robot Assistant — PHASE 2 (Local RAG)

A fully-local, grounded question-answering system for **New Mansoura
University (جامعة المنصورة الجديدة)** built in **PHASE 2** on top of the
website data scraped in **PHASE 1** (`data/documents.jsonl`).

Everything runs locally: a multilingual sentence-embedding model, ChromaDB for
vector search, a BGE reranker, and **Ollama** for generation. No cloud APIs,
no scraping, no training, and the dataset is never modified.

---

## Features

- **Arabic + English + mixed-language** questions and answers.
- **Grounded answers only**: the assistant answers from retrieved official
  NMU pages and cites their URLs; it refuses when the knowledge base lacks
  the information. Post-generation validation strips any fabricated URL.
- **Deterministic query routing** (no LLM): intent, category, faculty and
  language are detected locally; high-confidence routes apply metadata-first
  filters (language / content type / faculty) with an automatic BROAD fallback
  so recall is never silently lost.
- **Hybrid retrieval**: dense embeddings + Okapi BM25 with Arabic light
  normalization, fused via reciprocal-rank fusion; optional cross-encoder
  reranker for precision.
- **Evidence coverage guards**: list, tuition, location and named-faculty
  program queries preserve required evidence before the final top-k cut. A
  named faculty question hydrates that faculty's indexed page chunks from
  Chroma instead of relying on broad directory pages alone.
- **Fast path for structured queries** (faculties list / university location /
  contact info / clear academic hierarchy / tuition tables): answered
  deterministically from authoritative indexed evidence — no LLM call at all.
  Falls back to normal RAG unless matching authoritative chunks are present
  (nothing is hardcoded).
- **Intent-aware context sizing**: simple facts send 3 chunk groups to the
  LLM, lists/comparisons up to 6 — smaller prompts = faster generation
  without reducing retrieval top-k.
- **Content-aware chunking** (FAQ Q/A units, faculty pages, nav menus kept
  intact) with stable, traceable chunk IDs.
- **Deduplication**: duplicate IDs (PHASE 1 artifact) and mirrored
  `www` / non-`www` pages are collapsed at ingestion and retrieval time.
- **Caching**: embedding + retrieval LRU caches (TTL-bounded) make repeated
  queries resolve in milliseconds.
- **Query understanding** (no LLM): intent/language/entities + conservative
  **multi-intent splitting** (`وما`, `and what`, ...) — each sub-question is
  retrieved separately and merged into one coherent evidence-based answer.
- **High-recall paraphrase retrieval**: deterministic query variants cover
  Arabic/English synonyms, colloquial wording, CSE/faculty aliases, location
  wording, scholarships, tuition and program/department terminology.
- **Persistent semantic response cache**: repeated/similar questions short-
  circuit the whole pipeline, but ONLY when embedding similarity + intent +
  knowledge-base version + quality gates all pass; any uncertainty reruns full
  RAG.
- **Feedback + analytics**: `useful / somewhat / not_useful` ratings with per-
  rating reasons; SQLite runtime store tracks questions, clusters (top FAQs),
  failed answers, retrieval memory and strategy feedback. **Feedback is never
  used to fine-tune the model or memorize replacement answers** — it drives
  cache gating, retrieval hints, strategy selection and offline analysis.
- **Retrieval memory**: after a successful answer the best source URLs are
  remembered and seeded as *soft* hints (still re-ranked; fallback preserved).
- **Adaptive candidate pool + evidence-based reasoning**: high-confidence
  routes search narrower windows (recall floor enforced), and the prompt
  explicitly instructs combining evidence across passages.
- **Strict generation contract**: concise prompts tell the LLM to return only
  the final answer, with no source/context/retrieval discussion. Simple facts
  use smaller intent-aware output budgets and stop sequences to prevent
  runaway analysis.
- **Generation cleanup**: deterministic validation and post-processing remove
  leaked local model reasoning preambles / `<think>` blocks, source/context
  labels, bare URLs and repeated lines/sentences before caching or display.
- **Clean response contract**: the LLM receives URL-free evidence blocks, the
  API returns `answer` plus structured `sources`, and a final formatter strips
  source labels, raw context headers, bare URLs and duplicate lines from the
  user-facing answer.
- **Training-data export**: `scripts/export_training_data.py` writes a clean
  JSONL for a future LoRA / DPO / reranker run (offline only).
- Evaluation harness (retrieval-only mode is fast; full mode exercises the
  LLM), plus a unit-test suite that does not require Ollama or model
  downloads.

---

## Architecture

```
data/documents.jsonl ──► ingestion (loader + text filter + validator)
                             │  + metadata enrichment (faculty from URL)
                       chunking (content-type aware, stable IDs)
                             │
                    embeddings (multilingual, local, CPU)
                             │
                     vector store (ChromaDB, manifest-versioned)
                             │
              routing (intent/category/faculty/language + confidence)
                             │
     retrieval (dense + BM25 hybrid, RRF fusion, metadata filters,
                 BROAD fallback, evidence expansion/coverage,
                 dynamic top-k, optional reranker)
                             │
        ┌────────────────────┴─────────────────────┐
   fast path (structured            clean evidence context + compression
   location / faculties /                │
   contact queries)                   Ollama (auto-selected model,
        │                          intent-aware context/output caps,
        │                          think:false)
        ▼                              │
   answer + sources ◄─ formatter ◄ validation ◄─┘
```

- **Fast path**: for LIST/FACULTY/PROGRAM/TUITION/LOCATION/CONTACT intents,
  deterministic extractors build answers straight from retrieved authoritative
  evidence when the structure is clear: faculty directories, location/contact
  rows, department/program labels, and tuition tables. If evidence is not
  explicit enough, the request flows through normal RAG generation.
- **Model selection**: the configured model (`OLLAMA_MODEL`) must be installed
  or the server reports a clear error. When unset, the server auto-selects the
  first *installed* model from `OLLAMA_PREFERRED_MODELS` (faster text models
  first) — it never downloads anything.
- **Concurrency**: Ollama generation concurrency is bounded by
  `MAX_CONCURRENT_GENERATIONS` (default 1); a request beyond the limit gets an
  immediate `busy` (HTTP 409) response.
- **Context contract**: grounding context is formatted as compact
  `Evidence item` blocks with factual content and minimal metadata. URLs and
  source scores are not sent to the LLM; source URLs stay in the structured
  `sources` field returned by the API/GUI.
- **Response contract**: the final `answer` field is only user-facing text.
  Retrieval traces, chunks, scores, validation diagnostics and source URLs are
  backend-only unless a development script/API explicitly requests them.

| Layer        | Component                                             | File                               |
| ------------ | ----------------------------------------------------- | ---------------------------------- |
| Config       | env-driven settings                                   | `src/rag/config.py`              |
| Ingestion    | JSONL loader, dedup, text filter, validator, metadata | `src/rag/ingestion/`             |
| Chunking     | content-type-aware splitter                           | `src/rag/chunking/chunker.py`    |
| Embeddings   | sentence-transformers (e5-small) + LRU cache          | `src/rag/embeddings/embedder.py` |
| Vector store | ChromaDB persistent + manifest + compat checks        | `src/rag/vectorstore/store.py`   |
| Routing      | deterministic query router (no LLM)                   | `src/rag/routing/`               |
| Retrieval    | hybrid dense+BM25, RRF, filters, fallback, reranker   | `src/rag/retrieval/`             |
| Query/Context| normalization/expansion, context builder/compressor  | `src/rag/query/`, `src/rag/context/` |
| Generation   | Ollama HTTP client + grounded prompts + validation   | `src/rag/generation/`            |
| Pipeline     | RAG orchestration                                     | `src/rag/pipeline/rag.py`        |
| Understanding| query understanding (Phase 1) + multi-intent splitting| `src/rag/query/understanding.py`, `multi_intent.py` |
| Cache        | persistent semantic response cache + SQLite store     | `src/rag/cache/`                 |
| Feedback     | rating validation + analytics / FAQ clusters          | `src/rag/feedback/`              |
| Quality      | soft answer-quality scoring                           | `src/rag/quality/validator.py`   |
| Memory       | retrieval-memory hints                                | `src/rag/retrieval/retrieval_memory.py` |
| API          | `/chat` + `/feedback` + `/stats` + `/health`          | `api/server.py`                  |

> See `docs/performance/current_architecture.md` for the detailed post-
> optimization architecture and `docs/performance/optimization_report.md` for
> the BEFORE/AFTER benchmark results.

---

## Setup

### 1. Prerequisites

- Python **3.10+** (tested with 3.11) — a clean virtual environment is
  strongly recommended (the machine global env has conflicting ML packages).
- **Ollama** running with a chat model installed:

  ```powershell
  ollama serve
  ollama pull qwen3-vl:8b        # default (only model installed here)
  # Faster alternative on CPU:
  ollama pull qwen3:4b           # then set OLLAMA_MODEL=qwen3:4b in .env
  ```

### 2. Create a virtual environment and install

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 3. Configure

Copy `.env.example` to `.env` and adjust as needed:

```powershell
Copy-Item .env.example .env
```

Key settings (defaults are sensible for this machine):

| Variable                                                  | Default                            | Notes                                     |
| --------------------------------------------------------- | ---------------------------------- | ----------------------------------------- |
| `OLLAMA_MODEL`                                          | empty / auto                     | First installed model from `OLLAMA_PREFERRED_MODELS`; explicit values must already be installed |
| `OLLAMA_TIMEOUT`                                        | `600`                            | Generation is slow on CPU                 |
| `OLLAMA_THINK`                                          | `false`                          | Disables Qwen3 reasoning tokens           |
| `EMBEDDING_MODEL`                                       | `intfloat/multilingual-e5-small` | 384-dim, ~470 MB, CPU                     |
| `EMBEDDING_QUERY_PREFIX` / `EMBEDDING_PASSAGE_PREFIX` | `query: ` / `passage: `        | e5-style model prefixes                   |
| `TOP_K`                                                 | `8`                              | Retrieved chunks before context assembly  |
| `SIMILARITY_THRESHOLD`                                  | `0.25`                           | Below this, dense-only chunks are filtered out |
| `RERANKER_ENABLED`                                      | `true`                           | `BAAI/bge-reranker-base` (multilingual); set false for low-resource smoke tests |
| `RERANKER_DEVICE` / `RERANKER_BATCH_SIZE`              | `cpu` / `32`                   | Cross-encoder device + batch size (RAM)  |
| `CPU_THREADS`                                          | `0` (auto)                     | Pin torch thread pool on CPU (0=default)|
| `HYBRID_ENABLED` / `HYBRID_FUSION`                      | `true` / `rrf`                 | Dense+BM25 with RRF fusion                |
| `ROUTER_ENABLED` / `ROUTER_CONFIDENCE_THRESHOLD`        | `true` / `0.80`                | Deterministic routing (safe, optional)    |
| `ROUTER_MIN_RESULTS` / `ROUTER_FALLBACK_ENABLED`        | `3` / `true`                  | BROAD fallback protects recall            |
| `DYNAMIC_TOP_K_ENABLED` + `TOP_K_FACT/LIST/COMPLEX`     | `true` / `4/6/8`               | Final chunks vary by complexity           |
| `FINAL_CONTEXT_CHUNKS` / `CONTEXT_MAX_CHARS`            | `4` / `4000`                    | Context sent to the LLM                   |
| `TOP_CONTEXT_CHUNKS`                                    | `6`                             | Hard cap on chunks reaching the LLM (dedup + source diversity applied first) |
| `FAST_PATH_ENABLED` / `FAST_PATH_MIN_CONFIDENCE`        | `true` / `0.55`                | Deterministic no-LLM answers; skipped below this router confidence (e.g. FACULTY 0.53 never auto-answers) |
| `MAX_CONTEXT_TOKENS` / `MAX_GENERATION_TOKENS`          | `4096` / `2500`                 | Token budgets (generation conservative)   |
| `OLLAMA_MAX_OUTPUT_TOKENS`                              | `2500`                          | Hard ceiling for `num_predict`; the pipeline applies smaller per-intent caps for simple answers |
| `CACHE_ENABLED` + `CACHE_*`                             | `true` / 512 / 256 / 3600       | Embedding + retrieval LRU caches          |
| `CHUNK_SIZE` / `CHUNK_OVERLAP`                        | `800` / `100`                  |                                           |
| `CHROMA_COLLECTION`                                     | `nmu_documents`                  |                                           |

> On a non-Windows console always run scripts with UTF-8 output:
>
> ```powershell
> $env:PYTHONIOENCODING='utf-8'
> ```

### 4. Build the index

```powershell
.\.venv\Scripts\python.exe scripts\build_index.py
# or, to force a full rebuild:
.\.venv\Scripts\python.exe scripts\rebuild_index.py
```

The first run downloads the embedding model (~470 MB). Building 2475 chunks
takes about 2 minutes on CPU. A manifest is written to
`vectorstore/index_manifest.json`; repeated runs are idempotent (stable chunk
IDs) and the manifest is checked at query time (`INDEX OUT OF DATE` guard).

---

## Usage

### Chat with the assistant

```powershell
.\.venv\Scripts\python.exe scripts\chat.py          # interactive
.\.venv\Scripts\python.exe scripts\chat.py --debug  # show retrieved chunks
.\.venv\Scripts\python.exe scripts\chat.py -q "What faculties does NMU have?"
```

Type `exit` / `quit` (or `خروج`) to leave. Structured queries (faculties,
location, contact) answer in under a second via the fast path; open-ended
answers take ~1–6 minutes per question on a CPU-only machine with the 8B
model.

### Inspect / validate the dataset

```powershell
.\.venv\Scripts\python.exe scripts\inspect_dataset.py
.\.venv\Scripts\python.exe scripts\validate_dataset.py
```

### Test retrieval quality

```powershell
.\.venv\Scripts\python.exe scripts\test_retrieval.py
```

### Debug one retrieval

Use the diagnostic script when a question retrieves the wrong evidence or a
final answer seems incomplete:

```powershell
$env:PYTHONIOENCODING='utf-8'
.\.venv\Scripts\python.exe scripts\debug_retrieval.py -q "ما هي اقسام وبرامج كلية علوم وهندسة الحاسب"
.\.venv\Scripts\python.exe scripts\debug_retrieval.py --no-reranker -q "كم تبلغ رسوم الكليات السنوية؟"
```

It prints the Chroma collection/version, routed intent/faculty/language, query
variants, every retrieval stage, removed candidates, coverage checks, and the
exact final context sent to generation. The script reads the existing
vectorstore only; it does not rebuild or mutate the knowledge base.

### Run the evaluation

```powershell
# Fast: retrieval metrics only (no LLM)
.\.venv\Scripts\python.exe evaluation\evaluate.py

# Full: grounded answers through the LLM (slow, ~5 min per question on CPU)
.\.venv\Scripts\python.exe evaluation\evaluate.py --mode full --limit 5
```

Reports (JSON + Markdown) are written to `evaluation/results/`.

### Run the unit tests

```powershell
.\.venv\Scripts\python.exe -m pytest tests -v
```

Tests cover filtering, chunking, loading, validation, response formatting,
retrieval dedup, prompt/context assembly, query understanding, multi-intent
splitting, feedback validation and the runtime SQLite store — no Ollama or
model downloads required.

### Export a training dataset (offline, optional)

```powershell
.\.venv\Scripts\python.exe scripts\export_training_data.py
# -> data/training/nmu_training.jsonl
```

Produces a clean JSONL (question, answer, rating, sources, retrieval context)
for a future LoRA / DPO / reranker run. It never trains or fine-tunes the
assistant automatically.

---

## Chunking strategy

- **Content-type aware**: FAQ pages are split into question+answer units;
  news/regulation/faculty pages are split by headings and paragraphs.
- **Consecutive heading-like lines** (e.g. a nav menu of faculty names) are
  grouped into one section so short menu entries are not lost.
- Sections larger than `CHUNK_SIZE` are split at sentence boundaries with
  overlap; character split is the last resort.
- Stable chunk IDs: `sha256(document_id + "::" + index)` — re-indexing never
  creates duplicates.

## Retrieval strategy

- **Hybrid retrieval**: dense cosine (normalized e5 embeddings) fused with
  Okapi BM25 via reciprocal-rank fusion (RRF). The BM25 tokenizer applies a
  light Arabic normalizer (hamza unification, diacritics, `ال`/conjunction
  stripping) so `الشروط` matches `شروط`.
- **Deterministic routing + metadata-first filtering**: the query router
  detects intent/category/faculty/language and confidence. High-confidence
  routes add a Chroma `where` clause (language, content type, faculty) inside
  dense *and* BM25 search, plus a small primary-type boost so the on-topic page
  surfaces above generic pages. If a routed pool is too thin it is **never**
  used as-is — a BROAD fallback re-runs the unfiltered search and merges.
- **Dynamic top-k**: facts → 4 chunks, lists/programs → 6, complex → 8.
- **Text-signature dedup**: mirrored `www` / non-`www` pages collapse to the
  highest-scoring representative even though their raw hashes differ.
- Optional **cross-encoder reranker** (`BAAI/bge-reranker-base`, multilingual)
  scores the question against a larger candidate pool before the final
  ranking. Reranker scores are sigmoid-mapped to [0,1].

## Generation strategy

- **Model selection** (see `OLLAMA_MODEL` / `OLLAMA_PREFERRED_MODELS` in
  `src/rag/config.py`): leave `OLLAMA_MODEL` empty and the first *installed*
  model from the preferred list is used — smaller text models first. On this
  CPU-only laptop only `qwen3-vl:8b` is installed, so it is selected and
  logged; pulling `qwen3:4b` and setting `OLLAMA_MODEL=qwen3:4b` makes
  generation several times faster without touching the knowledge base.
- The model receives a strict grounding prompt: only official NMU context,
  no invented facts/dates/fees/names/URLs, answer in the user's language,
  **no source URLs inside the answer** (they are shown separately), and
  refuse clearly when context is insufficient.
- Generation parameters are configurable and conservative:
  `temperature 0.1`, `top_p 0.9`, `num_ctx 8192`, `num_predict 2500`,
  `keep_alive 30m` (model stays resident — no per-question cold reload).
  `think:false` keeps outputs concise.
- **Intent-aware context sizing**: the prompt size varies by intent
  (`INTENT_CONTEXT_CHUNKS`): 3 chunk groups for simple facts/location/contact,
  5 for admission/regulation, 6 for lists/comparisons — without reducing
  retrieval top-k.
- A **deterministic fast path** answers faculties-list, location and contact
  queries in milliseconds from the authoritative page (no LLM).
- Answers are **validated** after generation: any URL not present in the
  retrieved evidence is removed, and empty/invalid answers become a grounded
  refusal in the user's language.
- Timeouts are handled gracefully: a generation that exceeds `OLLAMA_TIMEOUT`
  returns a typed `generation_timeout` error with actionable advice rather
  than hanging; the timeout is configurable but is **not** silently doubled.

## Runtime/cache separation

- Authoritative knowledge lives in `data/documents.jsonl` and the ChromaDB
  index under `vectorstore/`.
- Runtime state lives in `data/runtime/nmu_runtime.db`: question events,
  feedback, semantic answer cache, retrieval-memory source hints, strategy
  feedback, and FAQ clusters.
- Runtime answers are never inserted into ChromaDB and never become retrieved
  evidence. Semantic cache hits are returned only when knowledge-base version,
  embedding similarity, intent, metadata, and quality gates all pass.
- Use `scripts/reset_runtime.py` to clear cache/feedback/runtime state without
  touching the authoritative dataset or vector index.

## Feedback & learning loop

- Every answered question is recorded (with a `response_id` / legacy
  `question_id`) in the runtime
  SQLite store (`RUNTIME_DB_PATH`, default `data/runtime/nmu_runtime.db`) —
  isolated from ChromaDB and the vector index.
- The GUI shows **Useful / Somewhat / Not useful** buttons under each answer;
  the API endpoint `POST /feedback` validates the rating and optional reason.
- Feedback **never changes model weights**. It is used for: semantic-cache
  quality gating, retrieval-memory quality, strategy-level retrieval/generation
  hints, FAQ clustering / top-failed questions analytics, and preparing an
  offline training dataset.
- `somewhat` and `not_useful` prevent the same semantic-cache entry from
  short-circuiting future requests and also bypass deterministic fast paths for
  the exact normalized query. They also force fresh retrieval so in-process
  retrieval caches cannot replay the same evidence set. Positive exact feedback
  can reuse the approved runtime answer only for the same KB version; similar
  paraphrases still retrieve current evidence. Positive strategy feedback can
  seed trusted source URLs as soft retrieval hints across matching semantic
  groups; negative strategy feedback triggers a broader diversified retrieval
  pass for matching semantic groups.
- `GET /stats` exposes lightweight counters (questions, ratings, cache hits).

## API endpoints

| Endpoint | Purpose |
| -------- | ------- |
| `GET /health`  | Liveness check |
| `POST /chat`   | `{message}` -> `{answer, sources, response_id, question_id, cache_hit, ...}` |
| `POST /feedback` | `{response_id, feedback, reason?}` or legacy `{question_id, rating, reason?}` -> stores a validated rating |
| `GET /stats`   | Runtime counters (questions / ratings / cache) |

## Performance benchmarks

- `evaluation/benchmark/benchmark_retrieval.py [--reranker]` — per-scenario
  retrieval latency + quality (P50/P90/P95).
- `evaluation/benchmark/benchmark_generation.py --n 3` — generation latency.
- `evaluation/benchmark/benchmark_pipeline.py --limit 3` — end-to-end latency
  and answer quality.
- `evaluation/benchmark/benchmark_report.py <json> --baseline evaluation/results/baseline.json`
  renders a Markdown report (with regression flags) into `evaluation/results/`.
- BEFORE/AFTER results: `docs/performance/optimization_report.md`;
  consolidated numbers in `evaluation/results/final_benchmark.json`.

### Measured runtime (this machine, CPU-only)

| Query type                               | Median latency |
| ---------------------------------------- | -------------- |
| Faculties list / location / contact (fast path) | **0.2 s** (warm cache) |
| Factual/regulation answer via qwen3-vl:8b | **~6 min** (generation-bound) |

The dominant cost is LLM generation of `qwen3-vl:8b` on a 4-core laptop CPU
(~1.4 tokens/s). Retrieval is ~2–4 s; the fast path removes generation
entirely for structured queries. To make open-ended answers practical, install
a smaller text model and set `OLLAMA_MODEL`:

```powershell
ollama pull qwen3:4b
# then set OLLAMA_MODEL=qwen3:4b in .env
```

Backend logs record a `[PERF]` line per request
(`model / llm_used / cache_hit / chunk counts / retrieval / bm25 / rerank /
context / fast_path / ollama_generation / total`). Models load **once** at
startup (`[MODEL]`/`[INDEX]`/`[OLLAMA]` lifecycle lines) and stay resident;
the first `/chat` is not penalized by cold-start model loading. These numbers
are measured, not projected.

---

## Troubleshooting

| Problem                                         | Fix                                                                     |
| ----------------------------------------------- | ----------------------------------------------------------------------- |
| `Generation timed out after 600s`               | Generation of the 8B model on CPU is slow. Install a smaller text model (`ollama pull qwen3:4b`) and set `OLLAMA_MODEL=qwen3:4b`, lower `OLLAMA_MAX_OUTPUT_TOKENS`, or raise `OLLAMA_TIMEOUT`. |
| `Model 'qwen3:4b' is not available`             | Explicit `OLLAMA_MODEL` must be installed: `ollama pull qwen3:4b`, then check `ollama list`. Leave `OLLAMA_MODEL` empty to auto-select an installed model. |
| Second request returns `busy` (409)             | Only one generation runs at a time on this machine. Wait for the first answer. |
| GUI says the assistant is "not available"       | The configured model is missing or Ollama is off. Check the API server console log. |
| Empty Arabic output in console                  | `$env:PYTHONIOENCODING='utf-8'` before running.                       |
| Embedding model download fails                  | Retry; or set `HF_HUB_OFFLINE=1` if already cached.                    |
| Index out of date after config change           | Run `scripts/rebuild_index.py`.                                        |
| Global Python has broken `torch`/`ml_dtypes`    | Always use the `.venv` interpreter.                                    |

---

## Roadmap (beyond PHASE 2)

- Voice input/output, vision (page images already exist in the gallery
  records), and a conversational/agent layer on top of the same pipeline.

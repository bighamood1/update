# Runtime & Performance Report (PHASE 2 — second pass)

Targeted, production-quality fixes for the runtime / performance problems
observed with the initial PHASE 2 build on a **CPU-only laptop**. No
destructive changes: the knowledge base, ChromaDB index, and retrieval quality
gates were preserved (see "Measured results" below).

---

## 1. Old bottleneck (measured, not projected)

Machine: Intel i7-1165G7 (4C/8T), 15.7 GB RAM, Intel Iris Xe (no CUDA),
Ollama on `localhost:11434`.

| Step                      | Measured latency |
| ------------------------- | ---------------- |
| Retrieval (hybrid + rerank) | ~2–4 s           |
| LLM generation (qwen3-vl:8b) | ~6.3 min for a normal admission answer |
| Fast-path structured query   | **0.15–0.2 s**   |

The dominant cost was **Ollama generation of `qwen3-vl:8b`** on a 4-core CPU
(~1.4 tokens/s). An unbounded generation can exceed the 600 s read timeout.
RAM is also tight: model + embedder + reranker + OS leave < 1 GB free, so
concurrent generation risks swapping.

## 2. What changed (by problem area)

### Generation speed / timeouts
- **Fast path** (`src/rag/generation/fast_path.py`): faculties-list, location
  and contact queries are answered deterministically from the authoritative
  indexed page (faculties directory, About page, Contact page) with **no LLM
  call**. Strictly gated: it only fires for matching intents when an
  authoritative chunk is retrieved and extraction is non-empty; otherwise the
  request uses normal RAG. Nothing is hardcoded.
- **Graceful timeouts**: `OllamaError` now carries a stable `error_type`.
  A read timeout becomes `generation_timeout` (HTTP 504) with actionable
  advice (faster model / lower output cap / raise timeout). The timeout is
  configurable (`OLLAMA_TIMEOUT`) and is **not** blindly doubled.
- **keep_alive 30m**: the model stays resident, avoiding per-question cold
  reloads that looked like timeouts.

### Model selection (no downloads)
- `OLLAMA_MODEL` empty → auto-select the first **installed** model from
  `OLLAMA_PREFERRED_MODELS` (smaller text models first), logged at startup.
- Explicit `OLLAMA_MODEL` that is not installed → clear `model_unavailable`
  error listing available models + the `ollama pull` command.
- On this machine only `qwen3-vl:8b` is installed, so it is auto-selected.
  `ollama pull qwen3:4b` + `OLLAMA_MODEL=qwen3:4b` is the documented path to
  faster open-ended answers. Nothing is ever downloaded automatically.

### Ollama HTTP / lifecycle
- Persistent `httpx.Client` with split timeouts (connect 10 s / read
  `OLLAMA_TIMEOUT` / write 60 s / pool 10 s). The connectivity/model check is
  cached for 300 s and now performs **one** `/api/tags` call per refresh.
- Heavy components load once per process: embedder, reranker, Chroma, and the
  BM25 index are built lazily and reused by every request.

### Retrieval efficiency (accuracy preserved)
- BM25 index built once per process (lazy) — no per-query rebuild.
- Hybrid retrieval, routing, RRF fusion, BROAD fallback, dynamic top-k and the
  optional reranker are unchanged.
- Verified via `evaluation/evaluate.py` — all gates pass (below).

### Generation parameters (configurable)
- `temperature 0.1`, `top_p 0.9`, `num_ctx 8192`, `num_predict 2500`,
  `keep_alive 30m` — all env-configurable.
- **Intent-aware context sizing** (`INTENT_CONTEXT_CHUNKS`): simple facts /
  location / contact → 3 chunk groups; admission / regulation → 5; lists /
  comparisons → 6. This shrinks the prompt (less generation time) **without**
  reducing the retriever's top-k.

### API behavior (`api/server.py`)
- Structured errors always carry GUI-safe defaults plus the type/message
  (top-level and legacy nested `error`): `{"success": false, "error": true,
  "error_type": "...", "message": "...", "answer": "", "sources": [],
  "error": {"type": "...", "message": "..."}}`.
  Types: `invalid_request` (422), `busy` (409), `connection_error` /
  `model_unavailable` (503), `generation_timeout` (504), `generation_error`
  (502), `backend_error` (503), `unexpected_error` (500). No stack traces are
  ever sent to the GUI; the real cause is logged server-side.
- **Single-generation lock**: only one Ollama generation runs at a time; a
  concurrent request gets an immediate `busy` (409) instead of queuing on a
  deadlock-prone lock.
- `[PERF]` instrumentation per request and at startup (backend logs only).

### GUI (`gui/`)
- Already client-only (pure HTTP; no RAG/Chroma/embeddings/Ollama imports).
- Requests run in a `QThread`; the UI never freezes and shows a loading
  bubble; friendly typed errors (connection / timeout / server / busy /
  model-unavailable) are displayed, never stack traces.
- The `QMessageBox.StandardButton.Clear` bug was already fixed in the earlier
  redesign (Clear Chat uses `Yes | Cancel`); verified no `setMaximumSize`
  overflow warnings at 1366x768/1920x1080/2560x1440 (offscreen test, no Qt
  warnings).
- Answers are rendered clean: inline `[Source 1]` / `Source N:` markers and
  bare URLs are stripped; sources appear only in the "View Sources" panel.

### Answer / source formatting
- Prompts now forbid URLs inside the answer and forbid the
  "Based on the provided context…" boilerplate.
- The LIST fallback answer no longer embeds the source URL in the text.
- The API returns only clean `{title, url}` sources (deduplicated); scores,
  chunk IDs, embeddings, routing and debug data are never exposed to the GUI.

## 3. Files changed / added

| File | Change |
| ---- | ------ |
| `src/rag/config.py` | `OLLAMA_PREFERRED_MODELS`, `OLLAMA_TOP_P`, `INTENT_CONTEXT_CHUNKS`, `FAST_PATH_ENABLED`, aliases `RETRIEVAL_TOP_K` / `ENABLE_RERANKER`, auto-model default |
| `src/rag/generation/ollama_client.py` | `OllamaError.error_type`, `resolve_model()` (auto-select + clear config error), single `/api/tags` per TTL refresh, graceful timeout message |
| `src/rag/generation/fast_path.py` | **new** deterministic fast answers (faculties / location / contact) |
| `src/rag/generation/prompts.py` | no URLs in answers, no boilerplate |
| `src/rag/pipeline/rag.py` | fast-path dispatch, intent-aware context sizing, `[PERF]` logging, URL-free LIST fallback, richer diagnostics |
| `src/rag/context/builder.py` | `build(chunks, max_chunks=, max_chars=)` per-call overrides |
| `api/server.py` | structured errors, single-generation lock, startup validation, `[PERF]` |
| `gui/api_client.py` | structured error parsing + friendly per-type messages |
| `.env.example`, `README.md` | new config keys, model selection, error contract, measured numbers, troubleshooting |
| `tests/test_fast_path.py` | **new** fast-path tests (13) |
| `tests/test_pipeline.py`, `tests/test_ollama_cache.py` | updated for new behavior |

## 4. Measured results (this machine)

### Retrieval quality gate — `evaluation/evaluate.py` (36 questions)
| Metric      | BEFORE | AFTER (this pass) |
| ----------- | ------ | ----------------- |
| hit_rate    | 0.9375 | **0.9688** |
| top1        | 0.5312 | **0.6562** |
| recall@5    | 0.7000 | **0.7667** |
| recall@8    | 0.7167 | **0.7667** |
| MRR         | 0.5528 | **0.5806** |
| source_hit  | 0.7667 | **0.8333** |

All gates pass; no regression.

### Latency (live API, this machine)
- Faculties list (AR): 16.4 s cold → **0.15 s** warm (fast path, no LLM).
- Location (EN): **~2 s** (fast path).
- Contact phone: **~1.9 s** (fast path).
- Open-ended admission answer (qwen3-vl:8b): **~6.3 min** — generation-bound.

### Unit tests
`103 passed` (includes 13 new fast-path tests; no Ollama or downloads).

## 5. Remaining bottleneck

**LLM generation on CPU.** `qwen3-vl:8b` (8.8B) at ~1.4 tokens/s on a 4-core
laptop is a hardware limit — no code change removes it. The fast path removes
it entirely for structured queries; for open-ended answers the practical fix
is a smaller text model (`ollama pull qwen3:4b` + `OLLAMA_MODEL=qwen3:4b`),
which this config fully supports. RAM headroom (< 1 GB free with the 8B model
resident) is the second constraint; disabling the optional reranker
(`RERANKER_ENABLED=false`) frees ~1.1 GB.

## 6. Backward compatibility

- All env variables keep their previous meaning; new ones are additive.
- The old `OLLAMA_MODEL=qwen3-vl:8b` value still works (explicit model must
  exist or startup explains the problem).
- The pipeline public API, CLI scripts and GUI are unchanged.
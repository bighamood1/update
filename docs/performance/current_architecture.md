# Current Architecture — NMU RAG (Phase 2)

> Status: **post-optimization** (RAG performance + retrieval-quality phase).
> All components run locally; no cloud APIs; no new model downloads beyond the
> existing three models (`intfloat/multilingual-e5-small`,
> `BAAI/bge-reranker-base`, `qwen3-vl:8b`).

## 1. Data flow

```
data/documents.jsonl
   │  (never modified; sha256-hashed into the index manifest)
   ▼
ingestion/loader.py  → validation → metadata enrichment (faculty from URL)
   ▼
ingestion/metadata.py  derive_faculty()/enrich()  → faculty + faculty_id
   ▼
chunking/chunker.py  → DocumentChunk (stable chunk_id = sha256(doc_id::index))
   ▼
embeddings/embedder.py  (multilingual-e5-small, 384-dim, LRU cache)
   ▼
vectorstore/store.py  → ChromaDB single collection (nmu_documents, 2475 chunks)
   ▼
index_manifest.json  (dataset hash, embedding model/dim, schema versions)
```

### Query path

```
question
   ▼
routing/router.py        deterministic router (no LLM): intent, category,
                         faculty (longest-alias), language, confidence
   ▼
retrieval/retriever.py   dense (e5) + lexical (BM25, Arabic-normalized)
                         fused via RRF; metadata where-clause for confident
                         routes; BROAD fallback on thin results; dynamic
                         top-k; dedup; optional cross-encoder rerank
   ▼
context/builder.py + context/compressor.py   (grouped by source, token budget)
   ▼
generation/ollama_client.py  persistent httpx client, split timeouts, keep_alive
   ▼
generation/validation.py   fabricated-URL removal + refusal fallback
   ▼
RAGResult {answer, sources, timings, route, diagnostics}
```

## 2. Components

| Component | Path | Responsibility |
|---|---|---|
| Config | `src/rag/config.py` | All knobs from `.env`; conservative defaults |
| Routing | `src/rag/routing/` | `QueryRouter.route()` → `RouteResult` |
| Retriever | `src/rag/retrieval/retriever.py` | Hybrid fusion, filters, fallback, top-k, rerank |
| BM25 | `src/rag/retrieval/bm25.py` | Okapi BM25, light Arabic normalizer, RRF |
| Query norm | `src/rag/query/normalizer.py` | `normalize_query` / `expand_query` / `is_arabic` |
| Context | `src/rag/context/` | `ContextBuilder` + `ContextCompressor` |
| Validation | `src/rag/generation/validation.py` | No fabricated URLs, refusal text |
| Ollama | `src/rag/generation/ollama_client.py` | Persistent client, timeout handling |
| Caching | `src/rag/utils/cache.py` | Thread-safe LRU (embedding + retrieval) |
| Vector store | `src/rag/vectorstore/store.py` | Chroma ops, manifest, `is_built`/`compatibility_errors` |
| Pipeline | `src/rag/pipeline/rag.py` | Orchestration (`RAGPipeline.ask`) |

## 3. Key design decisions

1. **Metadata-first retrieval**: the router emits a high-confidence
   `RouteResult`; the retriever applies a Chroma `where` (language, content
   type, faculty) *inside* vector + BM25 search — never as a Python post-filter
   over the full corpus.
2. **Routing is conservative and never mandatory**: `ROUTER_CONFIDENCE_THRESHOLD`
   (default 0.80); below it no filter is applied. `ROUTER_MIN_RESULTS` (3)
   triggers the **BROAD fallback** that re-runs an unfiltered search and merges,
   so a thin routed pool can never silently destroy recall.
3. **Hybrid retrieval**: dense e5 + Okapi BM25 with a Lucene-style Arabic
   normalizer (hamza unification, diacritic removal, `ال`/conjunction stripping),
   fused via reciprocal-rank fusion (RRF, `RRF_K=60`). BM25 stays inside the
   routed metadata scope so news/home pages do not crowd out the routed content
   type; the BROAD fallback still covers the unfiltered case.
4. **Primary-type boost**: on a confident route, chunks whose content type is
   primary for the intent (e.g. `admission` for ADMISSION) receive a small RRF
   score lift so the actual admission page surfaces above generic pages.
5. **Dynamic top-k**: facts → 4, lists/programs → 6, complex → 8
   (`MAX_CONTEXT_CHUNKS` caps context; `TOP_K_*` cap the final set).
6. **Caching**: `LRUCache` with TTL — embeddings (512 entries) and retrieval
   snapshots (256 entries, keyed by config + manifest signature) so identical
   queries resolve in <0.1s and never re-embed.
7. **Context compression**: `ContextCompressor` enforces
   `MAX_CONTEXT_TOKENS` deterministically (paragraph → sentence trimming, no LLM),
   so a long context never changes facts, only drops redundant material.
8. **Ollama optimizations**: one persistent `httpx.Client` (connection reuse),
   split timeouts (connect 10s / read 600s / write 60s / pool 10s),
   `keep_alive` keeps the model resident, and the `/api/tags` connectivity
   check is cached 300s and invalidated on failure. `num_predict` default is
   conservative (2500) to avoid truncated/empty answers on CPU.
9. **Index versioning**: the manifest records dataset hash, embedding
   model/dim, `chunking_version`, `metadata_schema_version`,
   `retrieval_version`. `VectorStore.compatibility_errors()` raises
   `INDEX OUT OF DATE` on model/dim mismatch so a stale index is never silently
   used; pre-versioning manifests produce warnings only.
10. **No fabricated sources**: `validate_answer` strips any URL not present in
    the retrieved evidence, and returns a controlled refusal in the question's
    language when the answer is empty/invalid.

## 4. Resource controls (`.env`)

- `RERANK_CANDIDATES=20`, `MAX_RETRIEVAL_RESULTS=30`, `MAX_RERANK_RESULTS=20`
- `MAX_CONTEXT_CHUNKS=6`, `MAX_CONTEXT_TOKENS=4096`, `MAX_GENERATION_TOKENS=2500`
- `CACHE_EMBEDDING_SIZE=512`, `CACHE_RETRIEVAL_SIZE=256`, `CACHE_RETRIEVAL_TTL=3600`
- `OLLAMA_TIMEOUT=600`, `OLLAMA_KEEP_ALIVE=5m`
- `ROUTER_*` and `HYBRID_*`, `DYNAMIC_TOP_K_*` toggles

## 5. GUI / API contract

- `api/server.py` exposes `POST /chat` → `{answer, sources}`; `/health`.
- `gui/` talks only to the API; both keep working against the same pipeline.
- Response payload carries `answer`, `sources[]`, `intent`, `timings`,
  `route`, `diagnostics` for observability (`scripts/chat.py --debug`).
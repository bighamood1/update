# Optimization Report — RAG Performance + Retrieval Quality

Phase of work: RAG performance & retrieval-quality optimization over the
existing NMU RAG system. All measurements are **local (CPU, Windows)**, with
`qwen3-vl:8b` via Ollama, `intfloat/multilingual-e5-small` embeddings and
`BAAI/bge-reranker-base` reranking. The dataset (`data/documents.jsonl`) and the
LLM were **not** changed.

## 1. Baseline (BEFORE)

Original dense-only system, old index (2609 chunks, no faculty metadata),
36-question evaluation set (`evaluation/questions.jsonl`):

| metric | BEFORE |
|---|---:|
| hit_rate | 0.9375 |
| top1_hit_rate | 0.5312 |
| recall@5 | 0.7000 |
| recall@8 | 0.7167 |
| mrr | 0.5528 |
| source_hit_rate | 0.7667 |
| mean latency (retrieval) | 4.9 s |

## 2. Optimizations applied

| # | Change | Why |
|---|---|---|
| 1 | Deterministic **query router** (`src/rag/routing/`) | intent/category/faculty/language + confidence without an LLM; enables metadata-first filtering |
| 2 | **Metadata-first retrieval** — Chroma `where` inside dense+BM25 | restrict search space for confident routes; never a Python post-filter |
| 3 | **BROAD fallback** on thin routed pools (`ROUTER_MIN_RESULTS`) | routing can never silently destroy recall |
| 4 | **Hybrid BM25** with **Arabic light normalizer** (`bm25.py`) | `الشروط` now matches `شروط` (hamza/diacritic/`ال` handling); big recall win on Arabic |
| 5 | **Routed BM25 scope** | news/home pages no longer crowd out the routed content type; safety fallback preserved |
| 6 | **Primary-type RRF boost** for confident routes | actual admission/tuition/contact pages surface above generic pages |
| 7 | **Dynamic top-k** (4/6/8) | facts use fewer chunks; lists/complex use more |
| 8 | **Embedding + retrieval LRU cache** with TTL | identical queries resolve <0.1s |
| 9 | **Context compression** to a token budget | keeps prompts bounded, never changes facts |
| 10 | **Index rebuild with faculty metadata** (idempotent, stable chunk IDs, manifest versioning) | enables faculty filtering; `INDEX OUT OF DATE` guard |
| 11 | **Persistent Ollama client** + split timeouts + `keep_alive` | connection reuse; connectivity check cached 300s |
| 12 | **Answer validation** (fabricated-URL removal + refusal) | hallucination control; grounded answers only |
| 13 | Conservative `num_predict` default (2500) | qwen3-vl returns empty content if capped too low |

## 3. Results (AFTER) — same 36-question evaluation set

| metric | BEFORE | AFTER | delta |
|---|---:|---:|---:|
| hit_rate | 0.9375 | 0.9688 | +0.0313 |
| top1_hit_rate | 0.5312 | 0.6562 | +0.1250 |
| recall@5 | 0.7000 | 0.7667 | +0.0667 |
| recall@8 | 0.7167 | 0.7667 | +0.0500 |
| mrr | 0.5528 | 0.5806 | +0.0278 |
| source_hit_rate | 0.7667 | 0.8333 | +0.0666 |
| precision@5 | - | 0.2944 | - |
| mean duplicate rate | - | 0.0 | - |
| mean retrieval latency | 4.9 s | 4.77 s | -0.13 s |

**Quality gates: all PASS** (AFTER ≥ BEFORE on recall@5, recall@8, MRR, source
hit, hit rate, top-1). No fabricated URLs observed in pipeline runs
(`hallucination_rate = 0.0`).

## 4. Benchmark scenarios (12 questions, EN/AR/mixed, incl. no-answer)

Retrieval-only with reranker (`evaluation/benchmark/benchmark_retrieval.py`):

| stat | value |
|---|---:|
| topic_hit_rate | 0.9167 |
| source_hit_rate (scoped, 4 q with gold) | 1.0000 |
| mean recall@5 (scoped) | 0.8750 |
| mean MRR (scoped) | 0.7083 |
| latency mean / p50 / p90 / p95 | 2.80 / 2.24 / 3.51 / 5.58 s |

Generation-only (bounded to 256 tokens): p50 53.3 s, p90 54.0 s, p95 54.1 s.

End-to-end pipeline (3 q): answer_rate 1.0, topic_hit 1.0,
hallucination_rate 0.0, mean_sources 5.0, total p50 ≈ 300 s (dominated by
qwen3-vl:8b generation on CPU; retrieval stage ≈ 2.2-2.8 s of that).

## 5. Notes / limitations

- The 36-question evaluation is the authoritative BEFORE/AFTER comparison; the
  12-question benchmark set adds broader scenario coverage.
- `minimum limits for faculties` gold pages (EN/AR) are absent from
  `data/documents.jsonl` (equivalent content lives in the FAQ page) — a dataset
  limitation, not a retrieval one.
- Latency numbers include CPU-only cross-encoder reranking; with caches enabled,
  repeated identical queries skip embedding and retrieval entirely (<0.1 s).

## 6. Reproducing

```
python scripts/build_index.py        # rebuild (idempotent) with metadata
python scripts/warmup.py             # warm embeddings + model
python evaluation/evaluate.py        # 36-q retrieval eval (same-set gate)
python evaluation/benchmark/benchmark_retrieval.py --reranker
python evaluation/benchmark/benchmark_pipeline.py --limit 3
python evaluation/benchmark/benchmark_generation.py --n 3
python evaluation/benchmark/benchmark_report.py <latest>.json --baseline evaluation/results/baseline.json
python -m pytest tests -q
```
Artifacts: `evaluation/results/final_benchmark.json`,
`evaluation/results/baseline.json`, `evaluation/results/benchmark_*.json|md`.
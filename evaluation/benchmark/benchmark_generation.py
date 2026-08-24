"""Generation benchmark (Phase 24): Ollama generation latency in isolation.

Warms the model, then measures P50 / P90 / P95 generation latency (and token
throughput) for a fixed prompt. Useful to separate model speed from retrieval.

Usage::

    python evaluation/benchmark/benchmark_generation.py [--n 5]
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from rag.generation.ollama_client import OllamaClient
from rag.generation.prompts import SYSTEM_PROMPT
from rag.utils.logging_utils import setup_logging

setup_logging()

RESULTS = ROOT / "evaluation" / "results"

PROMPT = (
    "Answer concisely in English: list the admission requirements and contact "
    "information of New Mansoura University, citing the provided sources."
)


def pct(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    return float(statistics.quantiles(sorted(values), n=100, method="inclusive")[q - 1])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5)
    args = ap.parse_args()

    client = OllamaClient()
    # Warm-up generation (also loads the model into memory).
    print("Warming model ...")
    client.generate(SYSTEM_PROMPT, "Say 'ok'.", num_predict=8)

    samples: list[float] = []
    tokens: list[int] = []
    for i in range(args.n):
        t0 = time.perf_counter()
        out = client.generate(SYSTEM_PROMPT, PROMPT, num_predict=256)
        elapsed = round(time.perf_counter() - t0, 3)
        samples.append(elapsed)
        # Rough token estimate: ~4 chars/token.
        tokens.append(max(1, len(out) // 4))
        print(f"  sample {i + 1}: {elapsed}s, {len(out)} chars")

    report = {
        "mode": "generation",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_samples": len(samples),
        "prompt_chars": len(PROMPT),
        "latency_s": {
            "mean": round(statistics.mean(samples), 3),
            "p50": round(pct(samples, 50), 3),
            "p90": round(pct(samples, 90), 3),
            "p95": round(pct(samples, 95), 3),
        },
        "throughput_tokens_per_s": {
            "mean": round(statistics.mean(tokens) / statistics.mean(samples), 2)
            if samples else None,
        },
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / f"benchmark_generation_{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Wrote", out)
    print(json.dumps(report["latency_s"], indent=2))
    print(json.dumps(report["throughput_tokens_per_s"], indent=2))


if __name__ == "__main__":
    main()
"""Export recorded question/answer pairs for future model training.

This script reads the runtime SQLite store (question_events joined with
feedback) and writes a clean, machine-readable dataset. It does NOT train or
fine-tune anything — it only prepares data so a later LoRA / DPO / reranker
training run can start from a curated file.

Usage::

    python scripts/export_training_data.py                 # -> data/training/nmu_training.jsonl
    python scripts/export_training_data.py --output out.jsonl
    python scripts/export_training_data.py --min-chars 20  # drop tiny answers

Only rows with a non-empty answer are exported. Ratings are kept verbatim
(``useful`` / ``medium`` / ``not_useful`` / ``None``) so downstream tools can
filter on them; the system itself never auto-trains from low ratings.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rag.cache.store import get_runtime_store


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=None, help="Output file (.jsonl)")
    parser.add_argument(
        "--min-chars", type=int, default=1,
        help="Minimum answer length to keep (default 1)",
    )
    args = parser.parse_args()

    rows = get_runtime_store().export_training_rows()
    keep = [r for r in rows if len((r.get("answer") or "").strip()) >= args.min_chars]

    if not keep:
        print("No training rows found yet. Ask questions in the chat first "
              "(ratings are optional).")
        return

    output = Path(args.output) if args.output else ROOT / "data" / "training" / "nmu_training.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as fh:
        for row in keep:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    rated = sum(1 for r in keep if r.get("rating"))
    print(f"Exported {len(keep)} rows -> {output}")
    print(f"  rated rows: {rated}")
    print("  This file is only for offline model training; it is never used "
          "to fine-tune the assistant automatically.")


if __name__ == "__main__":
    main()
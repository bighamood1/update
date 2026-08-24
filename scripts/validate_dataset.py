"""Validate data/documents.jsonl and report structured issues.

- Valid JSONL lines
- Valid/duplicate IDs
- Missing text (non-gallery records only)
- Missing URL, language, content_type
- Malformed metadata
- Duplicate content hashes

Gallery/image records may legitimately lack text and are NOT flagged.
The dataset is never modified.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rag.config import get_config
from rag.ingestion.validator import DatasetValidator
from rag.utils.logging_utils import setup_logging

setup_logging()


def main() -> None:
    cfg = get_config()
    data_path = Path(cfg["data_path"])

    if not data_path.exists():
        print(f"[ERROR] Dataset not found: {data_path}")
        sys.exit(1)

    validator = DatasetValidator(data_path)
    result = validator.validate()

    print("=" * 60)
    print("NMU DATASET VALIDATION REPORT")
    print("=" * 60)
    print(f"Dataset path      : {data_path}")
    print(f"Total records     : {result.total_records}")
    print(f"Malformed lines   : {len(result.malformed_lines)}")
    print(f"Duplicate IDs     : {len(result.duplicate_ids)}")
    print(f"Duplicate hashes  : {len(result.duplicate_hashes)}")
    print(f"Issues (errors)   : {len(result.errors)}")
    print(f"Issues (warnings) : {len(result.warnings)}")
    print()

    if result.malformed_lines:
        print("Malformed lines (first 10):")
        for ln in result.malformed_lines:
            print(f"  line {ln}")

    if result.duplicate_ids:
        print(f"\nDuplicate IDs (first 10): {result.duplicate_ids[:10]}")

    if result.duplicate_hashes:
        print(f"\nDuplicate hashes (first 10): {result.duplicate_hashes[:10]}")

    if result.errors:
        print("\nErrors (first 25):")
        for issue in result.errors[:25]:
            print(f"  [{issue.record_id}] {issue.field}: {issue.message}")
        print(f"  ... {len(result.errors) - min(25, len(result.errors))} more errors")
    else:
        print("\nErrors: none")

    if result.warnings:
        print("\nWarnings (first 15):")
        for issue in result.warnings[:15]:
            print(f"  [{issue.record_id}] {issue.field}: {issue.message}")

    print()
    verdict = "PASS" if result.is_valid() else "FAIL"
    print(f"VERDICT: {verdict}")
    sys.exit(0 if result.is_valid() else 1)


if __name__ == "__main__":
    main()
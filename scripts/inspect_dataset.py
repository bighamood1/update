"""Inspect data/documents.jsonl and print a clear statistical report.

Read-only: never modifies the dataset.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

# Make the src package importable when running from the project root.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rag.config import get_config
from rag.utils.logging_utils import setup_logging

setup_logging()


def _is_arabic(text: str) -> bool:
    return any("\u0600" <= ch <= "\u06FF" for ch in text)


def main() -> None:
    cfg = get_config()
    data_path = Path(cfg["data_path"])

    if not data_path.exists():
        print(f"[ERROR] Dataset not found: {data_path}")
        sys.exit(1)

    total = 0
    textual = 0
    gallery = 0
    arabic = 0
    english = 0
    mixed = 0
    missing_fields = Counter()
    seen_ids = set()
    seen_hashes = set()
    dup_ids = 0
    dup_hashes = 0
    content_types = Counter()
    languages = Counter()
    text_lengths = []

    with data_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue

            rid = rec.get("id")
            if rid in seen_ids:
                dup_ids += 1
            seen_ids.add(rid)

            chash = rec.get("content_hash")
            if chash in seen_hashes:
                dup_hashes += 1
            seen_hashes.add(chash)

            for field in ("id", "text", "title", "language", "content_type", "url"):
                if rec.get(field) in (None, ""):
                    missing_fields[field] += 1

            ct = rec.get("content_type") or "unknown"
            lang = rec.get("language") or "unknown"
            content_types[ct] += 1
            languages[lang] += 1

            text = (rec.get("text") or "").strip()
            if text:
                textual += 1
                text_lengths.append(len(text))
            else:
                if ct in ("gallery", "image", "images"):
                    gallery += 1

            has_ar = _is_arabic(text)
            has_en = any("a" <= ch.lower() <= "z" for ch in text)
            if has_ar and has_en:
                mixed += 1
            elif has_ar:
                arabic += 1
            elif has_en:
                english += 1

    print("=" * 60)
    print("NMU DATASET INSPECTION REPORT")
    print("=" * 60)
    print(f"Dataset path          : {data_path}")
    print(f"Total records         : {total}")
    print(f"Textual records       : {textual}")
    print(f"Gallery/image records : {gallery}")
    print(f"Arabic records        : {arabic}")
    print(f"English records       : {english}")
    print(f"Mixed (ar+en) records : {mixed}")
    print(f"Duplicate IDs         : {dup_ids}")
    print(f"Duplicate hashes      : {dup_hashes}")
    if text_lengths:
        avg = sum(text_lengths) // len(text_lengths)
        print(f"Text length  - avg    : {avg}")
        print(f"Text length  - min    : {min(text_lengths)}")
        print(f"Text length  - max    : {max(text_lengths)}")
    print()
    print("Content types:")
    for ct, n in content_types.most_common():
        print(f"  {ct:20s} {n}")
    print()
    print("Languages:")
    for lang, n in languages.most_common():
        print(f"  {lang:10s} {n}")
    print()
    print("Missing fields:")
    if missing_fields:
        for field, n in missing_fields.most_common():
            print(f"  {field:15s} {n}")
    else:
        print("  none")


if __name__ == "__main__":
    main()
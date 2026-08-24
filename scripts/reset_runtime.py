"""Reset the assistant's runtime state (nmu_runtime.db) to a clean, empty DB.

Clears ALL session-learned / generated data:

- cache_entries          semantic response cache (cached answers)
- question_events        answered questions (includes previous answers)
- feedback               user ratings
- retrieval_memory       learned retrieval source hints
- question_clusters      FAQ clustering / frequency analytics
- kb_versions            version markers

The knowledge base, source documents, crawled data, embeddings and the vector
index are NEVER touched. The schema is preserved; only rows are deleted
(plus AUTOINCREMENT counters reset and VACUUM).

Usage::

    python scripts/reset_runtime.py                 # backup + clear + verify
    python scripts/reset_runtime.py --no-backup     # clear without backup
    python scripts/reset_runtime.py --delete-db     # remove the file (app
                                                    # recreates it empty on
                                                    # startup)
    python scripts/reset_runtime.py --db path.db    # custom path

Run it while the API server is stopped so no new rows are written mid-reset.
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rag.cache.store import RUNTIME_TABLES, get_config, reset_runtime  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default=None, help="Runtime DB path (default: config)")
    parser.add_argument(
        "--no-backup", action="store_true",
        help="Do not create a timestamped backup before clearing",
    )
    parser.add_argument(
        "--delete-db", action="store_true",
        help="Delete the DB file entirely (the app recreates an empty schema "
             "on next startup)",
    )
    args = parser.parse_args()

    path = Path(args.db) if args.db else Path(get_config()["runtime_db_path"])

    if args.delete_db:
        if path.exists():
            stamp = ""
            if not args.no_backup:
                from datetime import datetime
                stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                bak = path.parent / f"nmu_runtime_backup_{stamp}.db"
                import shutil
                shutil.copy2(path, bak)
                print(f"Backup written: {bak}")
            path.unlink()
            print(f"Deleted runtime DB: {path}")
            print("The application recreates an EMPTY schema on startup "
                  "(CREATE TABLE IF NOT EXISTS).")
        else:
            print(f"Runtime DB not found (nothing to do): {path}")
        return 0

    if not path.exists():
        print(f"Runtime DB not found (nothing to do): {path}")

    report = reset_runtime(args.db, backup=not args.no_backup)
    if report.get("error"):
        print(f"[ERROR] reset failed at {report['error']}")
        return 1

    print("=" * 62)
    print("RUNTIME RESET")
    print("=" * 62)
    print("Deleted rows per table:")
    for table, n in report["deleted"].items():
        print(f"  {table:<20} {n}")
    if report.get("backup_path"):
        print(f"Backup: {report['backup_path']}")
    print("\nVerification (rows remaining):")
    for table, n in report["verified"].items():
        print(f"  {table:<20} {n}")
    ok = report["verified_all_empty"]
    print("\n" + ("VERIFIED: all runtime tables are EMPTY." if ok
                  else "FAIL: some runtime tables still have rows!"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
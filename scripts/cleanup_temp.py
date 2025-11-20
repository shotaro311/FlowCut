#!/usr/bin/env python3
"""Clean up stale temp/log files.

Usage:
  python scripts/cleanup_temp.py --days 3 --dry-run
"""
from __future__ import annotations

import argparse
from pathlib import Path

from src.utils.cleanup import cleanup_paths

DEFAULT_PATHS = [
    Path("temp/poc_samples"),
    Path("temp/progress"),
    Path("logs"),
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean temp/progress files older than N days")
    parser.add_argument("--days", type=int, default=3, help="削除対象とする経過日数")
    parser.add_argument("--paths", nargs="*", type=Path, default=None, help="対象パスを上書き")
    parser.add_argument("--dry-run", action="store_true", help="削除せず候補のみ表示")
    args = parser.parse_args()

    targets = args.paths if args.paths else DEFAULT_PATHS
    removed = cleanup_paths(targets, older_than_days=args.days, dry_run=args.dry_run)

    if args.dry_run:
        print(f"[dry-run] {len(removed)} files would be removed")
    else:
        print(f"Removed {len(removed)} files")
    for p in removed:
        print(p)


if __name__ == "__main__":
    main()

"""Utility to clean up stale temporary files and directories."""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Iterable, List


def cleanup_paths(paths: Iterable[Path], *, older_than_days: int = 3, dry_run: bool = False) -> List[Path]:
    """Remove files last modified before the cutoff.

    Args:
        paths: File or directory paths to inspect (non-existing are ignored).
        older_than_days: Files older than this threshold are removed.
        dry_run: If True, do not delete; only return candidates.

    Returns:
        List of paths that were removed (or would be removed in dry_run).
    """

    cutoff_ts = time.time() - older_than_days * 86400
    removed: List[Path] = []

    for target in paths:
        if not target.exists():
            continue
        # Expand directories to their files
        if target.is_dir():
            candidates = list(target.rglob("*"))
        else:
            candidates = [target]

        for p in candidates:
            if p.is_dir():
                continue
            try:
                mtime = p.stat().st_mtime
            except OSError:
                continue
            if mtime < cutoff_ts:
                removed.append(p)
                if not dry_run:
                    try:
                        p.unlink()
                    except OSError:
                        pass

        # remove empty directories
        if target.is_dir() and not dry_run:
            for dirpath in sorted(target.rglob("*"), reverse=True):
                if dirpath.is_dir():
                    try:
                        next(dirpath.iterdir())
                    except StopIteration:
                        dirpath.rmdir()
                    except OSError:
                        pass

    return removed


__all__ = ["cleanup_paths"]

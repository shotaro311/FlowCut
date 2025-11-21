from __future__ import annotations

import os
import time
from pathlib import Path

from src.utils.cleanup import cleanup_paths


def test_cleanup_removes_old_files(tmp_path: Path):
    old_file = tmp_path / "old.txt"
    new_file = tmp_path / "new.txt"
    old_file.write_text("old")
    new_file.write_text("new")

    old_ts = time.time() - 5 * 86400
    os.utime(old_file, (old_ts, old_ts))

    removed = cleanup_paths([tmp_path], older_than_days=3, dry_run=False)

    assert old_file in removed
    assert not new_file in removed
    assert not old_file.exists()
    assert new_file.exists()


def test_cleanup_dry_run(tmp_path: Path):
    f = tmp_path / "file.txt"
    f.write_text("x")
    old_ts = time.time() - 10 * 86400
    os.utime(f, (old_ts, old_ts))

    removed = cleanup_paths([f], older_than_days=1, dry_run=True)
    assert f in removed
    assert f.exists()

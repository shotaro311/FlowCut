"""
Fetch a prebuilt Windows ffmpeg (LGPL) binary and place it at:
  assets/ffmpeg/ffmpeg.exe

This script uses GitHub's API to find the latest release assets from:
  BtbN/FFmpeg-Builds

Why: we want an LGPL build to avoid GPL-only distribution requirements.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path


REPO = "BtbN/FFmpeg-Builds"
API_LATEST = f"https://api.github.com/repos/{REPO}/releases/latest"


def _http_get_json(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "FlowCut-packager"})
    with urllib.request.urlopen(req) as resp:  # nosec - intended network access
        return json.loads(resp.read().decode("utf-8"))


def _download(url: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "FlowCut-packager"})
    with urllib.request.urlopen(req) as resp:  # nosec - intended network access
        with dst.open("wb") as f:
            shutil.copyfileobj(resp, f)


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    out_exe = repo_root / "assets" / "ffmpeg" / "ffmpeg.exe"
    out_licenses = repo_root / "assets" / "ffmpeg" / "licenses"

    release = _http_get_json(API_LATEST)
    assets = release.get("assets", [])
    if not assets:
        print("No assets found in latest release", file=sys.stderr)
        return 1

    # Prefer a win64 LGPL zip asset.
    chosen = None
    for a in assets:
        name = (a.get("name") or "").lower()
        if name.endswith(".zip") and "win64" in name and "lgpl" in name and "shared" not in name:
            chosen = a
            break
    if not chosen:
        # Fallback: any win64 LGPL zip.
        for a in assets:
            name = (a.get("name") or "").lower()
            if name.endswith(".zip") and "win64" in name and "lgpl" in name:
                chosen = a
                break

    if not chosen:
        print("Could not find a win64 LGPL zip asset in latest release", file=sys.stderr)
        return 1

    url = chosen.get("browser_download_url")
    if not url:
        print("Chosen asset is missing browser_download_url", file=sys.stderr)
        return 1

    with tempfile.TemporaryDirectory(prefix="flowcut-ffmpeg-") as td:
        td_path = Path(td)
        zip_path = td_path / "ffmpeg-lgpl-win64.zip"
        _download(url, zip_path)

        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(td_path)

        ffmpeg_path = None
        for p in td_path.rglob("ffmpeg.exe"):
            # Prefer the one under a "bin" directory if present.
            if p.parent.name.lower() == "bin":
                ffmpeg_path = p
                break
            ffmpeg_path = p

        if not ffmpeg_path or not ffmpeg_path.exists():
            print("ffmpeg.exe not found inside downloaded archive", file=sys.stderr)
            return 1

        out_exe.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ffmpeg_path, out_exe)

        # Best-effort: copy license texts if present.
        shutil.rmtree(out_licenses, ignore_errors=True)
        out_licenses.mkdir(parents=True, exist_ok=True)
        for pat in ("COPYING*", "LICENSE*", "README*"):
            for lp in td_path.rglob(pat):
                if lp.is_file() and lp.stat().st_size < 2_000_000:
                    shutil.copy2(lp, out_licenses / lp.name)

    print(f"Wrote: {out_exe}")
    if any(out_licenses.iterdir()):
        print(f"Wrote licenses to: {out_licenses}")
    else:
        print("No license files found in archive (binary still written).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


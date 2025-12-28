#!/usr/bin/env python3
"""Standalone launcher script for the FlowCut Tkinter GUI.

This script exists so that packagers (e.g. PyInstaller) can point to a single
entrypoint that just launches the GUI, without going through the Typer CLI.
"""
from __future__ import annotations

import multiprocessing
import os
import sys
from pathlib import Path


def _setup_ssl_certificates() -> None:
    """PyInstallerバンドル時にSSL証明書のパスを設定する。"""
    if not getattr(sys, "frozen", False):
        return

    meipass = getattr(sys, "_MEIPASS", None)
    if not meipass:
        return

    # certifiのCA証明書ファイルを探す
    cert_file = Path(meipass) / "certifi" / "cacert.pem"
    if cert_file.exists():
        cert_path = str(cert_file)
        os.environ["SSL_CERT_FILE"] = cert_path
        os.environ["REQUESTS_CA_BUNDLE"] = cert_path


def main() -> None:
    """Launch the FlowCut GUI application."""
    # PyInstaller + multiprocessing で子プロセスが再度 GUI を起動しないようにする
    multiprocessing.freeze_support()

    # SSL証明書を設定（PyInstallerバンドル時）
    _setup_ssl_certificates()

    # Ensure project root is on sys.path so that `src` can be imported
    root = Path(__file__).resolve().parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    # PyInstaller バンドル時: ffmpeg などのバイナリが配置されるディレクトリを PATH に追加
    if getattr(sys, "frozen", False):
        # _MEIPASS: PyInstaller がランタイムで展開するディレクトリ（Resources や Frameworks）
        meipass = getattr(sys, "_MEIPASS", None)
        if meipass:
            meipass_path = Path(meipass)
            # ffmpeg_bin サブディレクトリを優先的にPATHに追加
            ffmpeg_bin = meipass_path / "ffmpeg_bin"
            if ffmpeg_bin.exists():
                os.environ["PATH"] = str(ffmpeg_bin) + os.pathsep + os.environ.get("PATH", "")
            os.environ["PATH"] = str(meipass_path) + os.pathsep + os.environ.get("PATH", "")
        # 念のため MacOS ディレクトリと ffmpeg_bin も追加
        bundle_dir = Path(sys.executable).resolve().parent
        ffmpeg_bin_bundle = bundle_dir / "ffmpeg_bin"
        if ffmpeg_bin_bundle.exists():
            os.environ["PATH"] = str(ffmpeg_bin_bundle) + os.pathsep + os.environ.get("PATH", "")
        os.environ["PATH"] = str(bundle_dir) + os.pathsep + os.environ.get("PATH", "")

    # Make relative paths (config/, logs/, output/ etc.) resolve from project root
    os.chdir(root)

    from src.gui.app import run_gui

    run_gui()


if __name__ == "__main__":  # pragma: no cover
    main()

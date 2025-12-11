"""Pathユーティリティ関数群。

ファイル保存時に「既に同名ファイルが存在する場合」は、
`name.ext`, `name (1).ext`, `name (2).ext` ... という
OSやクラウドストレージでよくある連番サフィックス付きの
ファイル名を自動で割り当てる。
"""
from __future__ import annotations

from pathlib import Path


def generate_sequential_path(base_path: Path) -> Path:
    """既存ファイルを上書きせず、連番サフィックス付きパスを返す。

    例:
        base_path = Path("audio.wav")
        - audio.wav が存在しなければ -> audio.wav
        - audio.wav が存在する場合 -> audio (1).wav
        - audio (1).wav も存在する場合 -> audio (2).wav
        ...
    """
    directory = base_path.parent
    stem = base_path.stem
    suffix = base_path.suffix

    # まだ存在しなければ、そのまま採用
    if not base_path.exists():
        return base_path

    index = 1
    while True:
        candidate = directory / f"{stem} ({index}){suffix}"
        if not candidate.exists():
            return candidate
        index += 1


__all__ = ["generate_sequential_path"]


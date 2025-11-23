"""SubtitleSegment から SRT テキストを生成するユーティリティ。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List


def _format_timestamp(seconds: float) -> str:
    total_ms = int(round(max(seconds, 0.0) * 1000))
    ms = total_ms % 1000
    total_sec = total_ms // 1000
    s = total_sec % 60
    total_min = total_sec // 60
    m = total_min % 60
    h = total_min // 60
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


@dataclass(slots=True)
class SubtitleSegment:
    index: int
    start: float
    end: float
    text: str

    def to_srt_block(self) -> str:
        start_ts = _format_timestamp(self.start)
        end_ts = _format_timestamp(self.end)
        return f"{self.index}\n{start_ts} --> {end_ts}\n{self.text}\n"


def segments_to_srt(segments: Iterable[SubtitleSegment]) -> str:
    blocks = [seg.to_srt_block().strip() for seg in segments]
    return "\n\n".join(blocks) + ("\n" if blocks else "")
__all__ = ["SubtitleSegment", "segments_to_srt"]

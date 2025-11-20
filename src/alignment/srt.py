"""AlignedLine から SRT セグメントを生成するユーティリティ。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

from .timestamp import AlignedLine, align_formatted_lines
from src.llm.formatter import FormattedLine
from src.transcribe.base import WordTimestamp


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


def build_segments(aligned_lines: Sequence[AlignedLine]) -> List[SubtitleSegment]:
    segments: List[SubtitleSegment] = []
    for idx, line in enumerate(aligned_lines, start=1):
        segments.append(SubtitleSegment(index=idx, start=line.start, end=line.end, text=line.text))
    return segments


def segments_to_srt(segments: Iterable[SubtitleSegment]) -> str:
    blocks = [seg.to_srt_block().strip() for seg in segments]
    return "\n\n".join(blocks) + ("\n" if blocks else "")


def align_to_srt(
    formatted_lines: Sequence[FormattedLine],
    words: Sequence[WordTimestamp],
    **align_kwargs,
) -> str:
    aligned = align_formatted_lines(formatted_lines, words, **align_kwargs)
    return segments_to_srt(build_segments(aligned))


__all__ = ["SubtitleSegment", "build_segments", "segments_to_srt", "align_to_srt"]

"""LLM向けのwordタイムスタンプ分割ユーティリティ。"""
from __future__ import annotations

from dataclasses import dataclass
import bisect
from typing import List, Sequence

from src.transcribe.base import WordTimestamp


@dataclass(slots=True)
class WordTimeChunk:
    index: int
    start_idx: int
    end_idx: int
    start_sec: float
    end_sec: float


def split_words_into_time_chunks(
    words: Sequence[WordTimestamp],
    *,
    chunk_sec: float = 300.0,
    snap_window_sec: float = 15.0,
    min_gap_sec: float = 0.2,
) -> List[WordTimeChunk]:
    """words を時間ベースで分割し、できるだけ無音ギャップ付近で境界を切る。

    - chunk_sec: 目安のチャンク長（秒）
    - snap_window_sec: 境界探索ウィンドウ（±秒）
    - min_gap_sec: 「区切りとして採用する」最小ギャップ（秒）

    返す start_idx/end_idx は両端inclusive。
    """
    if not words:
        return []

    try:
        chunk_sec = float(chunk_sec)
    except (TypeError, ValueError):
        chunk_sec = 300.0
    if chunk_sec <= 0:
        chunk_sec = 300.0

    try:
        snap_window_sec = float(snap_window_sec)
    except (TypeError, ValueError):
        snap_window_sec = 15.0
    if snap_window_sec < 0:
        snap_window_sec = 0.0

    try:
        min_gap_sec = float(min_gap_sec)
    except (TypeError, ValueError):
        min_gap_sec = 0.2
    if min_gap_sec < 0:
        min_gap_sec = 0.0

    starts = [float(w.start or 0.0) for w in words]
    ends = [float((w.end if w.end is not None else w.start) or 0.0) for w in words]

    chunks: List[WordTimeChunk] = []
    idx = 0
    chunk_index = 1
    n = len(words)

    while idx < n:
        start_idx = idx
        start_sec = starts[start_idx]

        if start_idx == n - 1:
            chunks.append(
                WordTimeChunk(
                    index=chunk_index,
                    start_idx=start_idx,
                    end_idx=start_idx,
                    start_sec=start_sec,
                    end_sec=ends[start_idx],
                )
            )
            break

        target_sec = start_sec + chunk_sec
        if target_sec >= ends[-1]:
            chunks.append(
                WordTimeChunk(
                    index=chunk_index,
                    start_idx=start_idx,
                    end_idx=n - 1,
                    start_sec=start_sec,
                    end_sec=ends[-1],
                )
            )
            break
        approx = bisect.bisect_left(starts, target_sec, lo=start_idx + 1)
        if approx >= n:
            approx = n - 1

        left = bisect.bisect_left(starts, target_sec - snap_window_sec, lo=start_idx + 1)
        right = bisect.bisect_right(starts, target_sec + snap_window_sec, lo=start_idx + 1)

        best_end_idx = None
        best_gap = -1.0
        for i in range(left, min(right, n - 1)):
            gap = starts[i] - ends[i - 1]
            if gap > best_gap:
                best_gap = gap
                best_end_idx = i - 1

        if best_end_idx is None or best_end_idx < start_idx:
            best_end_idx = max(start_idx, approx - 1)

        if best_gap < min_gap_sec:
            best_end_idx = max(start_idx, approx - 1)

        end_idx = min(max(best_end_idx, start_idx), n - 1)

        chunks.append(
            WordTimeChunk(
                index=chunk_index,
                start_idx=start_idx,
                end_idx=end_idx,
                start_sec=start_sec,
                end_sec=ends[end_idx],
            )
        )
        chunk_index += 1
        idx = end_idx + 1

    return chunks


__all__ = ["WordTimeChunk", "split_words_into_time_chunks"]

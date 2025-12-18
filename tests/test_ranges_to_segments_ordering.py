from __future__ import annotations

from src.llm.two_pass import LineRange, TwoPassFormatter
from src.transcribe.base import WordTimestamp


def test_ranges_to_segments_sorts_lines_by_start_idx():
    formatter = TwoPassFormatter.__new__(TwoPassFormatter)
    formatter.fill_gaps = False
    formatter.max_gap_duration = None
    formatter.gap_padding = 0.0
    formatter.start_delay = 0.0

    words = [
        WordTimestamp(word="A", start=0.0, end=1.0),
        WordTimestamp(word="B", start=1.0, end=2.0),
        WordTimestamp(word="C", start=2.0, end=3.0),
        WordTimestamp(word="D", start=3.0, end=4.0),
    ]

    lines = [
        LineRange(start_idx=2, end_idx=3, text="CD"),
        LineRange(start_idx=0, end_idx=1, text="AB"),
    ]

    segments = formatter._ranges_to_segments(words, lines)

    assert [seg.start for seg in segments] == [0.0, 2.0]
    assert [seg.end for seg in segments] == [2.0, 4.0]
    assert [seg.text for seg in segments] == ["AB", "CD"]


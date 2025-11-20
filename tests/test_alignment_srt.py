from __future__ import annotations

from pathlib import Path
import sys

test_dir = Path(__file__).resolve().parent
project_root = test_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.alignment.srt import align_to_srt, build_segments, segments_to_srt
from src.alignment.timestamp import AlignedLine
from src.llm.formatter import FormattedLine
from src.transcribe.base import WordTimestamp


def test_segments_to_srt_formats_timestamp_and_index():
    aligned = [
        AlignedLine(text="設定を開いて", anchor_word="開いて", start=0.0, end=1.2, match_type="exact"),
        AlignedLine(text="くださいね", anchor_word="くださいね", start=1.3, end=2.0, match_type="exact"),
    ]
    segments = build_segments(aligned)
    srt = segments_to_srt(segments)
    expected = "1\n00:00:00,000 --> 00:00:01,200\n設定を開いて\n\n2\n00:00:01,300 --> 00:00:02,000\nくださいね\n"
    assert srt == expected


def test_align_to_srt_runs_alignment_and_outputs_srt():
    words = [
        WordTimestamp(word="hello", start=0.0, end=0.5),
        WordTimestamp(word="world", start=0.6, end=1.1),
    ]
    lines = [FormattedLine(text="wolrd", anchor_word="wolrd", raw="wolrd[WORD: wolrd]", line_number=1)]

    srt = align_to_srt(lines, words)

    assert "00:00:00,000 --> 00:00:01,100" in srt
    assert "wolrd" in srt

from __future__ import annotations

import json
from pathlib import Path
import sys

test_dir = Path(__file__).resolve().parent
project_root = test_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import pytest

from src.alignment.timestamp import align_formatted_lines
from src.llm.formatter import FormattedLine
from src.transcribe.base import WordTimestamp


def test_align_lines_prefers_exact_match_and_keeps_order():
    words = [
        WordTimestamp(word="設定", start=0.0, end=0.4),
        WordTimestamp(word="を", start=0.4, end=0.5),
        WordTimestamp(word="開いて", start=0.5, end=1.2),
        WordTimestamp(word="くださいね", start=1.3, end=1.8),
    ]
    lines = [
        FormattedLine(text="設定を開いて", anchor_word="開いて", raw="設定を開いて[WORD: 開いて]", line_number=1),
        FormattedLine(text="くださいね", anchor_word="くださいね", raw="くださいね[WORD: くださいね]", line_number=2),
    ]

    aligned = align_formatted_lines(lines, words)

    assert [a.match_type for a in aligned] == ["exact", "exact"]
    assert aligned[0].start == pytest.approx(0.0)
    assert aligned[0].end == pytest.approx(1.2)
    assert aligned[1].start == pytest.approx(aligned[0].end + 0.1)
    assert aligned[1].end == pytest.approx(1.8)


def test_align_lines_uses_fuzzy_matching_when_exact_missing():
    words = [
        WordTimestamp(word="hello", start=0.0, end=0.5),
        WordTimestamp(word="world", start=0.6, end=1.1),
    ]
    lines = [
        FormattedLine(text="wolrd", anchor_word="wolrd", raw="wolrd[WORD: wolrd]", line_number=1),
    ]

    aligned = align_formatted_lines(lines, words)

    assert aligned[0].match_type == "fuzzy"
    assert aligned[0].match_score >= 80.0
    assert aligned[0].end == pytest.approx(1.1)


def test_align_lines_logs_warning_on_fallback(tmp_path: Path):
    warning_log = tmp_path / "alignment_warnings.json"
    words = [
        WordTimestamp(word="hello", start=0.0, end=0.5),
        WordTimestamp(word="world", start=0.6, end=1.4),
    ]
    lines = [
        FormattedLine(text="hello", anchor_word="hello", raw="hello[WORD: hello]", line_number=1),
        FormattedLine(text="???", anchor_word="missing", raw="???[WORD: missing]", line_number=2),
    ]

    aligned = align_formatted_lines(lines, words, warning_log=warning_log)

    assert aligned[1].match_type == "fallback"
    assert warning_log.exists()
    payload = json.loads(warning_log.read_text())
    assert payload[0]["line_number"] == 2
    assert payload[0]["anchor_word"] == "missing"
    assert payload[0]["reason"] == "anchor_not_found"

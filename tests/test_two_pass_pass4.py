import json

import pytest

from src.llm.two_pass import TwoPassFormatter
from src.transcribe.base import WordTimestamp


def _words():
    return [
        WordTimestamp(word="これは", start=0.0, end=0.3),
        WordTimestamp(word="とても", start=0.31, end=0.6),
        WordTimestamp(word="ながいながいながい", start=0.61, end=1.0),
        WordTimestamp(word="テキストです", start=1.01, end=1.5),
    ]


def _words_long_duration():
    return [
        WordTimestamp(word="まあ、", start=0.0, end=0.2),
        WordTimestamp(word="でもね、", start=0.21, end=0.5),
        WordTimestamp(word="そうだよね、", start=12.0, end=12.4),
        WordTimestamp(word="です", start=12.41, end=12.8),
    ]


def test_pass4_runs_for_overlength_line(monkeypatch):
    """
    Pass4 が長さ超過行に対してだけ実行され、結果が差し替えられることを確認。
    """
    calls = []

    def fake_call_llm(self, payload: str, model_override=None, pass_label=None):
        calls.append(payload)
        # call order: pass1, pass2, pass3, pass4
        if len(calls) == 1:
            return '{"operations":[]}'
        if len(calls) == 2:
            # pass2: overlength 1行
            return json.dumps(
                {"lines": [{"from": 0, "to": 3, "text": "これはとてもながいながいながいテキストです"}]},
                ensure_ascii=False,
            )
        if len(calls) == 3:
            # pass3: keep same (still overlength)
            return json.dumps(
                {"lines": [{"from": 0, "to": 3, "text": "これはとてもながいながいながいテキストです"}]},
                ensure_ascii=False,
            )
        # pass4: split into 2 lines
        return json.dumps(
            {
                "lines": [
                    {"from": 0, "to": 1, "text": "これはとても"},
                    {"from": 2, "to": 3, "text": "ながいながいながいテキストです"},
                ]
            },
            ensure_ascii=False,
        )

    monkeypatch.setattr("src.llm.validators.detect_issues", lambda lines, words: [])
    monkeypatch.setattr(TwoPassFormatter, "_call_llm", fake_call_llm)

    formatter = TwoPassFormatter(llm_provider="google")
    result = formatter.run(text="dummy", words=_words())

    assert len(calls) == 4  # pass1, pass2, pass3, pass4
    assert result is not None
    assert len(result.segments) == 2
    assert result.segments[0].text == "これはとても"
    assert result.segments[1].text.startswith("ながいながい")


def test_pass4_runs_for_long_duration_line(monkeypatch):
    calls = []

    def fake_call_llm(self, payload: str, model_override=None, pass_label=None):
        calls.append(payload)
        if len(calls) == 1:
            return '{"operations":[]}'
        if len(calls) == 2:
            return json.dumps(
                {"lines": [{"from": 0, "to": 3, "text": "まあ、でもね、そうだよね、です"}]},
                ensure_ascii=False,
            )
        if len(calls) == 3:
            return json.dumps(
                {"lines": [{"from": 0, "to": 3, "text": "まあ、でもね、そうだよね、です"}]},
                ensure_ascii=False,
            )
        return json.dumps(
            {
                "lines": [
                    {"from": 0, "to": 1, "text": "まあ、でもね、"},
                    {"from": 2, "to": 3, "text": "そうだよね、です"},
                ]
            },
            ensure_ascii=False,
        )

    monkeypatch.setattr("src.llm.validators.detect_issues", lambda lines, words: [])
    monkeypatch.setattr(TwoPassFormatter, "_call_llm", fake_call_llm)

    formatter = TwoPassFormatter(llm_provider="google")
    result = formatter.run(text="dummy", words=_words_long_duration())

    assert len(calls) == 4
    assert result is not None
    assert len(result.segments) == 2
    assert result.segments[0].text == "まあ、でもね、"
    assert result.segments[1].text == "そうだよね、です"

import json
from typing import List

import pytest

from src.llm.two_pass import LineRange, TwoPassFormatter
from src.transcribe.base import WordTimestamp


def _words_for_demo() -> List[WordTimestamp]:
    return [
        WordTimestamp(word="テスト", start=0.0, end=0.5),
        WordTimestamp(word="です", start=0.6, end=1.0),
        WordTimestamp(word="よろしく", start=1.1, end=1.6),
        WordTimestamp(word="ね", start=1.7, end=2.0),
    ]


def test_pass3_runs_even_when_no_issues(monkeypatch):
    """
    Pass3 が issues=0 でも実行されることを確認する。
    _call_llm をモックし、呼び出し回数で Pass3 実行を検証する。
    """
    calls: list[str] = []

    def fake_call_llm(self, payload: str, model_override=None, pass_label=None):
        calls.append(payload)
        # Pass1 → operations なし
        if len(calls) == 1:
            return '{"operations":[]}'
        # Pass2 / Pass3 → 同じ行分割を返す
        lines = {"lines": [{"from": 0, "to": 1, "text": "テストです"}, {"from": 2, "to": 3, "text": "よろしくね"}]}
        return json.dumps(lines, ensure_ascii=False)

    # detect_issues を空リスト返却にモック（issues=0ケースを強制）
    monkeypatch.setattr("src.llm.validators.detect_issues", lambda lines, words: [])
    monkeypatch.setattr(TwoPassFormatter, "_call_llm", fake_call_llm)

    formatter = TwoPassFormatter(llm_provider="google")
    result = formatter.run(text="テストです よろしくね", words=_words_for_demo())

    assert result is not None
    assert len(result.segments) == 2
    # Pass1 + Pass2 + Pass3 の3回が呼ばれていること
    assert len(calls) == 3

import json
from typing import List

from src.llm.two_pass import TwoPassFormatter
from src.transcribe.base import WordTimestamp


def _words_for_demo() -> List[WordTimestamp]:
    return [
        WordTimestamp(word="テスト", start=0.0, end=0.5),
        WordTimestamp(word="です", start=0.6, end=1.0),
        WordTimestamp(word="よろしく", start=1.1, end=1.6),
        WordTimestamp(word="ね", start=1.7, end=2.0),
    ]


def test_pass2_exclusive_end_is_normalized(monkeypatch):
    def fake_call_llm(self, payload: str, model_override=None, pass_label=None):
        if pass_label == "pass1":
            return json.dumps({"operations": []}, ensure_ascii=False)
        if pass_label == "pass2":
            return json.dumps(
                {
                    "lines": [
                        {"from": 0, "to": 2, "text": "あああああ"},
                        {"from": 2, "to": 4, "text": "いいいいい"},
                    ]
                },
                ensure_ascii=False,
            )
        if pass_label == "pass3":
            return "not json"
        return json.dumps({"lines": []}, ensure_ascii=False)

    monkeypatch.setattr("src.llm.validators.detect_issues", lambda lines, words: [])
    monkeypatch.setattr(TwoPassFormatter, "_call_llm", fake_call_llm)

    words = _words_for_demo()
    formatter = TwoPassFormatter(llm_provider="google")
    result = formatter.run(text="".join(w.word for w in words), words=words)

    assert result is not None
    assert [seg.text for seg in result.segments] == ["あああああ", "いいいいい"]


def test_pass2_one_based_is_normalized(monkeypatch):
    def fake_call_llm(self, payload: str, model_override=None, pass_label=None):
        if pass_label == "pass1":
            return json.dumps({"operations": []}, ensure_ascii=False)
        if pass_label == "pass2":
            return json.dumps(
                {
                    "lines": [
                        {"from": 1, "to": 2, "text": "あああああ"},
                        {"from": 3, "to": 4, "text": "いいいいい"},
                    ]
                },
                ensure_ascii=False,
            )
        if pass_label == "pass3":
            return "not json"
        return json.dumps({"lines": []}, ensure_ascii=False)

    monkeypatch.setattr("src.llm.validators.detect_issues", lambda lines, words: [])
    monkeypatch.setattr(TwoPassFormatter, "_call_llm", fake_call_llm)

    words = _words_for_demo()
    formatter = TwoPassFormatter(llm_provider="google")
    result = formatter.run(text="".join(w.word for w in words), words=words)

    assert result is not None
    assert [seg.text for seg in result.segments] == ["あああああ", "いいいいい"]

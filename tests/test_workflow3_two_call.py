import json

from src.llm.two_pass import TwoPassFormatter
from src.transcribe.base import WordTimestamp


def _words():
    return [
        WordTimestamp(word="今日は", start=0.0, end=0.2),
        WordTimestamp(word="いい", start=0.2, end=0.4),
        WordTimestamp(word="天気", start=0.4, end=0.6),
        WordTimestamp(word="ですね", start=0.6, end=0.8),
    ]


def test_workflow3_runs_two_call_path(monkeypatch):
    calls: list[tuple[str | None, str | None]] = []

    def fake_call_llm(self, payload: str, model_override=None, pass_label=None):
        calls.append((pass_label, model_override))
        if pass_label == "pass1":
            return json.dumps({"operations": []}, ensure_ascii=False)
        if pass_label == "pass2to4":
            return json.dumps(
                {
                    "lines": [
                        {"from": 0, "to": 3, "text": "今日はいい天気ですね"},
                    ]
                },
                ensure_ascii=False,
            )
        raise AssertionError(f"unexpected pass_label={pass_label}")

    monkeypatch.setattr("src.llm.validators.detect_issues", lambda lines, words: [])
    monkeypatch.setattr(TwoPassFormatter, "_call_llm", fake_call_llm)

    formatter = TwoPassFormatter(llm_provider="google", workflow="workflow3", run_id="test_run")
    result = formatter.run(text="今日はいい天気ですね", words=_words())

    assert result is not None
    assert calls[0][0] == "pass1"
    assert calls[1][0] == "pass2to4"
    assert len(calls) == 2
    assert len(result.segments) == 1
    assert result.segments[0].text == "今日はいい天気ですね"


def test_workflow3_falls_back_to_legacy_on_invalid_combined(monkeypatch):
    calls: list[str | None] = []

    def fake_call_llm(self, payload: str, model_override=None, pass_label=None):
        calls.append(pass_label)
        if pass_label == "pass1":
            return json.dumps({"operations": []}, ensure_ascii=False)
        if pass_label == "pass2to4":
            # invalid: does not cover all words
            return json.dumps(
                {"lines": [{"from": 0, "to": 1, "text": "今日はいい"}]},
                ensure_ascii=False,
            )
        if pass_label in {"pass2", "pass3"}:
            return json.dumps(
                {"lines": [{"from": 0, "to": 3, "text": "今日はいい天気ですね"}]},
                ensure_ascii=False,
            )
        raise AssertionError(f"unexpected pass_label={pass_label}")

    monkeypatch.setattr("src.llm.validators.detect_issues", lambda lines, words: [])
    monkeypatch.setattr(TwoPassFormatter, "_call_llm", fake_call_llm)

    formatter = TwoPassFormatter(llm_provider="google", workflow="workflow3", run_id="test_run")
    result = formatter.run(text="今日はいい天気ですね", words=_words())

    assert result is not None
    assert "pass2to4" in calls
    assert "pass2" in calls
    assert len(result.segments) == 1

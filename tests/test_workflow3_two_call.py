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


def test_workflow3_runs_pass1_pass2_and_skips_pass3(monkeypatch):
    calls: list[tuple[str | None, str | None]] = []

    def fake_call_llm(self, payload: str, model_override=None, pass_label=None):
        calls.append((pass_label, model_override))
        if pass_label == "pass1":
            return json.dumps({"operations": []}, ensure_ascii=False)
        if pass_label == "pass2":
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
    assert calls[1][0] == "pass2"
    assert len(calls) == 2
    assert len(result.segments) == 1
    assert result.segments[0].text == "今日はいい天気ですね"


def test_workflow3_accepts_prefix_lines_and_fills_trailing_coverage(monkeypatch):
    calls: list[str | None] = []

    def fake_call_llm(self, payload: str, model_override=None, pass_label=None):
        calls.append(pass_label)
        if pass_label == "pass1":
            return json.dumps({"operations": []}, ensure_ascii=False)
        if pass_label == "pass2":
            # prefix only: trailing words are filled by local fallback
            return json.dumps({"lines": [{"from": 0, "to": 1, "text": "今日はいい"}]}, ensure_ascii=False)
        raise AssertionError(f"unexpected pass_label={pass_label}")

    monkeypatch.setattr("src.llm.validators.detect_issues", lambda lines, words: [])
    monkeypatch.setattr(TwoPassFormatter, "_call_llm", fake_call_llm)

    formatter = TwoPassFormatter(llm_provider="google", workflow="workflow3", run_id="test_run")
    result = formatter.run(text="今日はいい天気ですね", words=_words())

    assert result is not None
    assert "pass2" in calls
    assert calls == ["pass1", "pass2"]
    assert len(result.segments) >= 1

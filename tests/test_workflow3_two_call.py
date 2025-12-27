"""workflow3のテスト。

workflow3はWhisper + Geminiハイブリッド文字起こしワークフロー。
hybrid_enabled=Trueで、Gemini 3 Flash Previewによる補正が有効。
"""
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


def test_workflow3_runs_standard_4_pass_mode(monkeypatch):
    """workflow3は標準の4パスモードで動作することを確認。"""
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
        if pass_label == "pass3":
            return json.dumps(
                {
                    "lines": [
                        {"from": 0, "to": 3, "text": "今日はいい天気ですね"},
                    ]
                },
                ensure_ascii=False,
            )
        if pass_label == "pass4":
            # pass4は長い行のみ処理するので呼ばれない可能性がある
            return json.dumps({"lines": []}, ensure_ascii=False)
        raise AssertionError(f"unexpected pass_label={pass_label}")

    monkeypatch.setattr("src.llm.validators.detect_issues", lambda lines, words: [])
    monkeypatch.setattr(TwoPassFormatter, "_call_llm", fake_call_llm)

    formatter = TwoPassFormatter(llm_provider="google", workflow="workflow3", run_id="test_run")
    result = formatter.run(text="今日はいい天気ですね", words=_words())

    assert result is not None
    assert calls[0][0] == "pass1"
    assert calls[1][0] == "pass2"
    assert calls[2][0] == "pass3"
    # pass4は長い行がない場合スキップされる
    assert len(result.segments) == 1
    assert result.segments[0].text == "今日はいい天気ですね"


def test_workflow3_hybrid_enabled():
    """workflow3はハイブリッド処理が有効。"""
    from src.llm.workflows.registry import get_workflow

    wf = get_workflow("workflow3")
    assert wf.hybrid_enabled is True
    assert wf.hybrid_thinking_level == "medium"
    assert wf.hybrid_similarity_threshold == 0.8
    assert wf.allow_pass3_range_change is True
    assert wf.pass3_enabled is True

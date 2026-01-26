import json

from src.llm.two_pass_optimized import TwoPassFormatter
from src.transcribe.base import WordTimestamp


def _words_for_demo():
    return [
        WordTimestamp(word="菅", start=0.0, end=0.2),
        WordTimestamp(word="義偉", start=0.2, end=0.5),
        WordTimestamp(word="です", start=0.5, end=0.8),
        WordTimestamp(word="公明党", start=0.8, end=1.1),
        WordTimestamp(word="です", start=1.1, end=1.4),
    ]


def test_workflow2_pass3_prompt_contains_glossary(monkeypatch):
    calls: list[tuple[str | None, str]] = []

    def fake_call_llm(self, payload: str, model_override=None, pass_label=None):
        calls.append((pass_label, payload))
        if pass_label == "pass1":
            return '{"operations":[]}'
        if pass_label in {"pass2", "pass3"}:
            # Pass2/Pass3: 正常な長さの2行（Pass4不要）
            lines = {
                "lines": [
                    {"from": 0, "to": 2, "text": "菅義偉です"},
                    {"from": 3, "to": 4, "text": "公明党です"},
                ]
            }
            return json.dumps(lines, ensure_ascii=False)
        raise AssertionError(f"unexpected pass_label={pass_label}")

    monkeypatch.setattr("src.llm.validators.detect_issues", lambda lines, words: [])
    monkeypatch.setattr(TwoPassFormatter, "_call_llm", fake_call_llm)

    formatter = TwoPassFormatter(
        llm_provider="google",
        workflow="workflow2",
        glossary_terms=["菅義偉", "公明党"],
    )
    result = formatter.run(text="dummy", words=_words_for_demo())

    assert result is not None
    # pass1, pass2, pass3 が実行される
    assert [label for label, _ in calls] == ["pass1", "pass2", "pass3"]

    pass3_prompt = next(payload for label, payload in calls if label == "pass3")
    assert "Glossary" in pass3_prompt
    assert "菅義偉" in pass3_prompt
    assert "公明党" in pass3_prompt
    assert "from/to（範囲）は一切変更しない" in pass3_prompt


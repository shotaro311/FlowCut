from src.llm.workflows.workflow2 import build_pass1_prompt
from src.transcribe.base import WordTimestamp


def test_workflow2_pass1_prompt_includes_political_examples_and_rules():
    words = [
        WordTimestamp(word="石川", start=0.0, end=0.2),
        WordTimestamp(word="です", start=0.2, end=0.5),
    ]
    prompt = build_pass1_prompt("dummy", words)

    assert "明らかな誤変換" in prompt
    assert "有名な固有名詞" in prompt
    assert "石破" in prompt
    assert "参政党" in prompt

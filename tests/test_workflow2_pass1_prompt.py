from src.llm.workflows.workflow2 import build_pass1_prompt
from src.transcribe.base import WordTimestamp


def test_workflow2_pass1_prompt_includes_glossary_and_political_terms():
    words = [
        WordTimestamp(word="菅", start=0.0, end=0.2),
        WordTimestamp(word="義偉", start=0.2, end=0.5),
        WordTimestamp(word="です", start=0.5, end=0.8),
        WordTimestamp(word="参政党", start=0.8, end=1.1),
        WordTimestamp(word="です", start=1.1, end=1.4),
    ]
    prompt = build_pass1_prompt("dummy", words, ["菅義偉", "参政党"])

    assert "Glossary" in prompt
    assert "菅義偉" in prompt
    assert "参政党" in prompt
    assert "政治関連用語" in prompt

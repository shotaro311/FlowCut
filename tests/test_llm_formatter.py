from pathlib import Path
import sys

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.llm.formatter import (  # noqa: E402
    BaseLLMProvider,
    FormatterRequest,
    LLMFormatter,
    FormatValidationError,
    register_provider,
)


@register_provider
class DummyLLMProvider(BaseLLMProvider):
    slug = "test-dummy-llm"
    display_name = "Dummy LLM Provider"

    def format(self, prompt, request):
        assert "17文字" in prompt.user_prompt
        return "設定を開いて[WORD: 開いて]\nくださいね[WORD: くださいね]"


@register_provider
class OverflowLLMProvider(BaseLLMProvider):
    slug = "test-overflow-llm"
    display_name = "Overflow LLM Provider"

    def format(self, prompt, request):
        long_line = "あ" * 20
        return f"{long_line}[WORD: あ]"


@register_provider
class MissingTagLLMProvider(BaseLLMProvider):
    slug = "test-missing-tag-llm"
    display_name = "Missing Tag LLM Provider"

    def format(self, prompt, request):
        return "タグがありません"


def test_formatter_structures_llm_output():
    formatter = LLMFormatter()
    request = FormatterRequest(block_text="設定を開いてください", provider="test-dummy-llm", metadata={"block_id": "B-1"})
    result = formatter.format_block(request)
    assert result.is_valid is True
    assert [line.anchor_word for line in result.lines] == ["開いて", "くださいね"]


def test_formatter_raises_on_length_violation():
    formatter = LLMFormatter()
    request = FormatterRequest(block_text="長文", provider="test-overflow-llm")
    with pytest.raises(FormatValidationError) as exc_info:
        formatter.format_block(request)
    reasons = {issue.reason for issue in exc_info.value.issues}
    assert "exceeds_length" in reasons


def test_formatter_collects_issues_when_non_strict():
    formatter = LLMFormatter(strict_validation=False)
    request = FormatterRequest(block_text="タグなし", provider="test-missing-tag-llm")
    result = formatter.format_block(request)
    reasons = {issue.reason for issue in result.issues}
    assert "missing_word_tag" in reasons
    assert "missing_anchor" in reasons
    assert result.is_valid is False

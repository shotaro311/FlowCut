from __future__ import annotations

from pathlib import Path
import sys

import pytest

test_dir = Path(__file__).resolve().parent
project_root = test_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import src.llm.formatter as fmt
from src.llm.prompts import build_subtitle_prompt
from src.llm.formatter import FormatterRequest, FormatterError


class FlakyProvider(fmt.BaseLLMProvider):
    slug = "flaky"
    display_name = "Flaky Provider"

    def __init__(self, fail_times: int):
        self.fail_times = fail_times
        self.calls = 0

    def format(self, prompt, request):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise FormatterError("mock fail")
        return "ok[WORD: ok]"


def test_llm_formatter_retries_on_failure(monkeypatch):
    provider = FlakyProvider(fail_times=2)
    monkeypatch.setattr(fmt, "get_provider", lambda slug: provider)
    sleep_calls = []
    monkeypatch.setattr(fmt.time, "sleep", lambda s: sleep_calls.append(s))

    formatter = fmt.LLMFormatter(strict_validation=False)
    req = FormatterRequest(block_text="hello", provider="flaky", max_retries=3)
    result = formatter.format_block(req)

    assert provider.calls == 3
    assert sleep_calls == [1, 3]
    assert result.lines[0].text == "ok"


def test_llm_formatter_raises_after_max(monkeypatch):
    provider = FlakyProvider(fail_times=5)
    monkeypatch.setattr(fmt, "get_provider", lambda slug: provider)
    monkeypatch.setattr(fmt.time, "sleep", lambda s: None)

    formatter = fmt.LLMFormatter(strict_validation=False)
    req = FormatterRequest(block_text="hello", provider="flaky", max_retries=2)
    with pytest.raises(FormatterError):
        formatter.format_block(req)

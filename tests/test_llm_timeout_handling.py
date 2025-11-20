from __future__ import annotations

from pathlib import Path
import sys

import requests
import pytest

test_dir = Path(__file__).resolve().parent
project_root = test_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.llm.providers.openai_provider import OpenAIChatProvider
from src.llm.providers.google_provider import GoogleGeminiProvider
from src.llm.providers.anthropic_provider import AnthropicClaudeProvider
from src.llm.prompts import build_subtitle_prompt
from src.llm.formatter import FormatterRequest, FormatterError
from src.config import reload_settings


class RaiseTimeout:
    def __call__(self, *args, **kwargs):
        raise requests.Timeout("mock timeout")


@pytest.mark.parametrize(
    "provider_cls, env_key, env_value",
    [
        (OpenAIChatProvider, "OPENAI_API_KEY", "sk-test"),
        (GoogleGeminiProvider, "GOOGLE_API_KEY", "gg-test"),
        (AnthropicClaudeProvider, "ANTHROPIC_API_KEY", "sk-ant"),
    ],
)
def test_providers_wrap_request_exceptions(monkeypatch, provider_cls, env_key, env_value):
    monkeypatch.setenv(env_key, env_value)
    reload_settings()
    provider_module = provider_cls.__module__
    module = sys.modules[provider_module]
    monkeypatch.setattr(module.requests, "post", RaiseTimeout())
    provider = provider_cls()
    req = FormatterRequest(block_text="hello", provider=provider.slug)
    prompt = build_subtitle_prompt("hello")
    with pytest.raises(FormatterError):
        provider.format(prompt, req)


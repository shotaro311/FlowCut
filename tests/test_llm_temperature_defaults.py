from __future__ import annotations

from pathlib import Path
import sys

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


class DummyResponse:
    def __init__(self, payload: dict, status_code: int = 200):
        self._payload = payload
        self.status_code = status_code
        self.text = "dummy"

    def json(self):
        return self._payload


def test_openai_default_temperature(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    reload_settings()
    payload = {"choices": [{"message": {"content": "ok[WORD: ok]"}}]}

    def fake_post(*args, **kwargs):
        assert kwargs["json"]["temperature"] == 1
        return DummyResponse(payload)

    import src.llm.providers.openai_provider as openai_provider

    monkeypatch.setattr(openai_provider.requests, "post", fake_post)
    provider = OpenAIChatProvider()
    req = FormatterRequest(block_text="hello", provider="openai")
    provider.format(build_subtitle_prompt("hello"), req)


def test_google_default_temperature(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "gg-test")
    reload_settings()
    payload = {"candidates": [{"content": {"parts": [{"text": "ok[WORD: ok]"}]}}]}

    def fake_post(*args, **kwargs):
        assert kwargs["json"]["generationConfig"]["temperature"] == 1
        return DummyResponse(payload)

    import src.llm.providers.google_provider as google_provider

    monkeypatch.setattr(google_provider.requests, "post", fake_post)
    provider = GoogleGeminiProvider()
    req = FormatterRequest(block_text="hello", provider="google")
    provider.format(build_subtitle_prompt("hello"), req)


def test_anthropic_temperature_optional(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant")
    reload_settings()
    payload = {"content": [{"text": "ok[WORD: ok]"}]}

    def fake_post(*args, **kwargs):
        assert "temperature" not in kwargs["json"]
        return DummyResponse(payload)

    import src.llm.providers.anthropic_provider as anthropic_provider

    monkeypatch.setattr(anthropic_provider.requests, "post", fake_post)
    provider = AnthropicClaudeProvider()
    req = FormatterRequest(block_text="hello", provider="anthropic")
    provider.format(build_subtitle_prompt("hello"), req)

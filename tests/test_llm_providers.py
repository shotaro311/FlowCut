from pathlib import Path
import sys

test_dir = Path(__file__).resolve().parent
project_root = test_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import pytest

from src.config import reload_settings
from src.llm.prompts import build_subtitle_prompt
from src.llm.providers.openai_provider import OpenAIChatProvider
from src.llm.providers.google_provider import GoogleGeminiProvider
from src.llm.providers.anthropic_provider import AnthropicClaudeProvider
from src.llm.formatter import FormatterRequest, FormatterError


class DummyResponse:
    def __init__(self, payload: dict, status_code: int = 200):
        self._payload = payload
        self.status_code = status_code
        self.text = "dummy"

    def json(self):
        return self._payload


def test_openai_provider_parses_response(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-test")
    reload_settings()
    payload = {
        "choices": [
            {
                "message": {"content": "設定を開いて[WORD: 開いて]\nくださいね[WORD: くださいね]"}
            }
        ]
    }

    def fake_post(*args, **kwargs):
        return DummyResponse(payload)

    import src.llm.providers.openai_provider as openai_provider

    monkeypatch.setattr(openai_provider.requests, "post", fake_post)
    provider = OpenAIChatProvider()
    prompt = build_subtitle_prompt("設定を開いてください")
    request = FormatterRequest(block_text="設定", provider="openai")
    text = provider.format(prompt, request)
    assert "[WORD: 開いて]" in text


def test_openai_provider_requires_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    reload_settings()
    provider = OpenAIChatProvider()
    prompt = build_subtitle_prompt("設定")
    request = FormatterRequest(block_text="設定", provider="openai")
    with pytest.raises(FormatterError):
        provider.format(prompt, request)


def test_google_provider_parses_response(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "gg-test")
    monkeypatch.setenv("GOOGLE_MODEL", "gemini-test")
    reload_settings()
    payload = {
        "candidates": [
            {
                "content": {"parts": [{"text": "設定を開いて[WORD: 開いて]\nくださいね[WORD: くださいね]"}]}
            }
        ]
    }

    def fake_post(*args, **kwargs):
        return DummyResponse(payload)

    import src.llm.providers.google_provider as google_provider

    monkeypatch.setattr(google_provider.requests, "post", fake_post)
    provider = GoogleGeminiProvider()
    prompt = build_subtitle_prompt("設定を開いてください")
    request = FormatterRequest(block_text="設定", provider="google")
    text = provider.format(prompt, request)
    assert "設定を開いて" in text


def test_google_provider_requires_api_key(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    reload_settings()
    provider = GoogleGeminiProvider()
    prompt = build_subtitle_prompt("設定")
    request = FormatterRequest(block_text="設定", provider="google")
    with pytest.raises(FormatterError):
        provider.format(prompt, request)


def test_google_provider_structured_output_sets_response_mime_and_schema(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "gg-test")
    monkeypatch.setenv("GOOGLE_MODEL", "gemini-test")
    reload_settings()

    schema = {
        "type": "object",
        "properties": {"operations": {"type": "array", "items": {"type": "object"}}},
        "required": ["operations"],
    }
    payload = {"candidates": [{"content": {"parts": [{"text": '{"operations": []}'}]}}]}

    def fake_post(*args, **kwargs):
        gen = kwargs["json"]["generationConfig"]
        assert gen["responseMimeType"] == "application/json"
        assert gen["responseJsonSchema"] == schema
        return DummyResponse(payload)

    import src.llm.providers.google_provider as google_provider

    monkeypatch.setattr(google_provider.requests, "post", fake_post)
    provider = GoogleGeminiProvider()
    prompt = build_subtitle_prompt("設定")
    request = FormatterRequest(
        block_text="設定",
        provider="google",
        metadata={
            "google_response_mime_type": "application/json",
            "google_response_json_schema": schema,
        },
    )
    text = provider.format(prompt, request)
    assert '"operations"' in text


def test_google_provider_structured_output_falls_back_on_error(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "gg-test")
    monkeypatch.setenv("GOOGLE_MODEL", "gemini-test")
    reload_settings()

    schema = {"type": "object", "properties": {"x": {"type": "string"}}, "required": ["x"]}
    ok_payload = {"candidates": [{"content": {"parts": [{"text": "ok[WORD: ok]"}]}}]}

    calls = {"n": 0}

    def fake_post(*args, **kwargs):
        calls["n"] += 1
        gen = kwargs["json"]["generationConfig"]
        if calls["n"] == 1:
            assert gen.get("responseMimeType") == "application/json"
            # Simulate API rejecting structured output.
            resp = DummyResponse({}, status_code=400)
            resp.text = "bad request"
            return resp
        assert "responseMimeType" not in gen
        assert "responseSchema" not in gen
        assert "responseJsonSchema" not in gen
        return DummyResponse(ok_payload)

    import src.llm.providers.google_provider as google_provider

    monkeypatch.setattr(google_provider.requests, "post", fake_post)
    provider = GoogleGeminiProvider()
    prompt = build_subtitle_prompt("設定")
    request = FormatterRequest(
        block_text="設定",
        provider="google",
        metadata={
            "google_response_mime_type": "application/json",
            "google_response_json_schema": schema,
        },
    )
    text = provider.format(prompt, request)
    assert "ok" in text
    assert calls["n"] == 2


def test_anthropic_provider_parses_response(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "ak-test")
    monkeypatch.setenv("ANTHROPIC_MODEL", "claude-test")
    monkeypatch.setenv("ANTHROPIC_API_BASE", "https://api.fake/v1")
    reload_settings()

    payload = {
        "content": [
            {"type": "text", "text": "設定を開いて[WORD: 開いて]\nください"},
        ]
    }

    def fake_post(*args, **kwargs):
        return DummyResponse(payload)

    import src.llm.providers.anthropic_provider as anthropic_provider

    monkeypatch.setattr(anthropic_provider.requests, "post", fake_post)
    provider = AnthropicClaudeProvider()
    prompt = build_subtitle_prompt("設定を開いてください")
    request = FormatterRequest(block_text="設定", provider="anthropic")
    text = provider.format(prompt, request)
    assert "設定を開いて" in text


def test_anthropic_provider_requires_api_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    reload_settings()
    provider = AnthropicClaudeProvider()
    prompt = build_subtitle_prompt("設定")
    request = FormatterRequest(block_text="設定", provider="anthropic")
    with pytest.raises(FormatterError):
        provider.format(prompt, request)

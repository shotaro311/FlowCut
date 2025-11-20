"""OpenAI Chat Completions provider."""
from __future__ import annotations

import json
from typing import Any, Dict

import requests

from src.config import get_settings

from ..formatter import BaseLLMProvider, FormatterError, FormatterRequest, register_provider
from ..prompts import PromptPayload


@register_provider
class OpenAIChatProvider(BaseLLMProvider):
    slug = "openai"
    display_name = "OpenAI Chat Completions"

    def format(self, prompt: PromptPayload, request: FormatterRequest) -> str:
        settings = get_settings().llm
        if not settings.openai_api_key:
            raise FormatterError("OPENAI_API_KEY が未設定です")
        model = request.metadata.get("openai_model") if request.metadata else None
        model = model or settings.openai_model
        payload = {
            "model": model,
            "messages": prompt.as_messages(),
            "temperature": request.temperature if request.temperature is not None else 0.2,
        }
        response = requests.post(
            settings.openai_base_url.rstrip("/") + "/chat/completions",
            headers={
                "Authorization": f"Bearer {settings.openai_api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=settings.request_timeout,
        )
        if response.status_code >= 400:
            raise FormatterError(f"OpenAI API エラー: {response.status_code} {response.text}")
        data: Dict[str, Any] = response.json()
        try:
            content = data["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError) as exc:  # pragma: no cover - defensive
            raise FormatterError(f"OpenAI API 応答を解釈できません: {json.dumps(data)}") from exc
        return content

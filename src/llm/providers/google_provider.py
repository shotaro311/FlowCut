"""Google Gemini provider implementation."""
from __future__ import annotations

import json
from typing import Any, Dict

import requests

from src.config import get_settings

from ..formatter import BaseLLMProvider, FormatterError, FormatterRequest, register_provider
from ..prompts import PromptPayload


@register_provider
class GoogleGeminiProvider(BaseLLMProvider):
    slug = "google"
    display_name = "Google Gemini"

    def format(self, prompt: PromptPayload, request: FormatterRequest) -> str:
        settings = get_settings().llm
        if not settings.google_api_key:
            raise FormatterError("GOOGLE_API_KEY が未設定です")
        model = request.metadata.get("google_model") if request.metadata else None
        model = model or settings.google_model
        endpoint = f"{settings.google_api_base.rstrip('/')}/models/{model}:generateContent"
        payload = {
            "systemInstruction": {"parts": [{"text": prompt.system_prompt}]},
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": prompt.user_prompt}],
                }
            ],
            "generationConfig": {
                # Geminiはデフォルト1を推奨。None時は1を設定。
                "temperature": 1 if request.temperature is None else request.temperature,
            },
        }
        try:
            response = requests.post(
                endpoint,
                params={"key": settings.google_api_key},
                json=payload,
                timeout=settings.request_timeout,
            )
        except requests.RequestException as exc:
            raise FormatterError(f"Google Gemini API リクエストに失敗しました: {exc}") from exc
        if response.status_code >= 400:
            raise FormatterError(f"Google Gemini API エラー: {response.status_code} {response.text}")
        data: Dict[str, Any] = response.json()
        try:
            candidates = data["candidates"]
            content = candidates[0]["content"]["parts"][0]["text"].strip()
        except (KeyError, IndexError) as exc:  # pragma: no cover - defensive
            raise FormatterError(f"Google Gemini API 応答を解釈できません: {json.dumps(data)}") from exc
        return content

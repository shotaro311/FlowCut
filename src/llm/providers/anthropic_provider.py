"""Anthropic Claude Messages provider implementation."""
from __future__ import annotations

import json
from typing import Any, Dict, List

import requests

from src.config import get_settings

from ..formatter import BaseLLMProvider, FormatterError, FormatterRequest, register_provider
from ..prompts import PromptPayload

_ANTHROPIC_VERSION = "2023-06-01"


def _extract_text_blocks(content: Any) -> List[str]:
    texts: List[str] = []
    if not isinstance(content, list):
        return texts
    for block in content:
        if not isinstance(block, dict):
            continue
        text = block.get("text")
        if block.get("type") not in (None, "text"):
            continue
        if isinstance(text, str):
            stripped = text.strip()
            if stripped:
                texts.append(stripped)
    return texts


@register_provider
class AnthropicClaudeProvider(BaseLLMProvider):
    slug = "anthropic"
    display_name = "Anthropic Claude Messages"

    def format(self, prompt: PromptPayload, request: FormatterRequest) -> str:
        settings = get_settings().llm
        if not settings.anthropic_api_key:
            raise FormatterError("ANTHROPIC_API_KEY が未設定です")

        metadata = request.metadata or {}
        model = metadata.get("anthropic_model") or settings.anthropic_model
        try:
            max_tokens = int(metadata.get("anthropic_max_tokens", 1024))
        except (TypeError, ValueError) as exc:
            raise FormatterError("anthropic_max_tokens は数値で指定してください") from exc
        if max_tokens <= 0:
            raise FormatterError("anthropic_max_tokens は正の値で指定してください")

        payload = {
            "model": model,
            "system": prompt.system_prompt,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt.user_prompt,
                        }
                    ],
                }
            ],
            "max_tokens": max_tokens,
        }
        if request.temperature is not None:
            payload["temperature"] = request.temperature

        endpoint = settings.anthropic_api_base.rstrip("/") + "/messages"
        timeout = request.timeout if request.timeout is not None else settings.request_timeout
        try:
            response = requests.post(
                endpoint,
                headers={
                    "x-api-key": settings.anthropic_api_key,
                    "anthropic-version": _ANTHROPIC_VERSION,
                    "content-type": "application/json",
                },
                json=payload,
                timeout=timeout,
            )
        except requests.RequestException as exc:
            raise FormatterError(f"Anthropic API リクエストに失敗しました: {exc}") from exc
        if response.status_code >= 400:
            raise FormatterError(f"Anthropic API エラー: {response.status_code} {response.text}")

        data: Dict[str, Any] = response.json()
        texts = _extract_text_blocks(data.get("content"))
        if texts:
            return "\n".join(texts)
        # 一部のモック/旧仕様では top-level text がある場合がある
        if isinstance(data.get("text"), str) and data["text"].strip():
            return data["text"].strip()
        raise FormatterError(f"Anthropic API 応答を解釈できません: {json.dumps(data)}")


__all__ = ["AnthropicClaudeProvider"]

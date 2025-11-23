"""OpenAI Chat Completions provider."""
from __future__ import annotations

import json
from typing import Any, Dict
import logging

import requests

from src.config import get_settings

from ..formatter import BaseLLMProvider, FormatterError, FormatterRequest, register_provider
from ..prompts import PromptPayload
from ..api_client import post_json_request


@register_provider
class OpenAIChatProvider(BaseLLMProvider):
    slug = "openai"
    display_name = "OpenAI Chat Completions"

    def format(self, prompt: PromptPayload, request: FormatterRequest) -> str:
        logger = logging.getLogger(__name__)
        settings = get_settings().llm
        if not settings.openai_api_key:
            raise FormatterError("OPENAI_API_KEY が未設定です")
        model = request.metadata.get("openai_model") if request.metadata else None
        model = model or settings.openai_model
        payload = {
            "model": model,
            "messages": prompt.as_messages(),
            # gpt-4o-mini 系は temperature=1 が必須。指定が無い場合は1に固定する。
            "temperature": 1 if request.temperature is None else request.temperature,
        }
        timeout = request.timeout if request.timeout is not None else settings.request_timeout
        
        headers = {
            "Authorization": f"Bearer {settings.openai_api_key}",
            "Content-Type": "application/json",
        }

        data = post_json_request(
            url=settings.openai_base_url.rstrip("/") + "/chat/completions",
            payload=payload,
            headers=headers,
            timeout=timeout,
            error_prefix="OpenAI API",
        )
        try:
            content = data["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError) as exc:  # pragma: no cover - defensive
            raise FormatterError(f"OpenAI API 応答を解釈できません: {json.dumps(data)}") from exc

        usage = data.get("usage", {})
        logger.info(
            "llm_usage provider=openai model=%s prompt_tokens=%s completion_tokens=%s total_tokens=%s",
            model,
            usage.get("prompt_tokens"),
            usage.get("completion_tokens"),
            usage.get("total_tokens"),
        )
        return content

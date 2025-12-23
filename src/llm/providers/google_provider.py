"""Google Gemini provider implementation."""
from __future__ import annotations

import json
from typing import Any, Dict
import logging

import requests

from src.config import get_settings

from ..formatter import BaseLLMProvider, FormatterError, FormatterRequest, register_provider
from ..usage_metrics import record_usage_from_request
from ..prompts import PromptPayload
from ..api_client import post_json_request


@register_provider
class GoogleGeminiProvider(BaseLLMProvider):
    slug = "google"
    display_name = "Google Gemini"

    def format(self, prompt: PromptPayload, request: FormatterRequest) -> str:
        logger = logging.getLogger(__name__)
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
        if request.metadata:
            schema = request.metadata.get("structured_output_schema")
            mime_type = request.metadata.get("structured_output_mime_type")
            if schema and mime_type:
                payload["generationConfig"]["response_mime_type"] = mime_type
                payload["generationConfig"]["response_json_schema"] = schema
        timeout = request.timeout if request.timeout is not None else settings.request_timeout
        
        data = post_json_request(
            url=endpoint,
            payload=payload,
            params={"key": settings.google_api_key},
            timeout=timeout,
            error_prefix="Google Gemini API",
        )
        try:
            candidates = data["candidates"]
            content = candidates[0]["content"]["parts"][0]["text"].strip()
        except (KeyError, IndexError) as exc:  # pragma: no cover - defensive
            raise FormatterError(f"Google Gemini API 応答を解釈できません: {json.dumps(data)}") from exc

        usage = data.get("usageMetadata", {})
        prompt_tokens = usage.get("promptTokenCount")
        completion_tokens = usage.get("candidatesTokenCount")
        total_tokens = usage.get("totalTokenCount")
        logger.info(
            "llm_usage provider=google model=%s prompt_tokens=%s completion_tokens=%s total_tokens=%s",
            model,
            prompt_tokens,
            completion_tokens,
            total_tokens,
        )
        # メトリクス集約（run_id / pass_label 単位）
        record_usage_from_request(
            request.metadata,
            provider="google",
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        )
        return content

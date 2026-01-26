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

    def _apply_structured_output_config(self, generation_config: Dict[str, Any], request: FormatterRequest) -> bool:
        """Apply Gemini structured-output config when requested via metadata.

        Returns True if structured-output related fields were applied.
        """
        metadata = request.metadata or {}
        mime_type = metadata.get("google_response_mime_type")
        response_schema = metadata.get("google_response_schema")
        response_json_schema = metadata.get("google_response_json_schema")

        used = False
        if mime_type:
            generation_config["responseMimeType"] = str(mime_type)
            used = True

        # If a schema is provided but mime type is omitted, default to JSON.
        if (response_schema is not None or response_json_schema is not None) and not generation_config.get("responseMimeType"):
            generation_config["responseMimeType"] = "application/json"
            used = True

        if response_schema is not None:
            generation_config["responseSchema"] = response_schema
            used = True
        if response_json_schema is not None:
            generation_config["responseJsonSchema"] = response_json_schema
            used = True

        return used

    def format(self, prompt: PromptPayload, request: FormatterRequest) -> str:
        logger = logging.getLogger(__name__)
        settings = get_settings().llm
        if not settings.google_api_key:
            raise FormatterError("GOOGLE_API_KEY が未設定です")
        model = request.metadata.get("google_model") if request.metadata else None
        model = model or settings.google_model
        endpoint = f"{settings.google_api_base.rstrip('/')}/models/{model}:generateContent"
        generation_config: Dict[str, Any] = {
            # Geminiはデフォルト1を推奨。None時は1を設定。
            "temperature": 1 if request.temperature is None else request.temperature,
        }
        used_structured = self._apply_structured_output_config(generation_config, request)

        payload: Dict[str, Any] = {
            "systemInstruction": {"parts": [{"text": prompt.system_prompt}]},
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": prompt.user_prompt}],
                }
            ],
            "generationConfig": generation_config,
        }
        timeout = request.timeout if request.timeout is not None else settings.request_timeout
        
        def _request(current_payload: Dict[str, Any]) -> Dict[str, Any]:
            return post_json_request(
                url=endpoint,
                payload=current_payload,
                params={"key": settings.google_api_key},
                timeout=timeout,
                error_prefix="Google Gemini API",
            )

        try:
            data = _request(payload)
        except FormatterError as exc:
            # Structured output sometimes fails for specific models/rollouts.
            # Fall back to plain text mode for resiliency.
            if used_structured:
                logger.warning("Google Gemini structured output failed; retrying without responseMimeType/schema: %s", exc)
                fallback_payload = dict(payload)
                fallback_generation = dict(generation_config)
                fallback_generation.pop("responseMimeType", None)
                fallback_generation.pop("responseSchema", None)
                fallback_generation.pop("responseJsonSchema", None)
                fallback_payload["generationConfig"] = fallback_generation
                data = _request(fallback_payload)
            else:
                raise
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

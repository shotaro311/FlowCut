"""openai/whisper ランナー実装。"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import requests

from src.config import get_settings

from .base import (
    BaseTranscribeRunner,
    TranscriptionConfig,
    TranscriptionError,
    TranscriptionResult,
    WordTimestamp,
    register_runner,
)

logger = logging.getLogger(__name__)


@register_runner
class OpenAIWhisperRunner(BaseTranscribeRunner):
    slug = 'openai'
    display_name = 'OpenAI Whisper Large-v3'
    default_model = 'openai/whisper-large-v3'
    requires_gpu = False

    def transcribe(self, audio_path: Path, config: TranscriptionConfig) -> TranscriptionResult:
        if config.simulate:
            return self.simulate_transcription(audio_path, config)

        settings = get_settings().llm
        if not settings.openai_api_key:
            raise TranscriptionError("OPENAI_API_KEY が未設定です")
        if not audio_path.exists():
            raise FileNotFoundError(f"音声ファイルが見つかりません: {audio_path}")

        files = {
            "file": (audio_path.name, audio_path.open("rb")),
        }
        data: Dict[str, Any] = {
            "model": settings.openai_whisper_model,
            "response_format": "verbose_json",
            "timestamp_granularities[]": ["word"],
        }
        if config.language:
            data["language"] = config.language

        try:
            response = requests.post(
                settings.openai_base_url.rstrip("/") + "/audio/transcriptions",
                headers={"Authorization": f"Bearer {settings.openai_api_key}"},
                data=data,
                files=files,
                timeout=settings.request_timeout,
            )
        except requests.RequestException as exc:
            raise TranscriptionError(f"OpenAI Whisper リクエストに失敗しました: {exc}") from exc
        finally:
            try:
                files["file"][1].close()
            except Exception:  # pragma: no cover - best effort
                pass

        if response.status_code >= 400:
            raise TranscriptionError(f"OpenAI Whisper エラー: {response.status_code} {response.text}")

        try:
            payload = response.json()
        except json.JSONDecodeError as exc:  # pragma: no cover - defensive
            raise TranscriptionError(f"OpenAI Whisper 応答をJSONとして解釈できません: {response.text}") from exc

        text = payload.get("text", "")
        words = _parse_words(payload.get("words", []))
        metadata = {
            "runner": self.slug,
            "model": settings.openai_model,
            "audio_file": str(audio_path),
            "simulate": False,
            "language": payload.get("language") or config.language,
        }
        return TranscriptionResult(text=text, words=words, metadata=metadata)


def _parse_words(raw_words: List[Dict[str, Any]]) -> List[WordTimestamp]:
    parsed: List[WordTimestamp] = []
    for item in raw_words or []:
        try:
            parsed.append(
                WordTimestamp(
                    word=str(item.get("word", "")).strip(),
                    start=float(item.get("start")),
                    end=float(item.get("end")),
                    confidence=float(item.get("confidence")) if item.get("confidence") is not None else None,
                )
            )
        except (TypeError, ValueError):
            # skip malformed entries
            continue
    return parsed

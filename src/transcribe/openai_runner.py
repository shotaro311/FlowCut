"""openai/whisper ランナー雛形。"""
from __future__ import annotations

import logging
from pathlib import Path

from .base import BaseTranscribeRunner, TranscriptionConfig, TranscriptionResult, register_runner

logger = logging.getLogger(__name__)


@register_runner
class OpenAIWhisperRunner(BaseTranscribeRunner):
    slug = 'openai'
    display_name = 'OpenAI Whisper Large-v3'
    default_model = 'openai/whisper-large-v3'
    requires_gpu = False

    def prepare(self, config: TranscriptionConfig) -> None:  # pragma: no cover
        if config.simulate:
            logger.debug('[openai] シミュレーションモード: API初期化をスキップ')
            return
        raise NotImplementedError('OpenAI Whisper 連携は今後追加予定です')

    def transcribe(self, audio_path: Path, config: TranscriptionConfig) -> TranscriptionResult:
        if config.simulate:
            return self.simulate_transcription(audio_path, config)
        raise NotImplementedError('OpenAI Whisper 連携は今後追加予定です')

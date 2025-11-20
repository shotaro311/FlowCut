"""kotoba-whisper (mlx) 向けのランナー雛形。"""
from __future__ import annotations

import logging
from pathlib import Path

from .base import BaseTranscribeRunner, TranscriptionConfig, TranscriptionResult, register_runner

logger = logging.getLogger(__name__)


@register_runner
class KotobaRunner(BaseTranscribeRunner):
    slug = 'kotoba'
    display_name = 'Kotoba Whisper v2.0 (MLX)'
    default_model = 'kaiinui/kotoba-whisper-v2.0-mlx'
    requires_gpu = True

    def transcribe(self, audio_path: Path, config: TranscriptionConfig) -> TranscriptionResult:
        if config.simulate:
            return self.simulate_transcription(audio_path, config)
        # 当面は OpenAI Whisper へのフォールバックで実行
        from .openai_runner import transcribe_via_openai_whisper

        logger.info("[kotoba] OpenAI Whisper へフォールバックして実行します")
        return transcribe_via_openai_whisper(
            audio_path,
            config,
            runner_slug=self.slug,
            runner_model=self.default_model,
            metadata_extra={"fallback": "openai_whisper"},
        )

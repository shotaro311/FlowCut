"""mlx-large-v3 ランナー雛形。"""
from __future__ import annotations

import logging
from pathlib import Path

from .base import BaseTranscribeRunner, TranscriptionConfig, TranscriptionResult, register_runner

logger = logging.getLogger(__name__)


@register_runner
class MlxRunner(BaseTranscribeRunner):
    slug = 'mlx'
    display_name = 'MLX Whisper Large-v3'
    default_model = 'mlx-community/whisper-large-v3-mlx'
    requires_gpu = True

    def transcribe(self, audio_path: Path, config: TranscriptionConfig) -> TranscriptionResult:
        if config.simulate:
            return self.simulate_transcription(audio_path, config)
        from .openai_runner import transcribe_via_openai_whisper

        logger.info("[mlx] OpenAI Whisper へフォールバックして実行します")
        return transcribe_via_openai_whisper(
            audio_path,
            config,
            runner_slug=self.slug,
            runner_model=self.default_model,
            metadata_extra={"fallback": "openai_whisper"},
        )

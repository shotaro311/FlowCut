"""kotoba-whisper (mlx) 向けのランナー雛形。"""
from __future__ import annotations

import logging
from pathlib import Path

from .base import BaseTranscribeRunner, TranscriptionConfig, TranscriptionResult, register_runner
from .mlx_common import run_mlx_whisper

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
        return run_mlx_whisper(audio_path, model_id=self.default_model, language=config.language)

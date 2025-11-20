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

    def prepare(self, config: TranscriptionConfig) -> None:  # pragma: no cover
        if config.simulate:
            logger.debug('[mlx] シミュレーションモードのため読み込みをスキップ')
            return
        raise NotImplementedError('mlx whisper 実装は今後追加予定です')

    def transcribe(self, audio_path: Path, config: TranscriptionConfig) -> TranscriptionResult:
        if config.simulate:
            return self.simulate_transcription(audio_path, config)
        raise NotImplementedError('mlx whisper 実装は今後追加予定です')

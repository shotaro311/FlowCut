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

    def prepare(self, config: TranscriptionConfig) -> None:  # pragma: no cover - import確認のみ
        if config.simulate:
            logger.debug('[kotoba] シミュレーションモードのためモデル読込をスキップ')
            return
        raise NotImplementedError('kotoba-mlx 実装は今後追加予定です')

    def transcribe(self, audio_path: Path, config: TranscriptionConfig) -> TranscriptionResult:
        if config.simulate:
            return self.simulate_transcription(audio_path, config)
        raise NotImplementedError('kotoba-mlx 実装は今後追加予定です')

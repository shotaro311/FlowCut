"""kotoba-whisper (mlx) 向けのランナー雛形。"""
from __future__ import annotations

import logging
from pathlib import Path

from .base import TranscriptionConfig, TranscriptionResult, register_runner
from .mlx_common import BaseMlxRunner

logger = logging.getLogger(__name__)


@register_runner
class KotobaRunner(BaseMlxRunner):
    slug = 'kotoba'
    display_name = 'Kotoba Whisper v2.0 (MLX)'
    default_model = 'kaiinui/kotoba-whisper-v2.0-mlx'

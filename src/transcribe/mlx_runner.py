"""mlx-large-v3 ランナー雛形。"""
from __future__ import annotations

import logging
from pathlib import Path

from .base import TranscriptionConfig, TranscriptionResult, register_runner
from .mlx_common import BaseMlxRunner

logger = logging.getLogger(__name__)


@register_runner
class MlxRunner(BaseMlxRunner):
    slug = 'mlx'
    display_name = 'MLX Whisper Large-v3'
    default_model = 'mlx-community/whisper-large-v3-mlx'

"""transcribe パッケージの公開API。"""
from __future__ import annotations

import sys

from .base import (
    BaseTranscribeRunner,
    TranscriptionConfig,
    TranscriptionResult,
    TranscriptionError,
    RunnerNotFoundError,
    WordTimestamp,
    available_runners,
    describe_runners,
    get_runner,
    register_runner,
)

# ランナーをインポートしてレジストリ登録を実行
from . import whisper_local_runner as _whisper_local_runner  # noqa: F401
if sys.platform == "darwin":
    from . import mlx_runner as _mlx_runner  # noqa: F401
else:
    from . import faster_whisper_runner as _faster_whisper_runner  # noqa: F401

__all__ = [
    'BaseTranscribeRunner',
    'TranscriptionConfig',
    'TranscriptionResult',
    'TranscriptionError',
    'RunnerNotFoundError',
    'WordTimestamp',
    'available_runners',
    'describe_runners',
    'get_runner',
    'register_runner',
]

"""transcribe パッケージの公開API。"""
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
from . import mlx_runner as _mlx_runner  # noqa: F401
from . import openai_runner as _openai_runner  # noqa: F401
from . import kotoba_runner as _kotoba_runner  # noqa: F401
from . import whisper_local_runner as _whisper_local_runner  # noqa: F401
from . import whisper_timestamped_runner as _whisper_timestamped_runner  # noqa: F401

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

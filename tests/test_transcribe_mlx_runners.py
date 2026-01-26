from __future__ import annotations

from pathlib import Path
import sys

import pytest

test_dir = Path(__file__).resolve().parent
project_root = test_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.transcribe.mlx_runner import MlxRunner
from src.transcribe.base import TranscriptionConfig, TranscriptionError


def _patch_mlx(monkeypatch):
    import types

    fake_mod = types.SimpleNamespace()

    def fake_transcribe(path, path_or_hf_repo, word_timestamps, language, **kwargs):
        return {
            "text": "dummy text",
            "segments": [
                {"words": [{"word": "hello", "start": 0.0, "end": 0.5, "probability": 0.9}]},
                {"words": [{"word": "world", "start": 0.6, "end": 1.0, "probability": 0.8}]},
            ],
            "language": language,
        }

    fake_mod.transcribe = fake_transcribe
    monkeypatch.setitem(sys.modules, "mlx_whisper", fake_mod)


def test_mlx_runner_uses_native(monkeypatch, tmp_path: Path):
    audio = tmp_path / "audio.m4a"
    audio.write_bytes(b"dummy")
    _patch_mlx(monkeypatch)

    runner = MlxRunner()
    result = runner.transcribe(audio, TranscriptionConfig(simulate=False, language="ja"))
    assert result.text == "dummy text"
    assert len(result.words) == 2
    assert result.metadata["model"] == runner.default_model


def test_mlx_runners_require_dependency(monkeypatch, tmp_path: Path):
    audio = tmp_path / "audio.m4a"
    audio.write_bytes(b"dummy")
    monkeypatch.delitem(sys.modules, "mlx_whisper", raising=False)

    runner = MlxRunner()
    with pytest.raises(TranscriptionError):
        runner.transcribe(audio, TranscriptionConfig(simulate=False))

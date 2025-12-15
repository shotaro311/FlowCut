from __future__ import annotations

from pathlib import Path
import importlib
import sys

import pytest

test_dir = Path(__file__).resolve().parent
project_root = test_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.transcribe.openai_runner import OpenAIWhisperRunner
from src.transcribe.base import TranscriptionConfig, TranscriptionError


def test_openai_runner_parses_words(monkeypatch, tmp_path: Path):
    audio = tmp_path / "audio.m4a"
    audio.write_bytes(b"dummy")

    class DummyModel:
        def transcribe(self, *_args, **kwargs):
            assert kwargs.get("word_timestamps") is True
            return {
                "text": "hello world",
                "segments": [
                    {
                        "words": [
                            {"word": "hello", "start": 0.0, "end": 0.5, "confidence": 0.9},
                            {"word": "world", "start": 0.6, "end": 1.0, "confidence": 0.8},
                        ],
                    }
                ],
                "language": kwargs.get("language") or "ja",
            }

    class DummyWhisperModule:
        def load_model(self, model_name: str):
            assert model_name == "large-v3"
            return DummyModel()

    import src.transcribe.openai_runner as runner_mod

    runner_mod._MODEL_CACHE.clear()
    monkeypatch.setattr(importlib, "import_module", lambda name: DummyWhisperModule())

    runner = OpenAIWhisperRunner()
    result = runner.transcribe(audio, TranscriptionConfig(simulate=False, language="ja"))
    assert result.text == "hello world"
    assert len(result.words) == 2
    assert result.words[0].word == "hello"
    assert result.metadata["model"]


def test_openai_runner_requires_whisper_dependency(monkeypatch, tmp_path: Path):
    audio = tmp_path / "audio.m4a"
    audio.write_bytes(b"dummy")
    import src.transcribe.openai_runner as runner_mod

    runner_mod._MODEL_CACHE.clear()

    def missing_module(_name: str):
        raise ImportError("no whisper here")

    monkeypatch.setattr(importlib, "import_module", missing_module)
    runner = OpenAIWhisperRunner()
    with pytest.raises(TranscriptionError):
        runner.transcribe(audio, TranscriptionConfig(simulate=False))

from __future__ import annotations

from pathlib import Path
import sys

import pytest

test_dir = Path(__file__).resolve().parent
project_root = test_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.transcribe.openai_runner import OpenAIWhisperRunner
from src.transcribe.base import TranscriptionConfig, TranscriptionError
from src.config import reload_settings


class DummyResponse:
    def __init__(self, payload: dict, status_code: int = 200, text: str = "ok"):
        self._payload = payload
        self.status_code = status_code
        self.text = text

    def json(self):
        return self._payload


def test_openai_runner_parses_words(monkeypatch, tmp_path: Path):
    audio = tmp_path / "audio.m4a"
    audio.write_bytes(b"dummy")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("OPENAI_WHISPER_MODEL", "whisper-test")
    reload_settings()
    payload = {
        "text": "hello world",
        "words": [
            {"word": "hello", "start": 0.0, "end": 0.5, "confidence": 0.9},
            {"word": "world", "start": 0.6, "end": 1.0, "confidence": 0.8},
        ],
    }

    def fake_post(*args, **kwargs):
        # ensure correct endpoint and payload fields
        assert "audio/transcriptions" in args[0]
        assert kwargs["data"]["model"] == "whisper-test"
        return DummyResponse(payload)

    import src.transcribe.openai_runner as runner_mod

    monkeypatch.setattr(runner_mod.requests, "post", fake_post)
    runner = OpenAIWhisperRunner()
    result = runner.transcribe(audio, TranscriptionConfig(simulate=False, language="ja"))
    assert result.text == "hello world"
    assert len(result.words) == 2
    assert result.words[0].word == "hello"
    assert result.metadata["model"]


def test_openai_runner_requires_api_key(monkeypatch, tmp_path: Path):
    audio = tmp_path / "audio.m4a"
    audio.write_bytes(b"dummy")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    reload_settings()
    runner = OpenAIWhisperRunner()
    with pytest.raises(TranscriptionError):
        runner.transcribe(audio, TranscriptionConfig(simulate=False))

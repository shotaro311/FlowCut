from __future__ import annotations

from pathlib import Path
import sys

import pytest

test_dir = Path(__file__).resolve().parent
project_root = test_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.transcribe.kotoba_runner import KotobaRunner
from src.transcribe.mlx_runner import MlxRunner
from src.transcribe.base import TranscriptionConfig
from src.config import reload_settings


class DummyResponse:
    def __init__(self, payload: dict, status_code: int = 200):
        self._payload = payload
        self.status_code = status_code
        self.text = "ok"

    def json(self):
        return self._payload


def _patch_openai(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("OPENAI_WHISPER_MODEL", "whisper-fb")
    reload_settings()
    payload = {
        "text": "dummy",
        "words": [{"word": "dummy", "start": 0.0, "end": 0.1}],
    }

    def fake_post(*args, **kwargs):
        assert kwargs["data"]["model"] == "whisper-fb"
        return DummyResponse(payload)

    import src.transcribe.openai_runner as openai_runner

    monkeypatch.setattr(openai_runner.requests, "post", fake_post)


@pytest.mark.parametrize("runner_cls", [KotobaRunner, MlxRunner])
def test_fallback_runners_use_openai(monkeypatch, tmp_path: Path, runner_cls):
    audio = tmp_path / "audio.m4a"
    audio.write_bytes(b"dummy")
    _patch_openai(monkeypatch)

    runner = runner_cls()
    result = runner.transcribe(audio, TranscriptionConfig(simulate=False))
    assert result.text == "dummy"
    assert result.metadata["fallback"] == "openai_whisper"
    assert result.metadata["api_model"] == "whisper-fb"

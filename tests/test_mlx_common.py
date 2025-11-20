from __future__ import annotations

from pathlib import Path
import sys

import pytest

test_dir = Path(__file__).resolve().parent
project_root = test_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


def test_run_mlx_whisper_parses_words(monkeypatch, tmp_path: Path):
    audio = tmp_path / "a.wav"
    audio.write_text("dummy")

    fake_mod = pytest.importorskip("types").SimpleNamespace()

    def fake_transcribe(path_or_file, path_or_hf_repo, word_timestamps, language):
        assert path_or_hf_repo == "test-model"
        return {
            "text": "hello world",
            "segments": [
                {"words": [{"word": "hello", "start": 0.0, "end": 0.5, "probability": 0.9}]},
                {"words": [{"word": "world", "start": 0.6, "end": 1.0, "probability": 0.8}]},
            ],
        }

    fake_mod.transcribe = fake_transcribe
    monkeypatch.setitem(sys.modules, "mlx_whisper", fake_mod)

    from src.transcribe.mlx_common import run_mlx_whisper

    result = run_mlx_whisper(audio, model_id="test-model", language="ja")
    assert result.text == "hello world"
    assert len(result.words) == 2
    assert result.words[0].word == "hello"


def test_run_mlx_whisper_requires_dependency(monkeypatch, tmp_path: Path):
    audio = tmp_path / "a.wav"
    audio.write_text("dummy")
    monkeypatch.setitem(sys.modules, "mlx_whisper", None)
    monkeypatch.delitem(sys.modules, "mlx_whisper", raising=False)

    from src.transcribe.mlx_common import run_mlx_whisper, TranscriptionError

    with pytest.raises(TranscriptionError):
        run_mlx_whisper(audio, model_id="test", language=None)

from __future__ import annotations

from pathlib import Path
import sys

import typer
from typer.testing import CliRunner

test_dir = Path(__file__).resolve().parent
project_root = test_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.cli.main import app

runner = CliRunner()


def test_align_options_are_parsed(monkeypatch, tmp_path: Path):
    # prepare dummy execution path by stubbing execute_poc_run
    called = {}

    def fake_execute(audio_files, model_slugs, options):
        called["align_kwargs"] = options.align_kwargs
        called["models"] = model_slugs
        called["audio"] = audio_files

    monkeypatch.setattr("src.cli.main.execute_poc_run", fake_execute)
    audio = tmp_path / "a.m4a"
    audio.write_text("x")

    result = runner.invoke(
        app,
        [
            "run",
            str(audio),
            "--llm",
            "openai",
            "--align-thresholds",
            "92,85",
            "--align-gap",
            "0.2",
            "--align-fallback-padding",
            "0.4",
        ],
    )

    assert result.exit_code == 0
    assert called["align_kwargs"] == {
        "fuzzy_thresholds": [92, 85],
        "gap_seconds": 0.2,
        "fallback_padding": 0.4,
    }


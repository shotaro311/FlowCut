from __future__ import annotations

from pathlib import Path
import sys

from typer.testing import CliRunner

test_dir = Path(__file__).resolve().parent
project_root = test_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.cli.main import app
import src.llm.formatter as fmt

runner = CliRunner()


class DummyProvider(fmt.BaseLLMProvider):
    slug = "dummy"
    display_name = "Dummy Provider"
    def __init__(self):
        self.calls = 0

    def format(self, prompt, request):
        # Pass1: operations / Pass2+: lines
        if "operations" in prompt.user_prompt:
            return '{"operations": []}'
        # Pass2 / Pass3 どちらでも同じレスポンスでOK
        return '{"lines":[{"from":0,"to":0,"text":"こんにちは世界"}]}'


def test_cli_run_simulate_generates_srt(monkeypatch, tmp_path: Path):
    # register dummy provider and patch get_provider
    provider = DummyProvider()
    # ダミー登録を追加
    fmt.register_provider(DummyProvider)
    monkeypatch.setattr(fmt, "get_provider", lambda slug: provider)
    # ensure runners write under temp dirs
    audio = tmp_path / "a.m4a"
    audio.write_text("dummy-audio")

    monkeypatch.chdir(tmp_path)

    result = runner.invoke(
        app,
        [
            "run",
            str(audio),
            "--models",
            "whisper-local",
            "--llm",
            "dummy",
            "--simulate",
            "--output-dir",
            str(tmp_path / "out"),
            "--progress-dir",
            str(tmp_path / "prog"),
        ],
    )
    assert result.exit_code == 0, result.output
    srt_files = list((Path("output")).glob("*.srt")) + list((tmp_path / "out").glob("*.srt"))
    # default subtitle_dir is output/ (relative cwd set to tmp_path)
    assert srt_files, "SRT should be generated in simulate mode with dummy LLM"

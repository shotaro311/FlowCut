from pathlib import Path
import sys

from typer.testing import CliRunner

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.cli.main import app  # noqa: E402  # isort:skip
from src.utils.progress import (  # noqa: E402  # isort:skip
    BlockProgress,
    ProgressRecord,
    load_progress,
    save_progress,
)


def _create_resume_file(base_dir: Path) -> Path:
    audio_path = base_dir / "sample_audio.m4a"
    audio_path.write_text("dummy audio data")

    record = ProgressRecord(
        run_id="sample_audio_openai_20251120T120000",
        audio_file=str(audio_path),
        model="openai",
        total_blocks=1,
        llm_provider=None,
        status="failed",
        created_at="2025-11-20T12:00:00",
        updated_at="2025-11-20T12:00:00",
        completed_blocks=0,
        blocks=[BlockProgress(index=1, status="pending")],
        metadata={
            "requested_at": "20251120T120000",
            "options": {
                "language": "ja",
                "chunk_size": None,
                "simulate": True,
                "llm_provider": None,
                "rewrite": False,
                "llm_temperature": None,
                "llm_timeout": None,
            },
        },
    )
    progress_path = base_dir / "progress.json"
    save_progress(record, progress_path)
    return progress_path


def test_cli_run_with_resume_smoke(tmp_path):
    progress_path = _create_resume_file(tmp_path)
    output_dir = tmp_path / "out"
    progress_dir = tmp_path / "prog"

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "run",
            "--resume",
            str(progress_path),
            "--output-dir",
            str(output_dir),
            "--progress-dir",
            str(progress_dir),
        ],
    )

    assert result.exit_code == 0, f"CLI failed: {result.stdout}"

    saved_json = list(output_dir.glob("*.json"))
    assert saved_json, "出力JSONが生成されていません"
    assert "sample_audio" in saved_json[0].stem

    progress_files = list(progress_dir.glob("*.json"))
    assert progress_files, "進捗ファイルが生成されていません"
    final_record = load_progress(progress_files[0])
    assert final_record.status == "completed"

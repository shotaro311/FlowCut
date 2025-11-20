from pathlib import Path
import sys

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline.poc import (  # noqa: E402
    PocRunOptions,
    ResumeCompletedError,
    prepare_resume_run,
)
from src.utils.progress import (  # noqa: E402
    BlockProgress,
    ProgressRecord,
    save_progress,
)


def create_progress(tmp_path: Path, *, status: str = "failed") -> Path:
    audio_path = tmp_path / "sample.wav"
    audio_path.write_text("dummy")
    record = ProgressRecord(
        run_id="sample_kotoba_20251120T173552",
        audio_file=str(audio_path),
        model="kotoba",
        total_blocks=2,
        llm_provider=None,
        status=status,
        created_at="2025-11-20T17:35:52",
        updated_at="2025-11-20T17:35:52",
        completed_blocks=1,
        blocks=[
            BlockProgress(index=1, status="completed", started_at="2025-11-20T17:35:52", completed_at="2025-11-20T17:35:53"),
            BlockProgress(index=2, status="pending"),
        ],
        metadata={
            "requested_at": "20251120T173552",
            "options": {"language": "ja", "chunk_size": 1024, "simulate": False},
        },
    )
    progress_path = tmp_path / "progress.json"
    save_progress(record, progress_path)
    return progress_path


def test_prepare_resume_run_uses_metadata(tmp_path):
    progress_path = create_progress(tmp_path)
    base_options = PocRunOptions(output_dir=tmp_path / "out", progress_dir=tmp_path / "prog", simulate=True, verbose=False)
    record, audio_files, models, resume_options = prepare_resume_run(progress_path, base_options=base_options)
    assert record.model == "kotoba"
    assert audio_files[0].name == "sample.wav"
    assert models == ["kotoba"]
    assert resume_options.language == "ja"
    assert resume_options.chunk_size == 1024
    assert resume_options.simulate is False
    assert resume_options.timestamp == "20251120T173552"
    assert resume_options.resume_source == progress_path


def test_prepare_resume_run_rejects_completed(tmp_path):
    progress_path = create_progress(tmp_path, status="completed")
    base_options = PocRunOptions()
    with pytest.raises(ResumeCompletedError):
        prepare_resume_run(progress_path, base_options=base_options)

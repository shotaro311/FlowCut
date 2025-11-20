from pathlib import Path
import sys

test_dir = Path(__file__).resolve().parent
project_root = test_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.utils.progress import (  # noqa: E402
    create_progress_record,
    mark_block_completed,
    mark_run_status,
    save_progress,
    load_progress,
)


def test_create_and_complete_blocks(tmp_path):
    record = create_progress_record(
        run_id="run-1",
        audio_file="samples/test.wav",
        model="kotoba",
        total_blocks=2,
    )
    assert record.total_blocks == 2
    mark_block_completed(record, 1)
    assert record.completed_blocks == 1
    mark_run_status(record, "running")
    save_path = tmp_path / "progress.json"
    save_progress(record, save_path)
    loaded = load_progress(save_path)
    assert loaded.completed_blocks == 1
    mark_block_completed(loaded, 2)
    mark_run_status(loaded, "completed")
    assert loaded.status == "completed"

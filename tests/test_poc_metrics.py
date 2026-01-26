from pathlib import Path
import json
import sys

test_dir = Path(__file__).resolve().parent
project_root = test_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from notebooks.poc_metrics import summarize_blocks, summarize_file  # noqa: E402


def test_summarize_blocks_detects_violations(tmp_path):
    blocks = [
        {
            "text": "あ" * 1300,
            "duration": 40,
            "sentences": [{"text": "あ", "overlap": False}, {"text": "い", "overlap": True}],
        },
        {
            "text": "短いブロック",
            "duration": 10,
            "sentences": [{"text": "短いブロック", "overlap": False}],
        },
    ]
    stats = summarize_blocks(blocks, char_limit=1200, duration_limit=30)
    assert stats["num_blocks"] == 2
    assert stats["char_violation_blocks"] == 1
    assert stats["duration_violation_blocks"] == 1
    assert stats["overlap_sentence_ratio"] == 0.333


def test_summarize_file_reads_metadata(tmp_path):
    payload = {
        "text": "テスト",
        "words": [{"word": "テスト"}],
        "blocks": [{"text": "テスト", "duration": 0, "sentences": []}],
        "metadata": {"model": "whisper-local", "audio_file": "samples/test.wav"},
    }
    path = tmp_path / "sample.json"
    path.write_text(json.dumps(payload, ensure_ascii=False))
    record = summarize_file(path, char_limit=1200, duration_limit=30)
    assert record["model"] == "whisper-local"
    assert record["num_blocks"] == 1

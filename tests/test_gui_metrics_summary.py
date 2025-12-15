import json
from pathlib import Path

from src.gui.controller import GuiController


def test_collect_metrics_summary_falls_back_to_prompt_plus_completion(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    metrics_dir = Path("logs/metrics")
    metrics_dir.mkdir(parents=True, exist_ok=True)

    audio_path = Path("audio.m4a")
    timestamp = "20251215T145827"
    run_id = f"{audio_path.stem}_mlx_{timestamp}"
    metrics_path = metrics_dir / f"{audio_path.name}_20251215_{run_id}_metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "llm_tokens": {
                    "pass1": {
                        "prompt_tokens": 10,
                        "completion_tokens": 5,
                        "total_tokens": 0,  # total_tokens が 0 のケース
                    },
                    "pass2": {
                        "prompt_tokens": 2,
                        "completion_tokens": 3,
                        # total_tokens が無いケース
                    },
                },
                "run_total_cost_usd": 0.123,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    controller = GuiController()
    summary = controller._collect_metrics_summary(  # noqa: SLF001 - テストのため private を直接検証
        audio_path=audio_path,
        model_slugs=["mlx"],
        timestamp=timestamp,
        total_elapsed_sec=10.0,
    )
    assert summary is not None
    assert summary["metrics_files_found"] == 1
    assert summary["total_tokens"] == 20
    assert summary["total_cost_usd"] == 0.123


def test_collect_metrics_summary_cost_fallback_sums_per_pass_costs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    metrics_dir = Path("logs/metrics")
    metrics_dir.mkdir(parents=True, exist_ok=True)

    audio_path = Path("audio.m4a")
    timestamp = "20251215T145827"
    run_id = f"{audio_path.stem}_mlx_{timestamp}"
    metrics_path = metrics_dir / f"{audio_path.name}_20251215_{run_id}_metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "llm_tokens": {
                    "pass1": {"prompt_tokens": 1, "completion_tokens": 1, "cost_total_usd": 0.01},
                    "pass2": {"prompt_tokens": 1, "completion_tokens": 1, "cost_total_usd": 0.02},
                },
                "run_total_cost_usd": 0.0,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    controller = GuiController()
    summary = controller._collect_metrics_summary(  # noqa: SLF001
        audio_path=audio_path,
        model_slugs=["mlx"],
        timestamp=timestamp,
        total_elapsed_sec=10.0,
    )
    assert summary is not None
    assert summary["metrics_files_found"] == 1
    assert summary["total_cost_usd"] == 0.03


def test_collect_metrics_summary_reports_missing_metrics_files(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    Path("logs/metrics").mkdir(parents=True, exist_ok=True)

    controller = GuiController()
    summary = controller._collect_metrics_summary(  # noqa: SLF001
        audio_path=Path("audio.m4a"),
        model_slugs=["mlx"],
        timestamp="20251215T145827",
        total_elapsed_sec=1.0,
    )
    assert summary is not None
    assert summary["metrics_files_found"] == 0


def test_collect_metrics_summary_reads_from_custom_metrics_root(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    metrics_dir = Path("output/logs/metrics")
    metrics_dir.mkdir(parents=True, exist_ok=True)

    audio_path = Path("audio.m4a")
    timestamp = "20251215T145827"
    run_id = f"{audio_path.stem}_mlx_{timestamp}"
    metrics_path = metrics_dir / f"{audio_path.name}_20251215_{run_id}_metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "llm_tokens": {
                    "pass1": {
                        "prompt_tokens": 1,
                        "completion_tokens": 2,
                    },
                },
                "run_total_cost_usd": 0.0,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    controller = GuiController()
    summary = controller._collect_metrics_summary(  # noqa: SLF001
        audio_path=audio_path,
        model_slugs=["mlx"],
        timestamp=timestamp,
        total_elapsed_sec=10.0,
        metrics_root=metrics_dir,
    )
    assert summary is not None
    assert summary["metrics_files_found"] == 1
    assert summary["total_tokens"] == 3

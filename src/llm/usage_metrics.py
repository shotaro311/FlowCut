"""LLM使用量とパス別メトリクスを集約するユーティリティ."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping
import json
import logging
import threading
from datetime import datetime

from src.llm.pricing import estimate_cost

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class UsageTotals:
    provider: str | None = None
    model: str | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    duration_sec: float = 0.0  # そのパスの合計実行時間（秒）


_LOCK = threading.Lock()
# run_id -> pass_label -> UsageTotals
_USAGE_BY_RUN: Dict[str, Dict[str, UsageTotals]] = {}


def _to_int(value) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def record_usage_from_request(
    request_metadata: Mapping[str, object] | None,
    *,
    provider: str,
    model: str,
    prompt_tokens,
    completion_tokens,
    total_tokens,
) -> None:
    """プロバイダー側から呼び出され、トークン使用量をパス別に集約する."""
    if not request_metadata:
        return
    run_id = request_metadata.get("run_id")
    pass_label = (request_metadata.get("pass_label") or "unknown") if run_id else None
    if not run_id or not isinstance(run_id, str):
        return

    pt = _to_int(prompt_tokens)
    ct = _to_int(completion_tokens)
    tt = _to_int(total_tokens)

    with _LOCK:
        per_run = _USAGE_BY_RUN.setdefault(run_id, {})
        totals = per_run.setdefault(pass_label, UsageTotals())
        totals.provider = provider
        totals.model = model
        totals.prompt_tokens += pt
        totals.completion_tokens += ct
        totals.total_tokens += tt


def record_pass_time(run_id: str | None, pass_label: str, duration_sec: float) -> None:
    """TwoPassFormatter から呼び出され、パスごとの実行時間を累積する."""
    if not run_id:
        return
    if duration_sec < 0:
        return
    with _LOCK:
        per_run = _USAGE_BY_RUN.setdefault(run_id, {})
        totals = per_run.setdefault(pass_label, UsageTotals())
        totals.duration_sec += float(duration_sec)


def consume_usage_for_run(run_id: str) -> Dict[str, UsageTotals]:
    """指定 run_id の使用量情報を取り出し、内部バッファから削除する."""
    with _LOCK:
        per_run = _USAGE_BY_RUN.pop(run_id, None)
    return dict(per_run or {})


def _format_seconds(secs: float) -> str:
    """人が読みやすい `Xm Y.YYs` 形式に変換する."""
    try:
        value = float(secs)
    except (TypeError, ValueError):
        return "0.00s"
    if value < 0:
        value = 0.0
    minutes = int(value // 60)
    seconds = value - minutes * 60
    if minutes > 0:
        return f"{minutes}m {seconds:.2f}s"
    return f"{seconds:.2f}s"


def write_run_metrics_file(
    *,
    run_id: str,
    source_name: str | None,
    runner_slug: str,
    timestamp: str,
    stage_timings_sec: Dict[str, float],
    total_elapsed_sec: float,
    usage_by_pass: Dict[str, UsageTotals],
    output_dir: Path | None = None,
    audio_duration_sec: float | None = None,
) -> Path:
    """1つの run（音声×モデル）についてメトリクスJSONを出力する."""
    metrics_root = output_dir or Path("logs/metrics")
    metrics_root.mkdir(parents=True, exist_ok=True)

    base_name = (source_name or run_id).replace("/", "_")
    date_str = datetime.utcnow().strftime("%Y%m%d")
    metrics_path = metrics_root / f"{base_name}_{date_str}_{run_id}_metrics.json"

    data = {
        "run_id": run_id,
        "audio_file": source_name,
        "runner": runner_slug,
        "requested_at": timestamp,
        "total_elapsed_time": _format_seconds(total_elapsed_sec),
        "audio_duration_time": _format_seconds(audio_duration_sec if audio_duration_sec is not None else 0.0),
        "stage_timings_time": {
            name: _format_seconds(value) for name, value in stage_timings_sec.items()
        },
        "llm_tokens": {},
    }

    # パス別の費用を計算しながら、合計コストも集計する
    run_total_cost = 0.0
    for label, totals in sorted(usage_by_pass.items()):
        cost = estimate_cost(
            provider=totals.provider,
            model=totals.model,
            prompt_tokens=totals.prompt_tokens,
            completion_tokens=totals.completion_tokens,
        )
        entry = {
            "provider": totals.provider,
            "model": totals.model,
            "prompt_tokens": totals.prompt_tokens,
            "completion_tokens": totals.completion_tokens,
            "total_tokens": totals.total_tokens,
            "duration_time": _format_seconds(totals.duration_sec),
        }
        if cost is not None:
            entry.update(
                {
                    "cost_input_usd": cost["input_cost_usd"],
                    "cost_output_usd": cost["output_cost_usd"],
                    "cost_total_usd": cost["total_cost_usd"],
                }
            )
            run_total_cost += cost["total_cost_usd"]
        data["llm_tokens"][label] = entry

    data["run_total_cost_usd"] = run_total_cost

    metrics_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("metrics saved: %s", metrics_path)
    return metrics_path


__all__ = [
    "UsageTotals",
    "record_usage_from_request",
    "record_pass_time",
    "consume_usage_for_run",
    "write_run_metrics_file",
]

"""Controller layer for running the transcription pipeline from the GUI."""
from __future__ import annotations

import json
import threading
from datetime import datetime
import time
from pathlib import Path
from typing import Callable, Iterable, List, Sequence, Dict, Any
from dataclasses import dataclass

from src.config.settings import get_settings
from src.gui.config import get_config
from src.llm.profiles import get_profile
from src.pipeline import PocRunOptions, ensure_audio_files, execute_poc_run, resolve_models

UiCallback = Callable[[], None]


@dataclass
class WorkflowInfo:
    """ワークフローの情報を管理する。"""
    workflow_id: str
    thread: threading.Thread
    is_running: bool = True
    start_time: float = 0.0


class GuiController:
    """Run the existing pipeline in a background thread for the GUI.
    
    NOTE: MLX Whisperはスレッドセーフではないため、同時に1つのワークフローのみ実行可能。
    複数のワークフローはキュー方式で順次実行される。
    """

    def __init__(self, ui_dispatch: Callable[[Callable[[], None]], None] | None = None) -> None:
        self.ui_dispatch = ui_dispatch or (lambda f: f())
        # ワークフロー管理
        self.workflows: Dict[str, WorkflowInfo] = {}
        self._lock = threading.Lock()
        # MLX Whisperの並列実行を防ぐためのグローバルロック
        self._execution_lock = threading.Lock()

    def run_workflow(
        self,
        workflow_id: str,
        audio_path: Path,
        *,
        subtitle_dir: Path | None = None,
        llm_provider: str | None = None,
        llm_profile: str | None = None,
        workflow: str = "workflow1",
        pass1_model: str | None = None,
        pass2_model: str | None = None,
        pass3_model: str | None = None,
        pass4_model: str | None = None,
        start_delay: float = 0.0,
        keep_extracted_audio: bool = False,
        enable_pass5: bool = False,
        pass5_max_chars: int = 17,
        pass5_model: str | None = None,
        save_logs: bool = False,
        on_start: Callable[[], None] | None = None,
        on_success: Callable[[List[Path], dict | None], None] | None = None,
        on_error: Callable[[Exception], None] | None = None,
        on_finish: Callable[[], None] | None = None,
        on_progress: Callable[[str, int], None] | None = None,
    ) -> None:
        """単一ワークフローを非同期で実行する。

        コールバックはui_dispatch経由でUIスレッドにディスパッチされる。
        """
        # 既存のワークフローが実行中かチェック
        with self._lock:
            if workflow_id in self.workflows and self.workflows[workflow_id].is_running:
                self._notify(on_error, RuntimeError(f"スロット {workflow_id} は既に実行中です"))
                return

        def worker() -> None:
            self._notify(on_start)
            t_start = time.perf_counter()
            
            # MLX Whisperの並列実行を防ぐためにロックを取得
            # 他のワークフローが実行中の場合は待機
            self._notify(on_progress, "待機中（他の処理完了待ち）", 0)
            with self._execution_lock:
                t_processing_start = time.perf_counter()
                try:
                    # プログレスコールバックをワークフローID付きでラップ
                    safe_progress_callback = None
                    if on_progress:
                        safe_progress_callback = lambda phase, progress: self._notify(on_progress, phase, progress)
                    
                    options = self._build_options(
                        subtitle_dir=subtitle_dir,
                        llm_provider=llm_provider,
                        llm_profile=llm_profile,
                        workflow=workflow,
                        pass1_model=pass1_model,
                        pass2_model=pass2_model,
                        pass3_model=pass3_model,
                        pass4_model=pass4_model,
                        start_delay=start_delay,
                        keep_extracted_audio=keep_extracted_audio,
                        enable_pass5=enable_pass5,
                        pass5_max_chars=pass5_max_chars,
                        pass5_model=pass5_model,
                        save_logs=save_logs,
                        progress_callback=safe_progress_callback,
                    )
                    # タイムスタンプを固定してGUI側からも出力パスを把握できるようにする
                    options.timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")

                    audio_files = ensure_audio_files([audio_path])
                    model_slugs = resolve_models(None)
                    result_paths = execute_poc_run(audio_files, model_slugs, options)
                    t_end = time.perf_counter()
                    total_elapsed_sec = max(0.0, t_end - t_start)
                    wait_elapsed_sec = max(0.0, t_processing_start - t_start)
                    processing_elapsed_sec = max(0.0, t_end - t_processing_start)

                    # メトリクス集計
                    metrics = self._collect_metrics_summary(
                        audio_path=audio_path,
                        model_slugs=model_slugs,
                        timestamp=options.timestamp,
                        total_elapsed_sec=total_elapsed_sec,
                        metrics_root=self._resolve_metrics_root(options, audio_path),
                    )
                    if metrics is None:
                        metrics = {}
                    metrics["wait_elapsed_sec"] = wait_elapsed_sec
                    metrics["processing_elapsed_sec"] = processing_elapsed_sec
                    metrics["pass5_enabled"] = bool(options.enable_pass5)
                    metrics["pass5_max_chars"] = int(options.pass5_max_chars)
                    metrics["pass5_model"] = options.pass5_model

                    self._notify(on_success, result_paths, metrics)
                except Exception as exc:
                    self._notify(on_error, exc)
                finally:
                    # ワークフロー情報をクリーンアップ
                    with self._lock:
                        if workflow_id in self.workflows:
                            self.workflows[workflow_id].is_running = False
                    self._notify(on_finish)

        # ワークフロー情報を登録してスレッドを開始
        thread = threading.Thread(target=worker, daemon=True)
        with self._lock:
            self.workflows[workflow_id] = WorkflowInfo(
                workflow_id=workflow_id,
                thread=thread,
                is_running=True,
                start_time=time.perf_counter()
            )
        thread.start()

    def _build_options(
        self,
        *,
        subtitle_dir: Path | None = None,
        llm_provider: str | None = None,
        llm_profile: str | None = None,
        workflow: str = "workflow1",
        pass1_model: str | None = None,
        pass2_model: str | None = None,
        pass3_model: str | None = None,
        pass4_model: str | None = None,
        start_delay: float = 0.0,
        keep_extracted_audio: bool = False,
        enable_pass5: bool = False,
        pass5_max_chars: int = 17,
        pass5_model: str | None = None,
        save_logs: bool = False,
        progress_callback: Callable[[str, int], None] | None = None,
    ) -> PocRunOptions:
        """GUIから渡された設定と環境値を組み合わせてPocRunOptionsを構築する。

        優先順位イメージ:
        1. GUIの詳細設定（pass1〜4モデル）
        2. プロファイル定義（llm_profile）
        3. 環境変数ベースのデフォルト設定（settings.llm）
        """
        settings = get_settings()

        # ベースは環境変数
        provider = llm_provider or settings.llm.default_provider
        p1 = pass1_model
        p2 = pass2_model
        p3 = pass3_model
        p4 = pass4_model

        # プロファイルが指定されていればそこから補完
        if llm_profile:
            profile = get_profile(llm_profile)
            if profile is not None:
                # プロバイダー未指定ならプロファイル側を採用
                if not llm_provider and profile.provider:
                    provider = profile.provider
                p1 = p1 or profile.pass1_model
                p2 = p2 or profile.pass2_model
                p3 = p3 or profile.pass3_model
                p4 = p4 or profile.pass4_model

        # それでも未設定の箇所は環境設定から埋める
        p1 = p1 or settings.llm.pass1_model
        p2 = p2 or settings.llm.pass2_model
        p3 = p3 or settings.llm.pass3_model
        p4 = p4 or settings.llm.pass4_model

        glossary_terms = get_config().get_glossary_terms()

        return PocRunOptions(
            output_dir=Path("temp/poc_samples"),
            progress_dir=Path("temp/progress"),
            subtitle_dir=subtitle_dir or Path("output"),
            simulate=False,
            llm_provider=provider,
            llm_profile=llm_profile,
            workflow=workflow,
            llm_pass1_model=p1,
            llm_pass2_model=p2,
            llm_pass3_model=p3,
            llm_pass4_model=p4,
            llm_timeout=settings.llm.request_timeout,
            glossary_terms=glossary_terms,
            start_delay=float(start_delay),
            keep_extracted_audio=bool(keep_extracted_audio),
            enable_pass5=bool(enable_pass5),
            pass5_max_chars=int(pass5_max_chars),
            pass5_model=pass5_model,
            progress_callback=progress_callback,
            save_logs=save_logs,
        )

    def _resolve_metrics_root(self, options: PocRunOptions, audio_path: Path) -> Path:
        """メトリクス（logs/metrics）の探索先を決定する。"""
        if options.save_logs:
            base_dir = options.subtitle_dir / f"{audio_path.stem}_{options.timestamp}"
            candidate = base_dir / "logs" / "metrics"
            if candidate.exists():
                return candidate

            # まれにディレクトリ名が連番になるケースに備えてフォールバック
            pattern = f"{audio_path.stem}_{options.timestamp}*"
            for path in sorted(options.subtitle_dir.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True):
                metrics_root = path / "logs" / "metrics"
                if metrics_root.exists():
                    return metrics_root
            return candidate
        return Path("logs/metrics")

    def _collect_output_paths(
        self,
        *,
        audio_path: Path,
        model_slugs: Sequence[str],
        timestamp: str | None,
        options: PocRunOptions,
        base_paths: Iterable[Path],
    ) -> List[Path]:
        collected: List[Path] = list(base_paths)
        if not timestamp:
            return collected
        for slug in model_slugs:
            srt_path = options.subtitle_dir / f"{audio_path.stem}_{slug}_{timestamp}.srt"
            if srt_path.exists():
                collected.append(srt_path)
        return collected

    def _collect_metrics_summary(
        self,
        audio_path: Path,
        model_slugs: Sequence[str],
        timestamp: str | None,
        total_elapsed_sec: float,
        *,
        metrics_root: Path | None = None,
    ) -> dict | None:
        """logs/metrics 配下のメトリクスを集計し、GUI向けの簡易サマリを返す。

        - 総トークン数（Pass1〜4・全モデルの合計）
        - 概算APIコスト（USD, run_total_cost_usd の合計）
        - 総処理時間（秒）
        """
        if not timestamp:
            return {
                "total_prompt_tokens": 0,
                "total_completion_tokens": 0,
                "total_tokens": 0,
                "total_cost_usd": 0.0,
                "total_elapsed_sec": total_elapsed_sec,
                "metrics_files_found": 0,
                "per_runner": {},
            }

        root = metrics_root or Path("logs/metrics")
        if not root.exists():
            return {
                "total_prompt_tokens": 0,
                "total_completion_tokens": 0,
                "total_tokens": 0,
                "total_cost_usd": 0.0,
                "total_elapsed_sec": total_elapsed_sec,
                "metrics_files_found": 0,
                "per_runner": {},
            }

        total_prompt_tokens = 0
        total_completion_tokens = 0
        total_tokens = 0
        total_cost = 0.0
        metrics_files_found = 0
        per_runner: dict = {}

        # ファイル名フォーマットは usage_metrics.write_run_metrics_file を参照
        # {audio_file.name}_{date_str}_{run_id}_metrics.json
        for slug in model_slugs:
            run_id = f"{audio_path.stem}_{slug}_{timestamp}"
            pattern = f"*_{run_id}_metrics.json"
            matched = sorted(root.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
            for path in matched[:1]:
                try:
                    data = json.loads(path.read_text(encoding="utf-8"))
                except Exception:
                    continue
                metrics_files_found += 1
                stage_timings_time = data.get("stage_timings_time") or {}
                llm_tokens = data.get("llm_tokens") or {}
                pass_durations: dict = {}
                pass_metrics: dict = {}
                if isinstance(llm_tokens, dict):
                    for label, entry in llm_tokens.items():
                        if not isinstance(entry, dict):
                            continue
                        prompt_tokens = int(entry.get("prompt_tokens") or 0)
                        completion_tokens = int(entry.get("completion_tokens") or 0)
                        raw_total = entry.get("total_tokens")
                        if isinstance(raw_total, (int, float)) and int(raw_total) > 0:
                            entry_total_tokens = int(raw_total)
                        else:
                            entry_total_tokens = prompt_tokens + completion_tokens

                        total_prompt_tokens += prompt_tokens
                        total_completion_tokens += completion_tokens
                        total_tokens += entry_total_tokens

                        duration = entry.get("duration_time")
                        if isinstance(duration, str) and duration.strip():
                            pass_durations[str(label)] = duration.strip()

                        pass_metrics[str(label)] = {
                            "provider": entry.get("provider"),
                            "model": entry.get("model"),
                            "prompt_tokens": prompt_tokens,
                            "completion_tokens": completion_tokens,
                            "total_tokens": entry_total_tokens,
                            "cost_total_usd": entry.get("cost_total_usd"),
                            "cost_input_usd": entry.get("cost_input_usd"),
                            "cost_output_usd": entry.get("cost_output_usd"),
                        }
                cost = data.get("run_total_cost_usd")
                if isinstance(cost, (int, float)) and float(cost) > 0:
                    total_cost += float(cost)
                else:
                    # run_total_cost_usd が無い/0 の場合は、パス別コストの合計で代替する
                    if isinstance(llm_tokens, dict):
                        for entry in llm_tokens.values():
                            if not isinstance(entry, dict):
                                continue
                            entry_cost = entry.get("cost_total_usd")
                            if isinstance(entry_cost, (int, float)):
                                total_cost += float(entry_cost)

                # GUI表示用の内訳（runnerごと）
                transcribe_time = stage_timings_time.get("transcribe_sec") if isinstance(stage_timings_time, dict) else None
                llm_two_pass_time = stage_timings_time.get("llm_two_pass_sec") if isinstance(stage_timings_time, dict) else None
                per_runner[slug] = {
                    "metrics_file": str(path),
                    "transcribe_time": transcribe_time,
                    "llm_two_pass_time": llm_two_pass_time,
                    "pass_durations": pass_durations,
                    "pass_metrics": pass_metrics,
                }

        return {
            "total_prompt_tokens": total_prompt_tokens,
            "total_completion_tokens": total_completion_tokens,
            "total_tokens": total_tokens,
            "total_cost_usd": total_cost,
            "total_elapsed_sec": total_elapsed_sec,
            "metrics_files_found": metrics_files_found,
            "per_runner": per_runner,
        }

    def _notify(self, callback: Callable[..., None] | None, *args, **kwargs) -> None:
        if callback is None:
            return
        if self.ui_dispatch:
            self.ui_dispatch(lambda: callback(*args, **kwargs))
        else:
            callback(*args, **kwargs)

    def is_workflow_running(self, workflow_id: str) -> bool:
        """指定されたワークフローが実行中かどうかを返す。"""
        with self._lock:
            return workflow_id in self.workflows and self.workflows[workflow_id].is_running
    
    def get_running_workflows(self) -> List[str]:
        """実行中のワークフローIDリストを返す。"""
        with self._lock:
            return [wf_id for wf_id, info in self.workflows.items() if info.is_running]
    
    def stop_workflow(self, workflow_id: str) -> bool:
        """指定されたワークフローを停止する（実装は将来対応）。"""
        # TODO: スレッドの停止を実装する必要がある
        return False


__all__ = ["GuiController"]

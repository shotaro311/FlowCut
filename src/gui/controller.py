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
        pass1_model: str | None = None,
        pass2_model: str | None = None,
        pass3_model: str | None = None,
        pass4_model: str | None = None,
        start_delay: float = 0.0,
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
                self._notify(on_error, RuntimeError(f"ワークフロー {workflow_id} は既に実行中です"))
                return

        def worker() -> None:
            self._notify(on_start)
            
            # MLX Whisperの並列実行を防ぐためにロックを取得
            # 他のワークフローが実行中の場合は待機
            self._notify(on_progress, "待機中（他の処理完了待ち）", 0)
            with self._execution_lock:
                try:
                    # プログレスコールバックをワークフローID付きでラップ
                    safe_progress_callback = None
                    if on_progress:
                        safe_progress_callback = lambda phase, progress: self._notify(on_progress, phase, progress)
                    
                    options = self._build_options(
                        subtitle_dir=subtitle_dir,
                        llm_provider=llm_provider,
                        llm_profile=llm_profile,
                        pass1_model=pass1_model,
                        pass2_model=pass2_model,
                        pass3_model=pass3_model,
                        pass4_model=pass4_model,
                        start_delay=start_delay,
                        progress_callback=safe_progress_callback,
                    )
                    # タイムスタンプを固定してGUI側からも出力パスを把握できるようにする
                    options.timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")

                    t_start = time.perf_counter()
                    audio_files = ensure_audio_files([audio_path])
                    model_slugs = resolve_models(None)
                    result_paths = execute_poc_run(audio_files, model_slugs, options)
                    t_end = time.perf_counter()
                    elapsed_sec = max(0.0, t_end - t_start)

                    # メトリクス集計
                    metrics = self._collect_metrics_summary(
                        audio_path=audio_path,
                        model_slugs=model_slugs,
                        timestamp=options.timestamp,
                        total_elapsed_sec=elapsed_sec,
                    )

                    # ログを収集して保存
                    log_dir = self._collect_and_save_logs(
                        audio_path=audio_path,
                        model_slugs=model_slugs,
                        timestamp=options.timestamp,
                        subtitle_dir=subtitle_dir,
                    )
                    if log_dir and metrics:
                        metrics["log_dir"] = str(log_dir)

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
        pass1_model: str | None = None,
        pass2_model: str | None = None,
        pass3_model: str | None = None,
        pass4_model: str | None = None,
        start_delay: float = 0.0,
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

        return PocRunOptions(
            output_dir=Path("temp/poc_samples"),
            progress_dir=Path("temp/progress"),
            subtitle_dir=subtitle_dir or Path("output"),
            simulate=False,
            llm_provider=provider,
            llm_profile=llm_profile,
            llm_pass1_model=p1,
            llm_pass2_model=p2,
            llm_pass3_model=p3,
            llm_pass4_model=p4,
            llm_timeout=settings.llm.request_timeout,
            start_delay=start_delay,
            progress_callback=progress_callback,
        )

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
    ) -> dict | None:
        """logs/metrics 配下のメトリクスを集計し、GUI向けの簡易サマリを返す。

        - 総トークン数（Pass1〜4・全モデルの合計）
        - 概算APIコスト（USD, run_total_cost_usd の合計）
        - 総処理時間（秒）
        """
        if not timestamp:
            return {
                "total_tokens": 0,
                "total_cost_usd": 0.0,
                "total_elapsed_sec": total_elapsed_sec,
            }

        metrics_root = Path("logs/metrics")
        if not metrics_root.exists():
            return {
                "total_tokens": 0,
                "total_cost_usd": 0.0,
                "total_elapsed_sec": total_elapsed_sec,
            }

        total_tokens = 0
        total_cost = 0.0

        # ファイル名フォーマットは usage_metrics.write_run_metrics_file を参照
        # {audio_file.name}_{date_str}_{run_id}_metrics.json
        for slug in model_slugs:
            run_id = f"{audio_path.stem}_{slug}_{timestamp}"
            pattern = f"{audio_path.name}_*_{run_id}_metrics.json"
            for path in metrics_root.glob(pattern):
                try:
                    data = json.loads(path.read_text(encoding="utf-8"))
                except Exception:
                    continue
                llm_tokens = data.get("llm_tokens") or {}
                if isinstance(llm_tokens, dict):
                    for entry in llm_tokens.values():
                        if not isinstance(entry, dict):
                            continue
                        total_tokens += int(entry.get("total_tokens") or 0)
                cost = data.get("run_total_cost_usd")
                if isinstance(cost, (int, float)):
                    total_cost += float(cost)

        return {
            "total_tokens": total_tokens,
            "total_cost_usd": total_cost,
            "total_elapsed_sec": total_elapsed_sec,
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
    
    def stop_workflow(self, workflow_id: str) -> None:
        """指定されたワークフローを停止する（実装は将来対応）。"""
        # TODO: 実装は難しいので一旦スキップ
        pass

    def _collect_and_save_logs(
        self,
        *,
        audio_path: Path,
        model_slugs: Sequence[str],
        timestamp: str,
        subtitle_dir: Path,
    ) -> Path | None:
        """ログファイルを収集し、出力ディレクトリ内のログフォルダに保存する。

        Args:
            audio_path: 音声ファイルパス
            model_slugs: モデルスラッグのリスト
            timestamp: タイムスタンプ
            subtitle_dir: SRT出力先ディレクトリ

        Returns:
            ログフォルダのパス（作成された場合）、失敗時は None
        """
        import shutil

        # ログソースディレクトリ
        project_root = Path.cwd()
        llm_raw_dir = project_root / "logs" / "llm_raw"
        poc_samples_dir = project_root / "temp" / "poc_samples"
        progress_dir = project_root / "temp" / "progress"

        # 出力先ログフォルダ名を構築
        audio_basename = audio_path.stem
        model_slug = model_slugs[0] if model_slugs else "mlx"
        log_folder_name = f"{audio_basename}_{model_slug}_{timestamp}_logs"
        log_output_dir = subtitle_dir / log_folder_name

        try:
            log_output_dir.mkdir(parents=True, exist_ok=True)

            # 各ログディレクトリをコピー
            if llm_raw_dir.exists():
                dest_llm_raw = log_output_dir / "llm_raw"
                if dest_llm_raw.exists():
                    shutil.rmtree(dest_llm_raw)
                shutil.copytree(llm_raw_dir, dest_llm_raw)

            if poc_samples_dir.exists():
                dest_poc_samples = log_output_dir / "poc_samples"
                if dest_poc_samples.exists():
                    shutil.rmtree(dest_poc_samples)
                shutil.copytree(poc_samples_dir, dest_poc_samples)

            if progress_dir.exists():
                dest_progress = log_output_dir / "progress"
                if dest_progress.exists():
                    shutil.rmtree(dest_progress)
                shutil.copytree(progress_dir, dest_progress)

            return log_output_dir
        except Exception as e:
            # ログコピー失敗は致命的エラーではないのでログを出して継続
            print(f"Warning: Failed to copy logs: {e}")
            return None


__all__ = ["GuiController"]

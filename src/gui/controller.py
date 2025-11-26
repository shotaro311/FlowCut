"""Controller layer for running the transcription pipeline from the GUI."""
from __future__ import annotations

import threading
from datetime import datetime
import time
from pathlib import Path
from typing import Callable, Iterable, List, Sequence

from src.config.settings import get_settings
from src.llm.profiles import get_profile
from src.pipeline import PocRunOptions, ensure_audio_files, execute_poc_run, resolve_models

UiCallback = Callable[[], None]


class GuiController:
    """Run the existing pipeline in a background thread for the GUI."""

    def __init__(self, ui_dispatch: Callable[[UiCallback], None] | None = None) -> None:
        self._ui_dispatch = ui_dispatch

    def run_pipeline(
        self,
        audio_path: Path,
        *,
        subtitle_dir: Path | None = None,
        llm_provider: str | None = None,
        llm_profile: str | None = None,
        pass1_model: str | None = None,
        pass2_model: str | None = None,
        pass3_model: str | None = None,
        pass4_model: str | None = None,
        on_start: Callable[[], None] | None = None,
        on_success: Callable[[List[Path], dict | None], None] | None = None,
        on_error: Callable[[Exception], None] | None = None,
        on_finish: Callable[[], None] | None = None,
    ) -> None:
        """Execute the pipeline asynchronously.

        The callbacks are dispatched on the UI thread via ``ui_dispatch`` if provided.
        """

        def worker() -> None:
            self._notify(on_start)
            try:
                options = self._build_options(
                    subtitle_dir=subtitle_dir,
                    llm_provider=llm_provider,
                    llm_profile=llm_profile,
                    pass1_model=pass1_model,
                    pass2_model=pass2_model,
                    pass3_model=pass3_model,
                    pass4_model=pass4_model,
                )
                # タイムスタンプを固定してGUI側からも出力パスを把握できるようにする
                options.timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")

                t_start = time.perf_counter()
                audio_files = ensure_audio_files([audio_path])
                model_slugs = resolve_models(None)
                result_paths = execute_poc_run(audio_files, model_slugs, options)
                t_end = time.perf_counter()
                elapsed_sec = max(0.0, t_end - t_start)

                output_paths = self._collect_output_paths(
                    audio_path=audio_path,
                    model_slugs=model_slugs,
                    timestamp=options.timestamp,
                    options=options,
                    base_paths=result_paths,
                )
                metrics_summary = self._collect_metrics_summary(
                    audio_path=audio_path,
                    model_slugs=model_slugs,
                    timestamp=options.timestamp,
                    total_elapsed_sec=elapsed_sec,
                )
                self._notify(on_success, output_paths, metrics_summary)
            except Exception as exc:  # pragma: no cover - GUI通知のみ
                self._notify(on_error, exc)
            finally:
                self._notify(on_finish)

        thread = threading.Thread(target=worker, daemon=True)
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
        *,
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
        if self._ui_dispatch:
            self._ui_dispatch(lambda: callback(*args, **kwargs))
        else:
            callback(*args, **kwargs)


__all__ = ["GuiController"]

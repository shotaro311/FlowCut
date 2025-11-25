"""Controller layer for running the transcription pipeline from the GUI."""
from __future__ import annotations

import threading
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable, List, Sequence

from src.config.settings import get_settings
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
        on_start: Callable[[], None] | None = None,
        on_success: Callable[[List[Path]], None] | None = None,
        on_error: Callable[[Exception], None] | None = None,
        on_finish: Callable[[], None] | None = None,
    ) -> None:
        """Execute the pipeline asynchronously.

        The callbacks are dispatched on the UI thread via ``ui_dispatch`` if provided.
        """

        def worker() -> None:
            self._notify(on_start)
            try:
                options = self._build_options(subtitle_dir=subtitle_dir)
                # タイムスタンプを固定してGUI側からも出力パスを把握できるようにする
                options.timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")

                audio_files = ensure_audio_files([audio_path])
                model_slugs = resolve_models(None)
                result_paths = execute_poc_run(audio_files, model_slugs, options)
                output_paths = self._collect_output_paths(
                    audio_path=audio_path,
                    model_slugs=model_slugs,
                    timestamp=options.timestamp,
                    options=options,
                    base_paths=result_paths,
                )
                self._notify(on_success, output_paths)
            except Exception as exc:  # pragma: no cover - GUI通知のみ
                self._notify(on_error, exc)
            finally:
                self._notify(on_finish)

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()

    def _build_options(self, *, subtitle_dir: Path | None = None) -> PocRunOptions:
        settings = get_settings()
        return PocRunOptions(
            output_dir=Path("temp/poc_samples"),
            progress_dir=Path("temp/progress"),
            subtitle_dir=subtitle_dir or Path("output"),
            simulate=False,
            llm_provider=settings.llm.default_provider,
            llm_pass1_model=settings.llm.pass1_model,
            llm_pass2_model=settings.llm.pass2_model,
            llm_pass3_model=settings.llm.pass3_model,
            llm_pass4_model=settings.llm.pass4_model,
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

    def _notify(self, callback: Callable[..., None] | None, *args, **kwargs) -> None:
        if callback is None:
            return
        if self._ui_dispatch:
            self._ui_dispatch(lambda: callback(*args, **kwargs))
        else:
            callback(*args, **kwargs)


__all__ = ["GuiController"]

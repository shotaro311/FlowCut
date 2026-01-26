"""Shared helpers for the Phase 1 PoC transcription flow."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple, Callable
import sys
import time

from src.llm.two_pass import TwoPassResult
from src.llm.formatter import FormatterError
from src.transcribe import (
    TranscriptionConfig,
    TranscriptionResult,
    available_runners,
    describe_runners,
    get_runner,
)
from src.utils.progress import (
    ProgressRecord,
    create_progress_record,
    load_progress,
    mark_block_completed,
    mark_run_status,
    save_progress,
)
from src.utils.paths import generate_sequential_path
from src.utils.audio_extractor import (
    is_video_file,
    extract_audio_from_video,
    cleanup_extracted_audio,
    AudioExtractionError,
)

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class PocRunOptions:
    language: str | None = None
    chunk_size: int | None = None
    output_dir: Path = Path("temp/poc_samples")
    progress_dir: Path = Path("temp/progress")
    subtitle_dir: Path = Path("output")
    simulate: bool = True
    verbose: bool = False
    timestamp: str | None = None
    resume_source: Path | None = None
    llm_provider: str | None = None
    llm_profile: str | None = None
    workflow: str = "workflow2"
    llm_pass1_model: str | None = None
    llm_pass2_model: str | None = None
    llm_pass3_model: str | None = None
    llm_pass4_model: str | None = None
    rewrite: bool | None = None
    llm_temperature: float | None = None
    llm_timeout: float | None = None
    glossary_terms: list[str] | None = None
    start_delay: float = 0.0
    keep_extracted_audio: bool = False
    enable_pass5: bool = False
    pass5_max_chars: int = 17
    pass5_provider: str | None = None
    pass5_model: str | None = None
    progress_callback: Callable[[str, int], None] | None = None
    save_logs: bool = False

    def normalized_timestamp(self) -> str:
        # ファイル名の衝突を避けるため、秒まで含めたタイムスタンプを使用（例: 20251124T125830）
        return self.timestamp or datetime.now().strftime("%Y%m%dT%H%M%S")


def resolve_models(raw: str | None) -> List[str]:
    """ランナー一覧文字列から使用するランナー slug のリストを返す。

    - raw が None/空文字の場合は「プラットフォーム別のデフォルト」を選ぶ
      - macOS (darwin): MLX ランナー（'mlx'）
      - それ以外（Windows / Linux 等）: Faster-Whisper ランナー（'faster'）
    """
    if not raw:
        runners = available_runners()
        default_slug = "mlx" if sys.platform == "darwin" else "faster"
        return [slug for slug in runners if slug == default_slug] or runners
    requested = [token.strip() for token in raw.split(",") if token.strip()]
    unknown = [slug for slug in requested if slug not in available_runners()]
    if unknown:
        raise ValueError(
            f"未登録のランナーが指定されました: {unknown}. 候補: {available_runners()}"
        )
    return requested


def list_models() -> List[dict]:
    return describe_runners()


def ensure_audio_files(paths: Iterable[Path]) -> List[Path]:
    resolved: List[Path] = []
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"音声ファイルが見つかりません: {path}")
        resolved.append(path)
    return resolved


def execute_poc_run(
    audio_files: Sequence[Path],
    models: Sequence[str],
    options: PocRunOptions,
) -> List[Path]:
    """Execute transcription for the provided audio files and models."""
    if not audio_files:
        raise ValueError("audio_files は1件以上指定してください")
    if not models:
        raise ValueError("models は1件以上指定してください")

    timestamp = options.normalized_timestamp()
    saved_paths: List[Path] = []
    json_output_dir = options.output_dir
    progress_dir = options.progress_dir
    metrics_output_dir: Path | None = None
    raw_llm_log_dir: Path | None = None
    enforce_file_limits = True
    run_output_dir: Path | None = None

    if options.save_logs:
        # 1回の実行ごとにフォルダを作り、SRTと logs/ をまとめて保存する
        run_base_dir = options.subtitle_dir / f"{audio_files[0].stem}_{timestamp}"
        run_output_dir = generate_sequential_path(run_base_dir)
        logs_root = run_output_dir / "logs"
        json_output_dir = logs_root / "poc_samples"
        progress_dir = logs_root / "progress"
        metrics_output_dir = logs_root / "metrics"
        raw_llm_log_dir = logs_root / "llm_raw"
        enforce_file_limits = False

    for slug in models:
        runner = get_runner(slug)
        config = TranscriptionConfig(
            language=options.language,
            chunk_size=options.chunk_size,
            simulate=options.simulate,
            extra={"requested_at": timestamp},
        )
        logger.info("=== %s (%s) ===", slug, runner.display_name)
        runner.prepare(config)
        for input_path in audio_files:
            # 動画ファイルの場合は音声を抽出
            extracted_audio_path: Path | None = None
            if is_video_file(input_path):
                if options.progress_callback:
                    options.progress_callback("動画から音声抽出中", 5)
                logger.info("動画ファイルを検出: %s", input_path.name)
                try:
                    extracted_audio_path = extract_audio_from_video(input_path)
                    audio_path = extracted_audio_path
                    # 元の動画ファイル名を保持（SRTファイル名に使用）
                    original_stem = input_path.stem
                except AudioExtractionError as exc:
                    logger.error("音声抽出に失敗しました: %s", exc)
                    raise
            else:
                audio_path = input_path
                original_stem = input_path.stem
            
            # Phase 1: Whisper transcription
            if options.progress_callback:
                options.progress_callback("Whisper文字起こし", 20)
            t_run_start = time.perf_counter()
            t_transcribe_start = time.perf_counter()
            result = runner.transcribe(audio_path, config)
            t_transcribe_end = time.perf_counter()
            logger.info(
                "transcription_done runner=%s audio=%s duration_sec=%.3f",
                slug,
                audio_path.name,
                t_transcribe_end - t_transcribe_start,
            )
            blocks = build_single_block(result)
            # run_idは元のファイル名（動画の場合は動画名）をベースにする
            run_id = f"{original_stem}_{slug}_{timestamp}"
            output_path = json_output_dir / f"{run_id}.json"
            save_result(
                result,
                output_path,
                blocks=blocks,
                enforce_file_limit=enforce_file_limits,
            )
            saved_paths.append(output_path)

            subtitle_path: Path | None = None
            if options.llm_provider:
                # 希望ファイル名 {run_id}.srt をベースに、
                # 既存ファイルがある場合は audio (1).srt 形式の
                # 連番サフィックス付きファイル名を採用する
                subtitle_base_dir = run_output_dir or options.subtitle_dir
                desired_subtitle_path = subtitle_base_dir / f"{run_id}.srt"
                subtitle_path = generate_sequential_path(desired_subtitle_path)
                subtitle_text: str | None = None
                if not result.words:
                    logger.warning("wordタイムスタンプが空のためSRT生成をスキップします: %s", run_id)
                else:

                    def _build_formatter(workflow: str):
                        from src.llm.workflows.registry import get_workflow

                        wf = get_workflow(workflow)
                        if wf.optimized_pass4:
                            try:
                                from src.llm.two_pass_optimized import TwoPassFormatter as Formatter
                                logger.info("Using Optimized TwoPassFormatter (%s)", wf.slug)
                            except ImportError:
                                logger.warning(
                                    "src.llm.two_pass_optimized not found; falling back to standard TwoPassFormatter"
                                )
                                from src.llm.two_pass import TwoPassFormatter as Formatter
                        else:
                            from src.llm.two_pass import TwoPassFormatter as Formatter

                        return Formatter(
                            llm_provider=options.llm_provider,
                            temperature=options.llm_temperature,
                            timeout=options.llm_timeout,
                            pass1_model=options.llm_pass1_model,
                            pass2_model=options.llm_pass2_model,
                            pass3_model=options.llm_pass3_model,
                            pass4_model=options.llm_pass4_model,
                            workflow=wf.slug,
                            glossary_terms=options.glossary_terms,
                            run_id=run_id,
                            source_name=input_path.name,
                            start_delay=options.start_delay,
                            raw_log_dir=raw_llm_log_dir,
                        )

                    two_pass = _build_formatter(options.workflow)
                    # Phase 2-5: LLM passes
                    t_llm_start = time.perf_counter()
                    if options.progress_callback:
                        options.progress_callback("LLM Pass 1", 40)
                    tp_result: TwoPassResult | None = two_pass.run(
                        text=result.text,
                        words=result.words or [],
                        max_chars=17.0,
                        progress_callback=options.progress_callback,
                    )
                    if tp_result:
                        subtitle_text = tp_result.srt_text
                    t_llm_end = time.perf_counter()
                    logger.info(
                        "llm_two_pass_done provider=%s audio=%s duration_sec=%.3f",
                        options.llm_provider,
                        audio_path.name,
                        t_llm_end - t_llm_start,
                    )
                    if options.enable_pass5 and subtitle_text:
                        try:
                            from src.llm.pass5_processor import Pass5Processor

                            if options.progress_callback:
                                options.progress_callback("LLM Pass 5", 98)
                            pass5_provider = options.pass5_provider or options.llm_provider
                            if pass5_provider is None:
                                raise ValueError("pass5_provider が未設定です")
                            if options.pass5_model:
                                model_override = options.pass5_model
                            elif options.pass5_provider and options.pass5_provider != options.llm_provider:
                                model_override = None
                            else:
                                model_override = options.llm_pass4_model or options.llm_pass3_model
                            subtitle_text = Pass5Processor(
                                provider=pass5_provider,
                                max_chars=options.pass5_max_chars,
                                model_override=model_override,
                                run_id=run_id,
                                source_name=input_path.name,
                                temperature=options.llm_temperature,
                                timeout=options.llm_timeout,
                            ).process(subtitle_text)
                        except Exception as exc:
                            logger.warning("Pass5処理に失敗しました: %s", exc)

                if subtitle_text is not None:
                    subtitle_path.parent.mkdir(parents=True, exist_ok=True)
                    subtitle_path.write_text(subtitle_text, encoding="utf-8")
                    logger.info("SRTを保存しました: %s", subtitle_path)
                    saved_paths.append(subtitle_path)

            # 処理時間を計算（save_progress_snapshot呼び出し前に計算）
            t_run_end_for_progress = time.perf_counter()
            transcribe_sec = t_transcribe_end - t_transcribe_start
            llm_two_pass_sec = (t_llm_end - t_llm_start) if options.llm_provider and result.words else 0.0
            total_elapsed_sec = t_run_end_for_progress - t_run_start
            
            save_progress_snapshot(
                run_id=run_id,
                audio_path=audio_path,
                runner_slug=slug,
                blocks=blocks,
                progress_dir=progress_dir,
                metadata=_build_progress_metadata(
                    options,
                    result.metadata,
                    timestamp,
                    subtitle_path=subtitle_path,
                    total_elapsed_sec=total_elapsed_sec,
                    stage_timings_sec={
                        "transcribe_sec": transcribe_sec,
                        "llm_two_pass_sec": llm_two_pass_sec,
                    },
                ),
                llm_provider=options.llm_provider,
                enforce_file_limit=enforce_file_limits,
            )
            t_run_end = time.perf_counter()

            # メトリクスファイル出力（LLM実行時のみ）
            if options.llm_provider:
                try:
                    from src.llm.usage_metrics import consume_usage_for_run, write_run_metrics_file

                    usage_by_pass = consume_usage_for_run(run_id)
                    # 音声ファイルの長さ（wordタイムスタンプの先頭〜末尾）
                    words = result.words or []
                    if words:
                        audio_duration = max(0.0, (words[-1].end or 0.0) - (words[0].start or 0.0))
                    else:
                        audio_duration = 0.0
                    stage_timings = {
                        "transcribe_sec": t_transcribe_end - t_transcribe_start,
                        "llm_two_pass_sec": (t_llm_end - t_llm_start) if result.words else 0.0,
                    }
                    total_elapsed = t_run_end - t_run_start
                    write_run_metrics_file(
                        run_id=run_id,
                        source_name=input_path.name,
                        runner_slug=slug,
                        timestamp=timestamp,
                        stage_timings_sec=stage_timings,
                        total_elapsed_sec=total_elapsed,
                        usage_by_pass=usage_by_pass,
                        output_dir=metrics_output_dir,
                        audio_duration_sec=audio_duration,
                    )
                except Exception as exc:  # pragma: no cover - メトリクス書き込み失敗は致命的でない
                    logger.warning("メトリクスファイルの出力に失敗しました: %s", exc)
            # Mark completion
            if options.progress_callback:
                options.progress_callback("完了", 100)

            if extracted_audio_path and not options.keep_extracted_audio:
                cleanup_extracted_audio(extracted_audio_path)
    return saved_paths


# --- helper functions shared by script/CLI ---

def build_single_block(result: TranscriptionResult) -> List[dict]:
    words = result.words or []
    start = words[0].start if words else 0.0
    end = words[-1].end if words else 0.0
    duration = max(0.0, end - start)
    return [
        {
            "index": 1,
            "text": result.text,
            "start": start,
            "end": end,
            "duration": duration,
            "sentences": [],
        }
    ]


def save_result(
    result: TranscriptionResult,
    dest: Path,
    *,
    blocks: List[dict] | None = None,
    enforce_file_limit: bool = True,
) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    payload = result.to_dict()
    if blocks is not None:
        payload["blocks"] = blocks
    dest.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    logger.info("結果を保存しました: %s", dest)
    if enforce_file_limit:
        _enforce_poc_samples_file_limit(dest.parent, max_files=5)


def save_progress_snapshot(
    *,
    run_id: str,
    audio_path: Path,
    runner_slug: str,
    blocks: List[dict],
    progress_dir: Path,
    metadata: dict,
    llm_provider: str | None = None,
    enforce_file_limit: bool = True,
) -> None:
    record = create_progress_record(
        run_id=run_id,
        audio_file=str(audio_path),
        model=runner_slug,
        total_blocks=len(blocks),
        llm_provider=llm_provider,
        metadata=metadata,
    )
    if blocks:
        mark_run_status(record, "running")
        for idx in range(len(blocks)):
            mark_block_completed(record, idx + 1)
    mark_run_status(record, "completed")
    save_path = progress_dir / f"{run_id}.json"
    progress_dir.mkdir(parents=True, exist_ok=True)
    save_progress(record, save_path)
    logger.info("進捗ファイルを保存しました: %s", save_path)
    if enforce_file_limit:
        _enforce_progress_file_limit(progress_dir, max_files=5)


class ResumeCompletedError(RuntimeError):
    """Raised when attempting to resume an already completed run."""


def prepare_resume_run(
    progress_path: Path,
    *,
    base_options: PocRunOptions,
) -> Tuple[ProgressRecord, List[Path], List[str], PocRunOptions]:
    record = load_progress(progress_path)
    if record.status == "completed":
        raise ResumeCompletedError(f"{progress_path} は既に完了済みです")
    audio_files = ensure_audio_files([Path(record.audio_file)])
    timestamp = _extract_timestamp(record)
    option_meta = (record.metadata or {}).get("options", {})
    raw_start_delay = option_meta.get("start_delay", base_options.start_delay)
    try:
        start_delay = float(raw_start_delay)
    except (TypeError, ValueError):
        start_delay = float(base_options.start_delay)
    resume_options = PocRunOptions(
        language=base_options.language or option_meta.get("language"),
        chunk_size=(
            base_options.chunk_size
            if base_options.chunk_size is not None
            else option_meta.get("chunk_size")
        ),
        output_dir=base_options.output_dir,
        progress_dir=base_options.progress_dir,
        simulate=option_meta.get("simulate", base_options.simulate),
        verbose=base_options.verbose,
        timestamp=timestamp,
        resume_source=progress_path,
        llm_provider=base_options.llm_provider or option_meta.get("llm_provider"),
        llm_profile=option_meta.get("llm_profile"),
        workflow=option_meta.get("workflow", base_options.workflow),
        llm_pass1_model=option_meta.get("llm_pass1_model"),
        llm_pass2_model=option_meta.get("llm_pass2_model"),
        llm_pass3_model=option_meta.get("llm_pass3_model"),
        llm_pass4_model=option_meta.get("llm_pass4_model"),
        rewrite=(
            base_options.rewrite
            if base_options.rewrite is not None
            else option_meta.get("rewrite")
        ),
        llm_temperature=(
            base_options.llm_temperature
            if base_options.llm_temperature is not None
            else option_meta.get("llm_temperature")
        ),
        llm_timeout=(
            base_options.llm_timeout if base_options.llm_timeout is not None else option_meta.get("llm_timeout")
        ),
        glossary_terms=option_meta.get("glossary_terms"),
        start_delay=start_delay,
        keep_extracted_audio=bool(option_meta.get("keep_extracted_audio", base_options.keep_extracted_audio)),
        save_logs=bool(option_meta.get("save_logs", base_options.save_logs)),
        enable_pass5=option_meta.get("enable_pass5", base_options.enable_pass5),
        pass5_max_chars=option_meta.get("pass5_max_chars", base_options.pass5_max_chars),
        pass5_provider=option_meta.get("pass5_provider") or base_options.pass5_provider,
        pass5_model=option_meta.get("pass5_model"),
    )
    return record, audio_files, [record.model], resume_options


def _enforce_progress_file_limit(progress_dir: Path, max_files: int = 5) -> None:
    """progress_dir（通常は temp/progress）内の進捗ファイル数を max_files までに保ち、
    古いものから削除して増え続けないようにする。

    - 対象: progress_dir 直下の `*.json`
    - 判定基準: ファイルの更新時刻（古い順）
    """
    try:
        if not progress_dir.exists():
            return
        files = sorted(
            progress_dir.glob("*.json"),
            key=lambda p: p.stat().st_mtime,
        )
        if len(files) <= max_files:
            return
        for old_path in files[:-max_files]:
            try:
                old_path.unlink()
                logger.info("古い進捗ファイルを削除しました: %s", old_path)
            except Exception as exc:  # pragma: no cover - ログのみ
                logger.warning("進捗ファイルの削除に失敗しました: %s (%s)", old_path, exc)
    except Exception as exc:  # pragma: no cover - ここで落ちないようにする
        logger.warning("進捗ファイル上限チェックに失敗しました: %s", exc)


def _enforce_poc_samples_file_limit(output_dir: Path, max_files: int = 5) -> None:
    """poc_samples（通常は temp/poc_samples）内のJSON数を max_files までに保ち、
    古いものから削除して増え続けないようにする。

    - 対象: output_dir 直下の `*.json`
    - 判定基準: ファイルの更新時刻（古い順）
    """
    try:
        if not output_dir.exists():
            return
        files = sorted(
            output_dir.glob("*.json"),
            key=lambda p: p.stat().st_mtime,
        )
        if len(files) <= max_files:
            return
        for old_path in files[:-max_files]:
            try:
                old_path.unlink()
                logger.info("古いPoCサンプルファイルを削除しました: %s", old_path)
            except Exception as exc:  # pragma: no cover - ログのみ
                logger.warning("PoCサンプルファイルの削除に失敗しました: %s (%s)", old_path, exc)
    except Exception as exc:  # pragma: no cover
        logger.warning("PoCサンプルファイル上限チェックに失敗しました: %s", exc)


def _extract_timestamp(record: ProgressRecord) -> str | None:
    meta_ts = (record.metadata or {}).get("requested_at")
    if meta_ts:
        return meta_ts
    parts = record.run_id.rsplit("_", 1)
    if len(parts) == 2:
        return parts[1]
    return None


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


def _build_progress_metadata(
    options: PocRunOptions,
    base_metadata: Dict[str, Any],
    timestamp: str,
    *,
    subtitle_path: Path | None = None,
    total_elapsed_sec: float | None = None,
    stage_timings_sec: Dict[str, float] | None = None,
) -> Dict[str, Any]:
    metadata: Dict[str, Any] = dict(base_metadata or {})
    metadata.setdefault("requested_at", timestamp)
    metadata["options"] = {
        "language": options.language,
        "chunk_size": options.chunk_size,
        "simulate": options.simulate,
        "llm_provider": options.llm_provider,
        "llm_profile": options.llm_profile,
        "workflow": options.workflow,
        "llm_pass1_model": options.llm_pass1_model,
        "llm_pass2_model": options.llm_pass2_model,
        "llm_pass3_model": options.llm_pass3_model,
        "llm_pass4_model": options.llm_pass4_model,
        "rewrite": options.rewrite,
        "llm_temperature": options.llm_temperature,
        "llm_timeout": options.llm_timeout,
        "glossary_terms": options.glossary_terms,
        "start_delay": options.start_delay,
        "keep_extracted_audio": options.keep_extracted_audio,
        "save_logs": options.save_logs,
        "enable_pass5": options.enable_pass5,
        "pass5_max_chars": options.pass5_max_chars,
        "pass5_provider": options.pass5_provider,
        "pass5_model": options.pass5_model,
    }
    if options.resume_source:
        metadata["resume_source"] = str(options.resume_source)
    if subtitle_path:
        metadata["subtitle_path"] = str(subtitle_path)
    # 処理時間情報を追加
    if total_elapsed_sec is not None:
        metadata["total_elapsed_sec"] = round(total_elapsed_sec, 2)
        metadata["total_elapsed_time"] = _format_seconds(total_elapsed_sec)
    if stage_timings_sec:
        metadata["stage_timings_sec"] = {k: round(v, 2) for k, v in stage_timings_sec.items()}
        metadata["stage_timings_time"] = {k: _format_seconds(v) for k, v in stage_timings_sec.items()}
    return metadata


__all__ = [
    "PocRunOptions",
    "execute_poc_run",
    "resolve_models",
    "ensure_audio_files",
    "list_models",
    "prepare_resume_run",
    "ResumeCompletedError",
]

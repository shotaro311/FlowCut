"""Shared helpers for the Phase 1 PoC transcription flow."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from src.alignment import align_to_srt
from src.llm.formatter import (
    FormatterError,
    FormatterRequest,
    FormatValidationError,
    LLMFormatter,
    FormattedLine,
)
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
    rewrite: bool | None = None
    llm_temperature: float | None = None
    llm_timeout: float | None = None
    align_kwargs: Dict[str, Any] = field(default_factory=dict)

    def normalized_timestamp(self) -> str:
        return self.timestamp or datetime.now().strftime("%Y%m%dT%H%M%S")


def resolve_models(raw: str | None) -> List[str]:
    """Return sorted list of runner slugs (all if raw is None)."""
    if not raw:
        # デフォルトはクラウドWhisperを避け、ローカルMLX large-v3のみを使用
        runners = available_runners()
        return [slug for slug in runners if slug == "mlx"] or runners
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
    *,
    formatter: LLMFormatter | None = None,
) -> List[Path]:
    """Execute transcription for the provided audio files and models."""
    if not audio_files:
        raise ValueError("audio_files は1件以上指定してください")
    if not models:
        raise ValueError("models は1件以上指定してください")

    timestamp = options.normalized_timestamp()
    saved_paths: List[Path] = []
    # LLM整形は本番API応答のゆらぎを許容するため、デフォルトは strict=False にして例外で止まらないようにする。
    formatter = formatter or LLMFormatter(strict_validation=False)

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
        for audio_path in audio_files:
            result = runner.transcribe(audio_path, config)
            blocks = build_single_block(result)
            run_id = f"{audio_path.stem}_{slug}_{timestamp}"
            output_path = options.output_dir / f"{run_id}.json"
            save_result(result, output_path, blocks=blocks)

            subtitle_path: Path | None = None
            if options.llm_provider:
                formatted_lines = _format_blocks(
                    blocks=blocks,
                    formatter=formatter,
                    llm_provider=options.llm_provider,
                    rewrite=options.rewrite or False,
                    run_id=run_id,
                    llm_temperature=options.llm_temperature,
                    llm_timeout=options.llm_timeout,
                )
                if formatted_lines:
                    subtitle_path = options.subtitle_dir / f"{run_id}.srt"
                    subtitle_text = align_to_srt(
                        formatted_lines,
                        result.words,
                        **(options.align_kwargs or {}),
                    )
                    subtitle_path.parent.mkdir(parents=True, exist_ok=True)
                    subtitle_path.write_text(subtitle_text, encoding="utf-8")
                    logger.info("SRTを保存しました: %s", subtitle_path)
                else:
                    logger.warning("LLM整形結果が空のためSRTを生成しませんでした: %s", run_id)

            save_progress_snapshot(
                run_id=run_id,
                audio_path=audio_path,
                runner_slug=slug,
                blocks=blocks,
                progress_dir=options.progress_dir,
                metadata=_build_progress_metadata(
                    options,
                    result.metadata,
                    timestamp,
                    subtitle_path=subtitle_path,
                ),
                llm_provider=options.llm_provider,
            )
            saved_paths.append(output_path)
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


def save_result(result: TranscriptionResult, dest: Path, *, blocks: List[dict] | None = None) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    payload = result.to_dict()
    if blocks is not None:
        payload["blocks"] = blocks
    dest.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    logger.info("結果を保存しました: %s", dest)


def save_progress_snapshot(
    *,
    run_id: str,
    audio_path: Path,
    runner_slug: str,
    blocks: List[dict],
    progress_dir: Path,
    metadata: dict,
    llm_provider: str | None = None,
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
        align_kwargs=(
            base_options.align_kwargs if base_options.align_kwargs else option_meta.get("align_kwargs") or {}
        ),
    )
    return record, audio_files, [record.model], resume_options


def _extract_timestamp(record: ProgressRecord) -> str | None:
    meta_ts = (record.metadata or {}).get("requested_at")
    if meta_ts:
        return meta_ts
    parts = record.run_id.rsplit("_", 1)
    if len(parts) == 2:
        return parts[1]
    return None


def _build_progress_metadata(
    options: PocRunOptions,
    base_metadata: Dict[str, Any],
    timestamp: str,
    *,
    subtitle_path: Path | None = None,
) -> Dict[str, Any]:
    metadata: Dict[str, Any] = dict(base_metadata or {})
    metadata.setdefault("requested_at", timestamp)
    metadata["options"] = {
        "language": options.language,
        "chunk_size": options.chunk_size,
        "simulate": options.simulate,
        "llm_provider": options.llm_provider,
        "rewrite": options.rewrite,
        "llm_temperature": options.llm_temperature,
        "llm_timeout": options.llm_timeout,
        "align_kwargs": options.align_kwargs,
    }
    if options.resume_source:
        metadata["resume_source"] = str(options.resume_source)
    if subtitle_path:
        metadata["subtitle_path"] = str(subtitle_path)
    return metadata


def _format_blocks(
    *,
    blocks: Sequence[dict],
    formatter: LLMFormatter,
    llm_provider: str,
    rewrite: bool,
    run_id: str,
    llm_temperature: float | None,
    llm_timeout: float | None,
) -> List[FormattedLine]:
    formatted: List[FormattedLine] = []
    for idx, block in enumerate(blocks, start=1):
        request = FormatterRequest(
            block_text=str(block.get("text", "")),
            provider=llm_provider,
            rewrite=rewrite,
            temperature=llm_temperature,
            timeout=llm_timeout,
            metadata={"run_id": run_id, "block_index": idx},
        )
        try:
            result = formatter.format_block(request)
        except FormatValidationError as exc:
            logger.warning("LLM出力にバリデーションエラー (block=%s): %s", idx, exc)
            continue
        except FormatterError as exc:
            logger.error("LLM整形に失敗しました (block=%s): %s", idx, exc)
            continue
        formatted.extend(result.lines)
    return formatted


__all__ = [
    "PocRunOptions",
    "execute_poc_run",
    "resolve_models",
    "ensure_audio_files",
    "list_models",
    "prepare_resume_run",
    "ResumeCompletedError",
]

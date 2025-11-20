"""Shared helpers for the Phase 1 PoC transcription flow."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
import logging
from pathlib import Path
from typing import Iterable, List, Sequence

from src.blocking.builders import sentences_from_words
from src.blocking.splitter import Block, BlockSplitter
from src.transcribe import (
    TranscriptionConfig,
    TranscriptionResult,
    available_runners,
    describe_runners,
    get_runner,
)
from src.utils.progress import (
    create_progress_record,
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
    simulate: bool = True
    verbose: bool = False
    timestamp: str | None = None

    def normalized_timestamp(self) -> str:
        return self.timestamp or datetime.now().strftime("%Y%m%dT%H%M%S")


def resolve_models(raw: str | None) -> List[str]:
    """Return sorted list of runner slugs (all if raw is None)."""
    if not raw:
        return available_runners()
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
    splitter = BlockSplitter()
    saved_paths: List[Path] = []

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
            sentences = sentences_from_words(result.words, fallback_text=result.text)
            blocks = splitter.split(sentences)
            block_payload = build_block_payload(blocks)
            run_id = f"{audio_path.stem}_{slug}_{timestamp}"
            output_path = options.output_dir / f"{run_id}.json"
            save_result(result, output_path, blocks=block_payload)
            save_progress_snapshot(
                run_id=run_id,
                audio_path=audio_path,
                runner_slug=slug,
                blocks=block_payload,
                progress_dir=options.progress_dir,
                metadata={"requested_at": timestamp, **result.metadata},
            )
            saved_paths.append(output_path)
    return saved_paths


# --- helper functions shared by script/CLI ---

def build_block_payload(blocks: List[Block]) -> List[dict]:
    payload: List[dict] = []
    for idx, block in enumerate(blocks, start=1):
        payload.append(
            {
                "index": idx,
                "text": block.text,
                "start": block.start,
                "end": block.end,
                "duration": block.duration,
                "sentences": [
                    {
                        "text": sentence.text,
                        "start": sentence.start,
                        "end": sentence.end,
                        "overlap": sentence.overlap,
                    }
                    for sentence in block.sentences
                ],
            }
        )
    return payload


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
) -> None:
    record = create_progress_record(
        run_id=run_id,
        audio_file=str(audio_path),
        model=runner_slug,
        total_blocks=len(blocks),
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


__all__ = [
    "PocRunOptions",
    "execute_poc_run",
    "resolve_models",
    "ensure_audio_files",
    "list_models",
]

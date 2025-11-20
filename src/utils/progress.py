"""Progress tracking helpers for Flow Cut CLI/PoC."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime
import json
from pathlib import Path
from typing import Any, Dict, List

_RUN_STATUS = {"pending", "running", "completed", "failed"}
_BLOCK_STATUS = {"pending", "in_progress", "completed"}


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


@dataclass(slots=True)
class BlockProgress:
    index: int
    status: str = "pending"
    started_at: str | None = None
    completed_at: str | None = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ProgressRecord:
    run_id: str
    audio_file: str
    model: str
    total_blocks: int
    llm_provider: str | None = None
    status: str = "pending"
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)
    completed_blocks: int = 0
    blocks: List[BlockProgress] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["blocks"] = [block.to_dict() for block in self.blocks]
        return data


def _validate_run_status(status: str) -> None:
    if status not in _RUN_STATUS:
        raise ValueError(f"不正なステータスです: {status}")


def _validate_block_status(status: str) -> None:
    if status not in _BLOCK_STATUS:
        raise ValueError(f"不正なブロックステータスです: {status}")


def create_progress_record(
    *,
    run_id: str,
    audio_file: str,
    model: str,
    total_blocks: int,
    llm_provider: str | None = None,
    metadata: Dict[str, Any] | None = None,
) -> ProgressRecord:
    if total_blocks < 0:
        raise ValueError("total_blocks must be >= 0")
    blocks = [BlockProgress(index=i + 1) for i in range(total_blocks)]
    return ProgressRecord(
        run_id=run_id,
        audio_file=audio_file,
        model=model,
        total_blocks=total_blocks,
        llm_provider=llm_provider,
        blocks=blocks,
        metadata=metadata or {},
    )


def mark_block_completed(record: ProgressRecord, block_index: int) -> None:
    if not 1 <= block_index <= len(record.blocks):
        raise IndexError("block_index is out of range")
    block = record.blocks[block_index - 1]
    block.status = "completed"
    if block.started_at is None:
        block.started_at = _now()
    block.completed_at = _now()
    record.completed_blocks = sum(1 for b in record.blocks if b.status == "completed")
    record.updated_at = _now()


def mark_run_status(record: ProgressRecord, status: str) -> None:
    _validate_run_status(status)
    record.status = status
    record.updated_at = _now()


def save_progress(record: ProgressRecord, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(record.to_dict(), ensure_ascii=False, indent=2))


def load_progress(path: Path) -> ProgressRecord:
    data = json.loads(path.read_text())
    blocks = [
        BlockProgress(
            index=item.get("index", idx + 1),
            status=item.get("status", "pending"),
            started_at=item.get("started_at"),
            completed_at=item.get("completed_at"),
        )
        for idx, item in enumerate(data.get("blocks", []))
    ]
    record = ProgressRecord(
        run_id=data["run_id"],
        audio_file=data.get("audio_file", ""),
        model=data.get("model", ""),
        total_blocks=data.get("total_blocks", len(blocks)),
        llm_provider=data.get("llm_provider"),
        status=data.get("status", "pending"),
        created_at=data.get("created_at", _now()),
        updated_at=data.get("updated_at", _now()),
        completed_blocks=data.get("completed_blocks", 0),
        blocks=blocks,
        metadata=data.get("metadata", {}),
    )
    return record


__all__ = [
    "BlockProgress",
    "ProgressRecord",
    "create_progress_record",
    "mark_block_completed",
    "mark_run_status",
    "save_progress",
    "load_progress",
]

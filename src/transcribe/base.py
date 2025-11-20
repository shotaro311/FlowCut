"""音声認識ランナー共通インターフェースとレジストリ実装。

Phase 1 では PoC 用のシミュレーションモードを備えておき、
本番モデル統合時は `transcribe()` を差し替える。
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Type

logger = logging.getLogger(__name__)


class TranscriptionError(RuntimeError):
    """ランナー内部で発生したドメインエラー。"""


class RunnerNotFoundError(KeyError):
    """要求されたランナーが未登録の場合に通知する例外。"""


@dataclass(slots=True)
class WordTimestamp:
    word: str
    start: float
    end: float
    confidence: float | None = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'word': self.word,
            'start': self.start,
            'end': self.end,
            'confidence': self.confidence,
        }


@dataclass(slots=True)
class TranscriptionResult:
    text: str
    words: List[WordTimestamp] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'text': self.text,
            'words': [word.to_dict() for word in self.words],
            'metadata': self.metadata,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)


@dataclass(slots=True)
class TranscriptionConfig:
    language: str | None = None
    chunk_size: int | None = None
    simulate: bool = False
    extra: Dict[str, Any] = field(default_factory=dict)

    def with_override(self, **kwargs: Any) -> 'TranscriptionConfig':
        data = {
            'language': self.language,
            'chunk_size': self.chunk_size,
            'simulate': self.simulate,
            'extra': {**self.extra},
        }
        data.update(kwargs)
        return TranscriptionConfig(**data)


class BaseTranscribeRunner:
    """全ランナーの基底クラス。"""

    slug: str = 'base'
    display_name: str = 'Base Transcribe Runner'
    default_model: str = 'n/a'
    requires_gpu: bool = False

    def prepare(self, config: TranscriptionConfig) -> None:  # pragma: no cover - デフォルトは何もしない
        logger.debug('prepare skipped for %s', self.slug)

    def transcribe(self, audio_path: Path, config: TranscriptionConfig) -> TranscriptionResult:
        raise NotImplementedError('各Runnerで実装してください')

    # --- 共通ユーティリティ ---
    def simulate_transcription(self, audio_path: Path, config: TranscriptionConfig) -> TranscriptionResult:
        """PoC 向けの簡易書き起こしを生成する。"""
        words: List[WordTimestamp] = []
        base_time = 0.0
        pseudo_text = f"[{self.slug}] Simulated transcript for {audio_path.stem or 'audio'}"
        for raw_word in pseudo_text.split():
            duration = max(0.18, min(0.65, 0.18 + len(raw_word) * 0.03))
            start = round(base_time, 2)
            end = round(start + duration, 2)
            base_time = end + 0.04
            words.append(WordTimestamp(word=raw_word.strip('[]'), start=start, end=end, confidence=0.5))
        metadata = {
            'runner': self.slug,
            'model': self.default_model,
            'audio_file': str(audio_path),
            'simulate': True,
        }
        metadata.update(config.extra)
        return TranscriptionResult(text=pseudo_text, words=words, metadata=metadata)


_RUNNER_REGISTRY: Dict[str, Type[BaseTranscribeRunner]] = {}


def register_runner(cls: Type[BaseTranscribeRunner]) -> Type[BaseTranscribeRunner]:
    if not cls.slug or cls.slug == 'base':
        raise ValueError('slug を固有値に設定してください')
    if cls.slug in _RUNNER_REGISTRY:
        logger.warning('Runner %s はすでに登録済みです。上書きします。', cls.slug)
    _RUNNER_REGISTRY[cls.slug] = cls
    return cls


def available_runners() -> List[str]:
    return sorted(_RUNNER_REGISTRY.keys())


def get_runner(slug: str) -> BaseTranscribeRunner:
    try:
        runner_cls = _RUNNER_REGISTRY[slug]
    except KeyError as exc:  # pragma: no cover - 単純なラッパー
        raise RunnerNotFoundError(f'Runner "{slug}" は未登録です: {available_runners()}') from exc
    return runner_cls()


def describe_runners() -> List[Dict[str, Any]]:
    data: List[Dict[str, Any]] = []
    for slug in available_runners():
        runner_cls = _RUNNER_REGISTRY[slug]
        data.append(
            {
                'slug': slug,
                'display_name': runner_cls.display_name,
                'default_model': runner_cls.default_model,
                'requires_gpu': runner_cls.requires_gpu,
            }
        )
    return data


__all__ = [
    'BaseTranscribeRunner',
    'TranscriptionConfig',
    'TranscriptionResult',
    'TranscriptionError',
    'RunnerNotFoundError',
    'WordTimestamp',
    'register_runner',
    'get_runner',
    'available_runners',
    'describe_runners',
]

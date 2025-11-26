"""Shared MLX Whisper helpers."""
from __future__ import annotations

import importlib
import logging
from pathlib import Path
from typing import Any, Dict, List, Sequence

from .base import BaseTranscribeRunner, TranscriptionConfig, TranscriptionError, TranscriptionResult, WordTimestamp

logger = logging.getLogger(__name__)


def _load_mlx_whisper():
    try:
        return importlib.import_module("mlx_whisper")
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise TranscriptionError(
            f"mlx-whisper のロードに失敗しました: {exc!r}"
        ) from exc


def _parse_segments(words: Sequence[Dict[str, Any]]) -> List[WordTimestamp]:
    parsed: List[WordTimestamp] = []
    for w in words:
        text = str(w.get("word", "")).strip()
        if not text:
            continue
        try:
            start = float(w.get("start"))
            end = float(w.get("end"))
        except (TypeError, ValueError):
            continue
        confidence = w.get("probability") or w.get("confidence")
        try:
            conf_val = float(confidence) if confidence is not None else None
        except (TypeError, ValueError):
            conf_val = None
        parsed.append(WordTimestamp(word=text, start=start, end=end, confidence=conf_val))
    return parsed


def run_mlx_whisper(
    audio_path: Path,
    *,
    model_id: str,
    language: str | None,
) -> TranscriptionResult:
    mlx_whisper = _load_mlx_whisper()
    if not audio_path.exists():
        raise FileNotFoundError(f"音声ファイルが見つかりません: {audio_path}")

    logger.info("[mlx-whisper] model=%s language=%s", model_id, language or "auto")
    try:
        output = mlx_whisper.transcribe(
            str(audio_path),
            path_or_hf_repo=model_id,
            word_timestamps=True,
            language=language,
            # ローカル実行を安定・高速化するための設定
            verbose=False,
            temperature=0.0,  # 単一温度でデコードを固定（リトライでの多段温度を避ける）
            condition_on_previous_text=False,  # ループ防止・速度優先
        )
    except Exception as exc:  # pragma: no cover - upstream error surface
        raise TranscriptionError(f"mlx-whisper 実行に失敗しました: {exc}") from exc

    text = output.get("text", "")
    segments = output.get("segments") or []
    words: List[WordTimestamp] = []
    for seg in segments:
        words.extend(_parse_segments(seg.get("words", [])))
    if not words and output.get("words"):
        words = _parse_segments(output["words"])

    metadata = {
        "runner": "mlx",
        "model": model_id,
        "language": language,
        "simulate": False,
    }
    return TranscriptionResult(text=text, words=words, metadata=metadata)


class BaseMlxRunner(BaseTranscribeRunner):
    """MLX Whisper ランナーの基底クラス。"""
    
    requires_gpu = True

    def transcribe(self, audio_path: Path, config: TranscriptionConfig) -> TranscriptionResult:
        if config.simulate:
            return self.simulate_transcription(audio_path, config)
        
        result = run_mlx_whisper(audio_path, model_id=self.default_model, language=config.language)
        
        # メタデータのrunnerを自身のslugで上書き
        result.metadata["runner"] = self.slug
        return result


__all__ = ["run_mlx_whisper"]

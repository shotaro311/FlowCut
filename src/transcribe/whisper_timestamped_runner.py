"""whisper-timestamped を使用したローカルランナー。

openai-whisper ベースで、より精度の高いワードレベルタイムスタンプを生成します。
VAD（音声検出）を使用してハルシネーションを防止し、
Cross-Attentionを活用して正確なタイムスタンプを推定します。
"""
from __future__ import annotations

import importlib
import logging
from pathlib import Path
from typing import Any, Dict, List, Sequence

from .base import (
    BaseTranscribeRunner,
    TranscriptionConfig,
    TranscriptionError,
    TranscriptionResult,
    WordTimestamp,
    register_runner,
)

logger = logging.getLogger(__name__)


def _load_whisper_timestamped():
    """whisper_timestampedモジュールをロードする。"""
    try:
        return importlib.import_module("whisper_timestamped")
    except ImportError as exc:
        raise TranscriptionError(
            f"whisper-timestamped のロードに失敗しました: {exc!r}"
        ) from exc


def _parse_segments(words: Sequence[Dict[str, Any]]) -> List[WordTimestamp]:
    """Whisperのwords出力をWordTimestampリストに変換する。"""
    parsed: List[WordTimestamp] = []
    for w in words:
        text = str(w.get("word") or w.get("text", "")).strip()
        if not text:
            continue
        try:
            start = float(w.get("start"))
            end = float(w.get("end"))
        except (TypeError, ValueError):
            continue
        confidence = w.get("confidence") or w.get("probability")
        try:
            conf_val = float(confidence) if confidence is not None else None
        except (TypeError, ValueError):
            conf_val = None
        parsed.append(WordTimestamp(word=text, start=start, end=end, confidence=conf_val))
    return parsed


def run_whisper_timestamped(
    audio_path: Path,
    *,
    model_name: str,
    language: str | None,
) -> TranscriptionResult:
    """whisper-timestampedでローカル実行し、FlowCut形式の結果を返す。"""
    whisper = _load_whisper_timestamped()
    if not audio_path.exists():
        raise FileNotFoundError(f"音声ファイルが見つかりません: {audio_path}")

    logger.info("[whisper-timestamped] model=%s language=%s", model_name, language or "auto")
    
    try:
        # モデルをロード
        model = whisper.load_model(model_name)
        
        # 文字起こし実行（whisper-timestamped は関数呼び出し形式）
        output = whisper.transcribe(
            model,
            str(audio_path),
            language=language,
            verbose=False,
            # whisper-timestamped 固有オプション
            vad=False,  # VADはsilero依存のためオフ（依存関係追加で有効化可能）
            detect_disfluencies=False,  # 言いよどみ検出（オフで高速化）
        )
    except Exception as exc:
        raise TranscriptionError(f"whisper-timestamped 実行に失敗しました: {exc}") from exc

    # segments[*].words をフラット化
    text = output.get("text", "")
    segments = output.get("segments") or []
    words: List[WordTimestamp] = []
    for seg in segments:
        words.extend(_parse_segments(seg.get("words", [])))
    # フォールバック
    if not words and output.get("words"):
        words = _parse_segments(output["words"])

    metadata = {
        "runner": "whisper-ts",
        "model": model_name,
        "language": output.get("language") or language,
        "simulate": False,
    }
    return TranscriptionResult(text=text, words=words, metadata=metadata)


@register_runner
class WhisperTimestampedRunner(BaseTranscribeRunner):
    """whisper-timestamped ローカル実行ランナー（高精度タイムスタンプ）。"""
    
    slug = 'whisper-ts'
    display_name = 'Whisper Timestamped Large-v3'
    default_model = 'large-v3'
    requires_gpu = False

    def transcribe(self, audio_path: Path, config: TranscriptionConfig) -> TranscriptionResult:
        if config.simulate:
            return self.simulate_transcription(audio_path, config)
        
        result = run_whisper_timestamped(audio_path, model_name=self.default_model, language=config.language)
        result.metadata["runner"] = self.slug
        return result

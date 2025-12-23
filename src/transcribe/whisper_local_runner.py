"""本家 OpenAI Whisper ローカル実行ランナー。

openai-whisper パッケージを利用してローカルでWhisperを実行します。
MLX Whisperと同じ変換ロジックを使用してFlowCut形式に変換します。
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


def _load_whisper():
    """本家whisperモジュールをロードする。"""
    try:
        return importlib.import_module("whisper")
    except ImportError as exc:
        raise TranscriptionError(
            f"openai-whisper のロードに失敗しました: {exc!r}"
        ) from exc


def _parse_segments(words: Sequence[Dict[str, Any]]) -> List[WordTimestamp]:
    """Whisperのwords出力をWordTimestampリストに変換する。
    
    probability キーを confidence に変換します。
    """
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
        # 本家Whisperは probability, MLXは probability または confidence
        confidence = w.get("probability") or w.get("confidence")
        try:
            conf_val = float(confidence) if confidence is not None else None
        except (TypeError, ValueError):
            conf_val = None
        parsed.append(WordTimestamp(word=text, start=start, end=end, confidence=conf_val))
    return parsed


def run_whisper_local(
    audio_path: Path,
    *,
    model_name: str,
    language: str | None,
) -> TranscriptionResult:
    """本家Whisperをローカルで実行し、FlowCut形式の結果を返す。"""
    whisper = _load_whisper()
    if not audio_path.exists():
        raise FileNotFoundError(f"音声ファイルが見つかりません: {audio_path}")

    logger.info("[whisper-local] model=%s language=%s", model_name, language or "auto")
    
    try:
        # モデルをロード
        model = whisper.load_model(model_name)

        # 文字起こし実行（ワードレベルタイムスタンプ有効）
        # 認識精度向上のためのパラメータ最適化:
        # - hallucination_silence_threshold: 無音区間でのhallucination（同じフレーズの繰り返し）を防止
        # - logprob_threshold: 低確率トークンを除外して誤認識を減らす
        # - compression_ratio_threshold: 繰り返し検出を厳格化（推奨値2.4）
        # - beam_size/best_of: デコーディング品質向上
        output = model.transcribe(
            str(audio_path),
            word_timestamps=True,
            language=language,
            verbose=False,
            temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),  # フォールバック温度
            condition_on_previous_text=True,  # 文脈を考慮
            no_speech_threshold=0.6,  # 無音判定（推奨値）
            compression_ratio_threshold=2.4,  # 繰り返し検出を厳格化（推奨値）
            logprob_threshold=-1.0,  # 低確率トークンを除外
            hallucination_silence_threshold=2.0,  # 無音区間でのhallucination防止
            beam_size=5,  # ビームサーチサイズ
            best_of=5,  # 複数候補から最良を選択
        )
    except Exception as exc:
        raise TranscriptionError(f"whisper 実行に失敗しました: {exc}") from exc

    # segments[*].words をフラット化
    text = output.get("text", "")
    segments = output.get("segments") or []
    words: List[WordTimestamp] = []
    for seg in segments:
        words.extend(_parse_segments(seg.get("words", [])))
    # フォールバック: トップレベルにwordsがある場合
    if not words and output.get("words"):
        words = _parse_segments(output["words"])

    metadata = {
        "runner": "whisper-local",
        "model": model_name,
        "language": output.get("language") or language,
        "simulate": False,
    }
    return TranscriptionResult(text=text, words=words, metadata=metadata)


@register_runner
class WhisperLocalRunner(BaseTranscribeRunner):
    """本家OpenAI Whisper large-v3 ローカル実行ランナー。"""
    
    slug = 'whisper-local'
    display_name = 'Whisper Large-v3 (Local)'
    default_model = 'large-v3'
    requires_gpu = False  # CPUでも動作可能（遅いが）

    def transcribe(self, audio_path: Path, config: TranscriptionConfig) -> TranscriptionResult:
        if config.simulate:
            return self.simulate_transcription(audio_path, config)
        
        result = run_whisper_local(audio_path, model_name=self.default_model, language=config.language)
        
        # メタデータのrunnerを自身のslugで上書き
        result.metadata["runner"] = self.slug
        return result

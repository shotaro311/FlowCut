"""OpenAI Whisper ランナー（ローカル実行版）。

Windows では OpenAI の Whisper モデル（large-v3 系）を
ローカルで実行することを前提とする。

* 依存ライブラリ: `pip install openai-whisper`
* 実体の import 名: `import whisper`
"""
from __future__ import annotations

import importlib
import logging
from pathlib import Path
from typing import Any, Dict, List

from .base import (
    BaseTranscribeRunner,
    TranscriptionConfig,
    TranscriptionError,
    TranscriptionResult,
    WordTimestamp,
    register_runner,
)

logger = logging.getLogger(__name__)


_MODEL_CACHE: Dict[str, Any] = {}


def _load_whisper() -> Any:
    """openai-whisper (whisper) を動的に import する。

    未インストールの場合は、ユーザーに pip コマンドを案内する。
    """
    try:
        return importlib.import_module("whisper")
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise TranscriptionError(
            "openai-whisper (whisper) のロードに失敗しました。"
            " Windows では次のコマンドでインストールしてください: "
            "`pip install openai-whisper`"
        ) from exc


def _get_model(model_name: str = "large-v3") -> Any:
    """whisper.load_model をラップし、同一プロセス内ではキャッシュする。"""
    if model_name in _MODEL_CACHE:
        return _MODEL_CACHE[model_name]
    whisper = _load_whisper()
    logger.info("loading whisper local model: %s", model_name)
    model = whisper.load_model(model_name)
    _MODEL_CACHE[model_name] = model
    return model


def _parse_words_from_segments(segments: List[Dict[str, Any]]) -> List[WordTimestamp]:
    words: List[WordTimestamp] = []
    for seg in segments or []:
        for w in seg.get("words") or []:
            text = str(w.get("word", "")).strip()
            if not text:
                continue
            try:
                start = float(w.get("start"))
                end = float(w.get("end"))
            except (TypeError, ValueError):
                continue
            confidence = w.get("confidence")
            try:
                conf_val = float(confidence) if confidence is not None else None
            except (TypeError, ValueError):
                conf_val = None
            words.append(WordTimestamp(word=text, start=start, end=end, confidence=conf_val))
    return words


@register_runner
class OpenAIWhisperRunner(BaseTranscribeRunner):
    slug = "openai"
    display_name = "OpenAI Whisper large-v3 (local)"
    # openai-whisper の `whisper.available_models()` に含まれるモデル名を指定する
    # large-v3 は精度重視のデフォルトモデルとして採用する。
    default_model = "large-v3"
    requires_gpu = False

    def transcribe(self, audio_path: Path, config: TranscriptionConfig) -> TranscriptionResult:
        # 明示的にシミュレーション指定なら、ベースクラスのダミー結果を返す
        if config.simulate:
            return self.simulate_transcription(audio_path, config)

        if not audio_path.exists():
            raise FileNotFoundError(f"音声ファイルが見つかりません: {audio_path}")

        model = _get_model(self.default_model)
        logger.info("[openai-whisper-local] model=%s language=%s", self.default_model, config.language or "auto")

        # word_timestamps=True でセグメントごとの単語タイムスタンプを取得する
        try:
            result: Dict[str, Any] = model.transcribe(
                str(audio_path),
                language=config.language,
                word_timestamps=True,
            )
        except Exception as exc:  # pragma: no cover - upstream error surface
            raise TranscriptionError(f"openai-whisper ローカル実行に失敗しました: {exc}") from exc

        text = str(result.get("text", ""))
        segments = result.get("segments") or []
        words = _parse_words_from_segments(segments)

        metadata: Dict[str, Any] = {
            "runner": self.slug,
            "model": self.default_model,
            "local": True,
            "audio_file": str(audio_path),
            "simulate": False,
            "language": result.get("language") or config.language,
        }
        return TranscriptionResult(text=text, words=words, metadata=metadata)

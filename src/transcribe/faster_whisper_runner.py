"""Faster-Whisper runner (CTranslate2 backend).

This runner is intended as the default on Windows for faster local transcription.
Dependency is optional at import time; we provide a helpful message when missing.
"""
from __future__ import annotations

import importlib
import logging
import os
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


def _load_faster_whisper() -> Any:
    # Avoid OpenMP duplicate initialization abort when bundled with other native stacks (e.g. torch).
    if os.name == "nt":
        os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    try:
        return importlib.import_module("faster_whisper")
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise TranscriptionError(
            "faster-whisper のロードに失敗しました。次のコマンドでインストールしてください: "
            "`pip install faster-whisper`"
        ) from exc


def _pick_device_and_compute_type(config: TranscriptionConfig) -> tuple[str, str]:
    device = str(config.extra.get("device") or "").strip().lower()
    compute_type = str(config.extra.get("compute_type") or "").strip().lower()

    if not device:
        device = "cpu"
    if not compute_type:
        # CPU 前提のデフォルトは int8（速くて軽い）。GPU は float16 を優先。
        compute_type = "float16" if device == "cuda" else "int8"
    return device, compute_type


def _get_model(model_name: str, *, device: str, compute_type: str) -> Any:
    key = f"{model_name}|{device}|{compute_type}"
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]

    fw = _load_faster_whisper()
    WhisperModel = getattr(fw, "WhisperModel", None)
    if WhisperModel is None:  # pragma: no cover - defensive
        raise TranscriptionError("faster_whisper.WhisperModel が見つかりません。依存が壊れている可能性があります。")

    logger.info("loading faster-whisper model=%s device=%s compute_type=%s", model_name, device, compute_type)
    model = WhisperModel(model_name, device=device, compute_type=compute_type)
    _MODEL_CACHE[key] = model
    return model


def _parse_words_from_segments(segments: List[Any]) -> List[WordTimestamp]:
    words: List[WordTimestamp] = []
    for seg in segments or []:
        seg_words = getattr(seg, "words", None) or []
        for w in seg_words:
            text = str(getattr(w, "word", "")).strip()
            if not text:
                continue
            try:
                start = float(getattr(w, "start"))
                end = float(getattr(w, "end"))
            except (TypeError, ValueError):
                continue
            prob = getattr(w, "probability", None)
            confidence: float | None
            try:
                confidence = float(prob) if prob is not None else None
            except (TypeError, ValueError):
                confidence = None
            words.append(WordTimestamp(word=text, start=start, end=end, confidence=confidence))
    return words


@register_runner
class FasterWhisperRunner(BaseTranscribeRunner):
    slug = "faster"
    display_name = "Faster-Whisper (CTranslate2)"
    default_model = "large-v3-turbo"
    requires_gpu = False

    def transcribe(self, audio_path: Path, config: TranscriptionConfig) -> TranscriptionResult:
        if config.simulate:
            return self.simulate_transcription(audio_path, config)

        if not audio_path.exists():
            raise FileNotFoundError(f"音声ファイルが見つかりません: {audio_path}")

        model_name = str(config.extra.get("faster_model") or self.default_model).strip() or self.default_model
        language = config.language
        device, compute_type = _pick_device_and_compute_type(config)

        model = _get_model(model_name, device=device, compute_type=compute_type)
        logger.info("[faster-whisper] model=%s device=%s compute_type=%s language=%s", model_name, device, compute_type, language or "auto")

        try:
            segments_iter, info = model.transcribe(
                str(audio_path),
                language=language,
                word_timestamps=True,
            )
            segments = list(segments_iter)
        except Exception as exc:  # pragma: no cover - upstream error surface
            raise TranscriptionError(f"faster-whisper ローカル実行に失敗しました: {exc}") from exc

        text = "".join(str(getattr(seg, "text", "")) for seg in segments).strip()
        words = _parse_words_from_segments(segments)

        detected_language = getattr(info, "language", None)
        metadata: Dict[str, Any] = {
            "runner": self.slug,
            "model": model_name,
            "local": True,
            "audio_file": str(audio_path),
            "simulate": False,
            "language": detected_language or language,
            "device": device,
            "compute_type": compute_type,
        }
        return TranscriptionResult(text=text, words=words, metadata=metadata)


__all__ = ["FasterWhisperRunner"]

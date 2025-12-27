"""Whisper + Gemini ハイブリッド文字起こしモジュール。

Whisperの単語タイムスタンプとGeminiの高精度テキストを組み合わせて、
精度の高い字幕用データを生成する。
"""
from __future__ import annotations

from src.transcribe.hybrid.gemini_transcriber import (
    GeminiModel,
    GeminiSegment,
    GeminiTranscriber,
    ThinkingLevel,
)
from src.transcribe.hybrid.aligner import AlignmentResult, TextAligner
from src.transcribe.hybrid.merger import MergedWord, WordMerger
from src.transcribe.hybrid.processor import HybridProcessor

__all__ = [
    "GeminiModel",
    "GeminiSegment",
    "GeminiTranscriber",
    "ThinkingLevel",
    "AlignmentResult",
    "TextAligner",
    "MergedWord",
    "WordMerger",
    "HybridProcessor",
]

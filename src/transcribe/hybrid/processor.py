"""Whisper + Gemini ハイブリッド処理の統合プロセッサ。"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Callable, List, Sequence

from src.transcribe.base import TranscriptionResult, WordTimestamp
from src.transcribe.hybrid.aligner import AlignmentBlock, TextAligner
from src.transcribe.hybrid.gemini_transcriber import (
    GeminiModel,
    GeminiSegment,
    GeminiTranscriber,
    GeminiTranscriberError,
    ThinkingLevel,
)
from src.transcribe.hybrid.merger import WordMerger

logger = logging.getLogger(__name__)


class HybridProcessorError(RuntimeError):
    """ハイブリッド処理エラー"""


class HybridProcessor:
    """
    Whisper + Gemini ハイブリッド文字起こしプロセッサ。

    Whisperの単語タイムスタンプとGeminiの高精度テキストを組み合わせて、
    精度の高い字幕用データを生成する。

    処理フロー:
    1. Whisperの文字起こし結果を受け取る（外部で実行済み）
    2. Geminiで同じ音声を文字起こし
    3. 2つの結果をアライメント
    4. 低類似度の区間をGeminiテキストで置換
    5. マージされたWordTimestampを返す
    """

    def __init__(
        self,
        api_key: str | None = None,
        gemini_model: GeminiModel = GeminiModel.FLASH_3_PREVIEW,
        thinking_level: ThinkingLevel = ThinkingLevel.MEDIUM,
        similarity_threshold: float = 0.8,
        time_tolerance_sec: float = 1.0,
        language: str = "ja",
        timeout: float = 300.0,
    ):
        """
        Args:
            api_key: Google API Key（未指定の場合は環境変数から取得）
            gemini_model: 使用するGeminiモデル
            thinking_level: Gemini 3 Flash Preview用のthinking_level
            similarity_threshold: この閾値未満の類似度の場合、Geminiテキストを採用
            time_tolerance_sec: 時間範囲のマッチングに許容する誤差（秒）
            language: 文字起こし言語
            timeout: APIタイムアウト（秒）
        """
        self.gemini_transcriber = GeminiTranscriber(
            api_key=api_key,
            model=gemini_model,
            thinking_level=thinking_level,
            language=language,
            timeout=timeout,
        )
        self.aligner = TextAligner(
            similarity_threshold=similarity_threshold,
            time_tolerance_sec=time_tolerance_sec,
        )
        self.merger = WordMerger()
        self.similarity_threshold = similarity_threshold

    def process(
        self,
        audio_path: Path,
        whisper_result: TranscriptionResult,
        *,
        progress_callback: Callable[[str, int], None] | None = None,
        log_dir: Path | None = None,
        run_id: str | None = None,
    ) -> TranscriptionResult:
        """
        Whisper結果をGeminiで補正して返す。

        Args:
            audio_path: 音声ファイルのパス
            whisper_result: Whisperの文字起こし結果
            progress_callback: 進捗コールバック (メッセージ, パーセント)
            log_dir: ログ出力ディレクトリ（指定時はGemini結果と統合結果をJSONで保存）
            run_id: 実行ID（ログファイル名に使用）

        Returns:
            補正されたTranscriptionResult
        """
        whisper_words = whisper_result.words or []

        if not whisper_words:
            logger.warning("Whisper words is empty, skipping hybrid processing")
            return whisper_result

        # Step 1: Geminiで文字起こし
        if progress_callback:
            progress_callback("Gemini音声認識中", 30)

        try:
            gemini_segments = self.gemini_transcriber.transcribe(audio_path)
        except GeminiTranscriberError as exc:
            logger.error("Gemini transcription failed: %s", exc)
            logger.warning("Falling back to Whisper-only result")
            return whisper_result

        if not gemini_segments:
            logger.warning("Gemini returned no segments, using Whisper-only result")
            return whisper_result

        # Gemini結果をログ保存
        if log_dir:
            self._save_gemini_log(log_dir, run_id, gemini_segments, whisper_words)

        # Step 2: アライメント
        if progress_callback:
            progress_callback("テキストアライメント中", 35)

        alignment_blocks = self.aligner.align(whisper_words, gemini_segments)

        # アライメント統計をログ
        self._log_alignment_stats(alignment_blocks)

        # Step 3: マージ
        if progress_callback:
            progress_callback("結果マージ中", 38)

        merged_words = self.merger.merge(
            whisper_words,
            gemini_segments,
            alignment_blocks,
        )

        # メタデータを更新
        metadata = dict(whisper_result.metadata)
        metadata["hybrid_processing"] = {
            "gemini_model": self.gemini_transcriber.model.value,
            "thinking_level": (
                self.gemini_transcriber.thinking_level.value
                if self.gemini_transcriber.thinking_level
                else None
            ),
            "similarity_threshold": self.similarity_threshold,
            "original_word_count": len(whisper_words),
            "gemini_segment_count": len(gemini_segments),
            "merged_word_count": len(merged_words),
            "blocks_using_gemini": sum(
                1 for b in alignment_blocks if b.action == "use_gemini"
            ),
            "blocks_using_whisper": sum(
                1 for b in alignment_blocks if b.action == "keep_whisper"
            ),
        }

        # 結合テキストを生成
        merged_text = "".join(w.word for w in merged_words)

        # 統合結果をログ保存
        if log_dir:
            self._save_merged_log(
                log_dir, run_id, whisper_words, gemini_segments,
                alignment_blocks, merged_words, merged_text
            )

        return TranscriptionResult(
            text=merged_text,
            words=merged_words,
            metadata=metadata,
        )

    def _save_gemini_log(
        self,
        log_dir: Path,
        run_id: str | None,
        gemini_segments: List[GeminiSegment],
        whisper_words: Sequence[WordTimestamp],
    ) -> None:
        """Gemini文字起こし結果をログファイルに保存。"""
        log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix = f"{run_id}_" if run_id else ""
        filename = f"{prefix}gemini_transcription_{timestamp}.json"
        filepath = log_dir / filename

        gemini_text = "".join(s.text for s in gemini_segments)
        whisper_text = "".join(w.word for w in whisper_words)

        log_data = {
            "type": "gemini_transcription",
            "timestamp": datetime.now().isoformat(),
            "run_id": run_id,
            "gemini": {
                "model": self.gemini_transcriber.model.value,
                "thinking_level": (
                    self.gemini_transcriber.thinking_level.value
                    if self.gemini_transcriber.thinking_level
                    else None
                ),
                "text": gemini_text,
                "segment_count": len(gemini_segments),
                "segments": [
                    {
                        "text": s.text,
                        "start_sec": s.start_sec,
                        "end_sec": s.end_sec,
                    }
                    for s in gemini_segments
                ],
            },
            "whisper_comparison": {
                "text": whisper_text,
                "word_count": len(whisper_words),
            },
        }

        filepath.write_text(json.dumps(log_data, ensure_ascii=False, indent=2))
        logger.info("Gemini transcription log saved: %s", filepath)

    def _save_merged_log(
        self,
        log_dir: Path,
        run_id: str | None,
        whisper_words: Sequence[WordTimestamp],
        gemini_segments: List[GeminiSegment],
        alignment_blocks: Sequence[AlignmentBlock],
        merged_words: List[WordTimestamp],
        merged_text: str,
    ) -> None:
        """統合結果をログファイルに保存。"""
        log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix = f"{run_id}_" if run_id else ""
        filename = f"{prefix}hybrid_merged_{timestamp}.json"
        filepath = log_dir / filename

        whisper_text = "".join(w.word for w in whisper_words)
        gemini_text = "".join(s.text for s in gemini_segments)

        # アライメントブロックの統計
        blocks_using_gemini = sum(1 for b in alignment_blocks if b.action == "use_gemini")
        blocks_using_whisper = sum(1 for b in alignment_blocks if b.action == "keep_whisper")
        avg_similarity = (
            sum(b.similarity for b in alignment_blocks) / len(alignment_blocks)
            if alignment_blocks else 0.0
        )

        log_data = {
            "type": "hybrid_merged",
            "timestamp": datetime.now().isoformat(),
            "run_id": run_id,
            "summary": {
                "whisper_word_count": len(whisper_words),
                "gemini_segment_count": len(gemini_segments),
                "merged_word_count": len(merged_words),
                "blocks_using_gemini": blocks_using_gemini,
                "blocks_using_whisper": blocks_using_whisper,
                "average_similarity": round(avg_similarity, 4),
                "similarity_threshold": self.similarity_threshold,
            },
            "texts": {
                "whisper": whisper_text,
                "gemini": gemini_text,
                "merged": merged_text,
            },
            "alignment_blocks": [
                {
                    "start_sec": b.start_sec,
                    "end_sec": b.end_sec,
                    "whisper_text": b.whisper_text,
                    "gemini_text": b.gemini_text,
                    "similarity": round(b.similarity, 4),
                    "action": b.action,
                }
                for b in alignment_blocks
            ],
            "merged_words": [
                {
                    "word": w.word,
                    "start": w.start,
                    "end": w.end,
                    "confidence": w.confidence,
                }
                for w in merged_words
            ],
        }

        filepath.write_text(json.dumps(log_data, ensure_ascii=False, indent=2))
        logger.info("Hybrid merged log saved: %s", filepath)

    def process_text_comparison(
        self,
        audio_path: Path,
        whisper_result: TranscriptionResult,
    ) -> dict:
        """
        WhisperとGeminiのテキストを比較し、詳細なアライメント情報を返す。

        デバッグや分析用途。

        Returns:
            比較結果の辞書
        """
        whisper_words = whisper_result.words or []

        try:
            gemini_segments = self.gemini_transcriber.transcribe(audio_path)
        except GeminiTranscriberError as exc:
            return {
                "error": str(exc),
                "whisper_text": whisper_result.text,
                "gemini_text": None,
            }

        alignment_blocks = self.aligner.align(whisper_words, gemini_segments)

        return {
            "whisper_text": whisper_result.text,
            "gemini_text": "".join(s.text for s in gemini_segments),
            "whisper_word_count": len(whisper_words),
            "gemini_segment_count": len(gemini_segments),
            "alignment_blocks": [
                {
                    "start_sec": b.start_sec,
                    "end_sec": b.end_sec,
                    "whisper_text": b.whisper_text,
                    "gemini_text": b.gemini_text,
                    "similarity": b.similarity,
                    "action": b.action,
                }
                for b in alignment_blocks
            ],
        }

    def _log_alignment_stats(self, blocks: Sequence[AlignmentBlock]) -> None:
        """アライメント統計をログ出力"""
        if not blocks:
            return

        total = len(blocks)
        using_gemini = sum(1 for b in blocks if b.action == "use_gemini")
        using_whisper = sum(1 for b in blocks if b.action == "keep_whisper")

        avg_similarity = sum(b.similarity for b in blocks) / total
        low_similarity = sum(1 for b in blocks if b.similarity < self.similarity_threshold)

        logger.info(
            "Alignment stats: blocks=%d, using_gemini=%d, using_whisper=%d, "
            "avg_similarity=%.3f, low_similarity_blocks=%d",
            total,
            using_gemini,
            using_whisper,
            avg_similarity,
            low_similarity,
        )

    @classmethod
    def create_for_use_case(
        cls,
        use_case: str,
        api_key: str | None = None,
        language: str = "ja",
    ) -> "HybridProcessor":
        """
        ユースケースに応じた最適な設定でインスタンスを生成。

        Args:
            use_case: "standard", "complex", "cost_efficient"
            api_key: Google API Key
            language: 文字起こし言語

        Returns:
            HybridProcessorインスタンス
        """
        if use_case == "standard":
            return cls(
                api_key=api_key,
                gemini_model=GeminiModel.FLASH_3_PREVIEW,
                thinking_level=ThinkingLevel.MEDIUM,
                similarity_threshold=0.8,
                language=language,
            )
        elif use_case == "complex":
            return cls(
                api_key=api_key,
                gemini_model=GeminiModel.FLASH_3_PREVIEW,
                thinking_level=ThinkingLevel.HIGH,
                similarity_threshold=0.7,  # より積極的にGeminiを採用
                language=language,
            )
        elif use_case == "cost_efficient":
            return cls(
                api_key=api_key,
                gemini_model=GeminiModel.FLASH_3_PREVIEW,
                thinking_level=ThinkingLevel.MINIMAL,
                similarity_threshold=0.9,  # Whisperを優先
                language=language,
            )
        else:
            raise ValueError(f"Unknown use_case: {use_case}")


__all__ = [
    "HybridProcessor",
    "HybridProcessorError",
]

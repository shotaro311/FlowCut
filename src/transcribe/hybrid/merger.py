"""アライメント結果に基づいてWordTimestampをマージ。"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import List, Sequence

from src.transcribe.base import WordTimestamp
from src.transcribe.hybrid.aligner import AlignmentBlock
from src.transcribe.hybrid.gemini_transcriber import GeminiSegment

logger = logging.getLogger(__name__)


@dataclass
class MergedWord:
    """マージされた単語"""

    word: str
    start: float
    end: float
    confidence: float
    source: str  # "whisper" | "gemini" | "merged"

    def to_word_timestamp(self) -> WordTimestamp:
        """WordTimestampに変換"""
        return WordTimestamp(
            word=self.word,
            start=self.start,
            end=self.end,
            confidence=self.confidence,
        )


class WordMerger:
    """アライメント結果に基づいてWordTimestampをマージ"""

    def __init__(
        self,
        min_word_duration: float = 0.05,
        default_word_duration: float = 0.2,
    ):
        """
        Args:
            min_word_duration: 単語の最小長さ（秒）
            default_word_duration: デフォルトの単語長さ（秒）
        """
        self.min_word_duration = min_word_duration
        self.default_word_duration = default_word_duration

    def merge(
        self,
        whisper_words: Sequence[WordTimestamp],
        gemini_segments: Sequence[GeminiSegment],
        alignment_blocks: Sequence[AlignmentBlock],
    ) -> List[WordTimestamp]:
        """
        アライメント結果に基づいて最終的なWordTimestamp列を生成。

        戦略:
        - action="keep_whisper": Whisperの単語とタイムスタンプをそのまま使用
        - action="use_gemini": Geminiのテキストを採用し、タイムスタンプは
          Whisperから推定または比例配分

        Args:
            whisper_words: Whisperの単語リスト
            gemini_segments: Geminiのセグメントリスト
            alignment_blocks: アライメント結果

        Returns:
            マージされたWordTimestampのリスト
        """
        if not alignment_blocks:
            # アライメントがない場合はWhisperをそのまま返す
            return list(whisper_words)

        merged_words: List[WordTimestamp] = []

        for block in alignment_blocks:
            if block.action == "keep_whisper":
                # Whisperの単語をそのまま使用
                words = self._extract_whisper_words(whisper_words, block)
                merged_words.extend(words)
            elif block.action == "use_gemini":
                # Geminiのテキストを単語に分割し、タイムスタンプを推定
                words = self._create_words_from_gemini(
                    block,
                    whisper_words,
                    gemini_segments,
                )
                merged_words.extend(words)
            else:
                # 未知のアクションはWhisperを使用
                logger.warning("Unknown action: %s, using whisper", block.action)
                words = self._extract_whisper_words(whisper_words, block)
                merged_words.extend(words)

        # 時間順にソート
        merged_words.sort(key=lambda w: (w.start, w.end))

        # 重複を除去（同じ時間範囲の単語）
        merged_words = self._remove_duplicates(merged_words)

        logger.info(
            "Merged words: whisper=%d, result=%d",
            len(whisper_words),
            len(merged_words),
        )

        return merged_words

    def _extract_whisper_words(
        self,
        whisper_words: Sequence[WordTimestamp],
        block: AlignmentBlock,
    ) -> List[WordTimestamp]:
        """ブロックに対応するWhisper単語を抽出"""
        if block.whisper_indices[0] < 0:
            return []

        start_idx, end_idx = block.whisper_indices
        return list(whisper_words[start_idx : end_idx + 1])

    def _create_words_from_gemini(
        self,
        block: AlignmentBlock,
        whisper_words: Sequence[WordTimestamp],
        gemini_segments: Sequence[GeminiSegment],
    ) -> List[WordTimestamp]:
        """
        Geminiのテキストから単語を作成し、タイムスタンプを推定。

        タイムスタンプ推定戦略:
        1. Whisper単語がある場合: その時間範囲内で比例配分
        2. Whisper単語がない場合: Geminiセグメントの時間範囲で比例配分
        """
        gemini_text = block.gemini_text
        if not gemini_text:
            return []

        # 日本語テキストを単語に分割
        words_text = self._tokenize_japanese(gemini_text)
        if not words_text:
            return []

        # 時間範囲を決定
        if block.whisper_indices[0] >= 0:
            start_idx, end_idx = block.whisper_indices
            start_sec = whisper_words[start_idx].start
            end_sec = whisper_words[end_idx].end
        elif block.gemini_segment_idx is not None:
            segment = gemini_segments[block.gemini_segment_idx]
            start_sec = segment.start_sec
            end_sec = segment.end_sec
        else:
            start_sec = block.start_sec
            end_sec = block.end_sec

        # 時間を比例配分
        duration = max(end_sec - start_sec, self.min_word_duration * len(words_text))
        time_per_char = duration / max(sum(len(w) for w in words_text), 1)

        result: List[WordTimestamp] = []
        current_time = start_sec

        for word_text in words_text:
            word_duration = max(
                len(word_text) * time_per_char,
                self.min_word_duration,
            )
            word_end = min(current_time + word_duration, end_sec)

            result.append(
                WordTimestamp(
                    word=word_text,
                    start=round(current_time, 3),
                    end=round(word_end, 3),
                    confidence=0.8,  # Gemini由来は少し低めに設定
                )
            )

            current_time = word_end

        return result

    def _tokenize_japanese(self, text: str) -> List[str]:
        """
        日本語テキストを単語に分割。

        シンプルな分割戦略:
        - 助詞・助動詞の後で分割
        - 句読点で分割
        - 長すぎる単語は文字数で分割
        """
        if not text:
            return []

        # 句読点を除去
        text = re.sub(r"[、。，．,.!?！？\-－―]", "", text)
        text = re.sub(r"[\[\]（）()「」『』]", "", text)
        text = text.strip()

        if not text:
            return []

        # 助詞パターン
        particle_pattern = r"(は|が|を|に|で|と|も|の|へ|から|まで|より|ね|よ|な|わ|か|ぞ|ぜ)"

        # 助詞の後で分割
        segments = re.split(f"({particle_pattern})", text)

        # 助詞を前の部分と結合
        words: List[str] = []
        current = ""
        for seg in segments:
            if not seg:
                continue
            if re.match(particle_pattern, seg):
                current += seg
            else:
                if current:
                    words.append(current)
                current = seg
        if current:
            words.append(current)

        # 長すぎる単語を分割（10文字以上）
        result: List[str] = []
        for word in words:
            if len(word) > 10:
                # 5文字ごとに分割
                for i in range(0, len(word), 5):
                    chunk = word[i : i + 5]
                    if chunk:
                        result.append(chunk)
            else:
                result.append(word)

        return [w for w in result if w]

    def _remove_duplicates(
        self,
        words: List[WordTimestamp],
    ) -> List[WordTimestamp]:
        """重複する単語を除去"""
        if not words:
            return []

        result: List[WordTimestamp] = []
        for word in words:
            # 同じ時間範囲の単語があるかチェック
            is_duplicate = False
            for existing in result:
                if (
                    abs(existing.start - word.start) < 0.01
                    and abs(existing.end - word.end) < 0.01
                ):
                    is_duplicate = True
                    break

            if not is_duplicate:
                result.append(word)

        return result


__all__ = [
    "MergedWord",
    "WordMerger",
]

"""Whisper出力とGemini出力のテキストアライメント。"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import List, Sequence, Tuple

from src.transcribe.base import WordTimestamp
from src.transcribe.hybrid.gemini_transcriber import GeminiSegment

logger = logging.getLogger(__name__)


@dataclass
class AlignmentResult:
    """アライメント結果"""

    whisper_start_idx: int
    whisper_end_idx: int
    gemini_segment_idx: int
    similarity: float
    whisper_text: str
    gemini_text: str
    action: str  # "keep_whisper" | "use_gemini" | "merge"


@dataclass
class AlignmentBlock:
    """アライメントブロック（時間範囲単位）"""

    start_sec: float
    end_sec: float
    whisper_indices: Tuple[int, int]  # (start_idx, end_idx) inclusive
    gemini_segment_idx: int | None
    similarity: float
    whisper_text: str
    gemini_text: str
    action: str


class TextAligner:
    """Whisper出力とGemini出力のアライメント"""

    def __init__(
        self,
        similarity_threshold: float = 0.8,
        time_tolerance_sec: float = 1.0,
    ):
        """
        Args:
            similarity_threshold: この閾値未満の類似度の場合、Geminiテキストを採用
            time_tolerance_sec: 時間範囲のマッチングに許容する誤差（秒）
        """
        self.similarity_threshold = similarity_threshold
        self.time_tolerance_sec = time_tolerance_sec

    def align(
        self,
        whisper_words: Sequence[WordTimestamp],
        gemini_segments: Sequence[GeminiSegment],
    ) -> List[AlignmentBlock]:
        """
        2つの文字起こし結果をアライメントする。

        アルゴリズム:
        1. Geminiセグメントの時間範囲に対応するWhisper単語を特定
        2. 各範囲でテキスト類似度を計算
        3. 低類似度区間を検出してアクションを決定

        Args:
            whisper_words: Whisperの単語リスト（タイムスタンプ付き）
            gemini_segments: Geminiのセグメントリスト

        Returns:
            AlignmentBlockのリスト
        """
        if not whisper_words:
            logger.warning("Whisper words is empty")
            return []

        if not gemini_segments:
            logger.warning("Gemini segments is empty, using whisper only")
            return self._create_whisper_only_blocks(whisper_words)

        blocks: List[AlignmentBlock] = []
        used_whisper_indices: set[int] = set()

        for seg_idx, segment in enumerate(gemini_segments):
            # このセグメントの時間範囲に対応するWhisper単語を特定
            whisper_indices = self._find_whisper_words_in_range(
                whisper_words,
                segment.start_sec,
                segment.end_sec,
            )

            if not whisper_indices:
                # 対応するWhisper単語がない場合（Whisperが欠落している可能性）
                logger.debug(
                    "No whisper words for gemini segment %d (%.2f-%.2f): %s",
                    seg_idx,
                    segment.start_sec,
                    segment.end_sec,
                    segment.text[:50],
                )
                blocks.append(
                    AlignmentBlock(
                        start_sec=segment.start_sec,
                        end_sec=segment.end_sec,
                        whisper_indices=(-1, -1),
                        gemini_segment_idx=seg_idx,
                        similarity=0.0,
                        whisper_text="",
                        gemini_text=segment.text,
                        action="use_gemini",
                    )
                )
                continue

            start_idx, end_idx = whisper_indices
            whisper_text = self._join_whisper_words(whisper_words, start_idx, end_idx)
            gemini_text = segment.text

            # 類似度を計算
            similarity = self._calculate_similarity(whisper_text, gemini_text)

            # アクションを決定
            if similarity >= self.similarity_threshold:
                action = "keep_whisper"
            else:
                action = "use_gemini"

            logger.debug(
                "Alignment block: whisper[%d:%d] (%.2f-%.2f) vs gemini[%d], "
                "similarity=%.3f, action=%s",
                start_idx,
                end_idx,
                whisper_words[start_idx].start,
                whisper_words[end_idx].end,
                seg_idx,
                similarity,
                action,
            )

            blocks.append(
                AlignmentBlock(
                    start_sec=whisper_words[start_idx].start,
                    end_sec=whisper_words[end_idx].end,
                    whisper_indices=(start_idx, end_idx),
                    gemini_segment_idx=seg_idx,
                    similarity=similarity,
                    whisper_text=whisper_text,
                    gemini_text=gemini_text,
                    action=action,
                )
            )

            for i in range(start_idx, end_idx + 1):
                used_whisper_indices.add(i)

        # 使用されなかったWhisper単語を処理
        blocks = self._add_unused_whisper_blocks(
            blocks, whisper_words, used_whisper_indices
        )

        # 時間順にソート
        blocks.sort(key=lambda b: b.start_sec)

        return blocks

    def _find_whisper_words_in_range(
        self,
        words: Sequence[WordTimestamp],
        start_sec: float,
        end_sec: float,
    ) -> Tuple[int, int] | None:
        """
        指定した時間範囲内のWhisper単語のインデックス範囲を返す。

        Returns:
            (start_idx, end_idx) のタプル、または見つからない場合はNone
        """
        start_with_tolerance = start_sec - self.time_tolerance_sec
        end_with_tolerance = end_sec + self.time_tolerance_sec

        matching_indices: List[int] = []
        for i, word in enumerate(words):
            # 単語の時間範囲が指定範囲と重なるかチェック
            word_start = word.start
            word_end = word.end
            if word_end >= start_with_tolerance and word_start <= end_with_tolerance:
                matching_indices.append(i)

        if not matching_indices:
            return None

        return (min(matching_indices), max(matching_indices))

    def _join_whisper_words(
        self,
        words: Sequence[WordTimestamp],
        start_idx: int,
        end_idx: int,
    ) -> str:
        """Whisper単語を結合してテキストにする"""
        return "".join(words[i].word for i in range(start_idx, end_idx + 1))

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """2つのテキストの類似度を計算（0.0-1.0）"""
        # 正規化: スペースや句読点を除去
        normalized1 = self._normalize_text(text1)
        normalized2 = self._normalize_text(text2)

        if not normalized1 and not normalized2:
            return 1.0
        if not normalized1 or not normalized2:
            return 0.0

        return SequenceMatcher(None, normalized1, normalized2).ratio()

    def _normalize_text(self, text: str) -> str:
        """テキストを正規化（比較用）"""
        # スペース、句読点、記号を除去
        text = re.sub(r"[\s\u3000]+", "", text)  # 空白
        text = re.sub(r"[、。，．,.!?！？\-－―]", "", text)  # 句読点
        text = re.sub(r"[\[\]（）()「」『』]", "", text)  # 括弧
        return text.lower()

    def _create_whisper_only_blocks(
        self,
        words: Sequence[WordTimestamp],
    ) -> List[AlignmentBlock]:
        """Geminiセグメントがない場合、Whisperのみのブロックを作成"""
        if not words:
            return []

        return [
            AlignmentBlock(
                start_sec=words[0].start,
                end_sec=words[-1].end,
                whisper_indices=(0, len(words) - 1),
                gemini_segment_idx=None,
                similarity=1.0,
                whisper_text=self._join_whisper_words(words, 0, len(words) - 1),
                gemini_text="",
                action="keep_whisper",
            )
        ]

    def _add_unused_whisper_blocks(
        self,
        blocks: List[AlignmentBlock],
        words: Sequence[WordTimestamp],
        used_indices: set[int],
    ) -> List[AlignmentBlock]:
        """使用されなかったWhisper単語をブロックとして追加"""
        all_indices = set(range(len(words)))
        unused = sorted(all_indices - used_indices)

        if not unused:
            return blocks

        # 連続するインデックスをグループ化
        groups: List[List[int]] = []
        current_group: List[int] = []

        for idx in unused:
            if not current_group or idx == current_group[-1] + 1:
                current_group.append(idx)
            else:
                if current_group:
                    groups.append(current_group)
                current_group = [idx]

        if current_group:
            groups.append(current_group)

        # 各グループをブロックとして追加
        for group in groups:
            start_idx = group[0]
            end_idx = group[-1]
            whisper_text = self._join_whisper_words(words, start_idx, end_idx)

            blocks.append(
                AlignmentBlock(
                    start_sec=words[start_idx].start,
                    end_sec=words[end_idx].end,
                    whisper_indices=(start_idx, end_idx),
                    gemini_segment_idx=None,
                    similarity=1.0,
                    whisper_text=whisper_text,
                    gemini_text="",
                    action="keep_whisper",
                )
            )

        return blocks


__all__ = [
    "AlignmentBlock",
    "AlignmentResult",
    "TextAligner",
]

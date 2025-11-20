"""`[WORD: ]` アンカーを単語タイムスタンプにアラインするユーティリティ。

Phase 2 要件に基づき、以下を実装する:
1. アンカー単語の完全一致を優先して検索
2. 一致しない場合は RapidFuzz で段階的なファジーマッチング (90→85→80%)
3. それでも見つからない場合はフォールバックの開始/終了時刻を推定し、警告ログへ記録
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence
import unicodedata

from rapidfuzz import fuzz

from src.llm.formatter import FormattedLine
from src.transcribe.base import WordTimestamp

logger = logging.getLogger(__name__)

# RapidFuzz は 0-100 範囲のスコアを返す
DEFAULT_THRESHOLDS: tuple[int, ...] = (90, 85, 80)


def _normalize_word(text: str | None) -> str:
    if text is None:
        return ""
    normalized = unicodedata.normalize("NFKC", text).strip()
    return normalized.lower()


def _initial_start(words: Sequence[WordTimestamp]) -> float:
    if not words:
        return 0.0
    first = words[0]
    if first.start is not None:
        return max(first.start, 0.0)
    if first.end is not None:
        return max(first.end - 0.5, 0.0)
    return 0.0


def _find_next_start(words: Sequence[WordTimestamp], start_index: int) -> tuple[int, float] | None:
    for idx in range(start_index, len(words)):
        candidate = words[idx].start
        if candidate is not None:
            return idx, candidate
    return None


def _persist_warnings(log_path: Path, warnings: Iterable[Dict[str, object]]) -> None:
    """警告情報をJSON配列として追記保存する。"""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    existing: List[Dict[str, object]] = []
    if log_path.exists():
        try:
            loaded = json.loads(log_path.read_text())
            if isinstance(loaded, list):
                existing = loaded
        except json.JSONDecodeError:
            logger.warning("alignment_warnings.json が壊れているため上書きします: %s", log_path)
    payload = [*existing, *warnings]
    log_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))


@dataclass(slots=True)
class AlignedLine:
    text: str
    anchor_word: str | None
    start: float
    end: float
    match_type: str
    match_score: float | None = None
    matched_word_index: int | None = None

    @property
    def duration(self) -> float:
        return max(self.end - self.start, 0.0)


def align_formatted_lines(
    formatted_lines: Sequence[FormattedLine],
    words: Sequence[WordTimestamp],
    *,
    fuzzy_thresholds: Sequence[int] = DEFAULT_THRESHOLDS,
    gap_seconds: float = 0.1,
    fallback_padding: float = 0.3,
    warning_log: Path | None = Path("logs/alignment_warnings.json"),
) -> List[AlignedLine]:
    """LLM整形済み行を word-level タイムスタンプへ割り当てる。

    Args:
        formatted_lines: LLM整形後の行（アンカー付き）
        words: 音声認識ランナーから得た WordTimestamp 配列
        fuzzy_thresholds: RapidFuzz で採用するスコア閾値の降順リスト
        gap_seconds: 行間の最小ギャップ（成功時）
        fallback_padding: フォールバック時に前行終了から足す秒数
        warning_log: フォールバック発生時に追記するログパス。None で無効化。
    """
    if not formatted_lines:
        return []
    if not words:
        raise ValueError("words は空にできません")

    normalized_words = [_normalize_word(w.word) for w in words]
    aligned: List[AlignedLine] = []
    warnings: List[Dict[str, object]] = []

    prev_end: float | None = None
    search_index = 0

    for line in formatted_lines:
        default_start = prev_end + gap_seconds if prev_end is not None else _initial_start(words)
        default_start = max(default_start, 0.0)

        match = _match_anchor(
            anchor_word=line.anchor_word,
            normalized_anchor=_normalize_word(line.anchor_word),
            normalized_words=normalized_words,
            search_index=search_index,
            thresholds=fuzzy_thresholds,
        )

        if match is not None:
            word_index, match_type, score = match
            target = words[word_index]
            end_time = target.end if target.end is not None else target.start or default_start
            start_time = default_start
            if end_time < start_time:
                start_time = max(end_time - 0.2, 0.0)
            aligned.append(
                AlignedLine(
                    text=line.text,
                    anchor_word=line.anchor_word,
                    start=start_time,
                    end=end_time,
                    match_type=match_type,
                    match_score=score,
                    matched_word_index=word_index,
                )
            )
            prev_end = end_time
            search_index = word_index + 1
        else:
            start_time = default_start if prev_end is None else prev_end + fallback_padding
            next_candidate = _find_next_start(words, search_index)
            if next_candidate:
                next_idx, next_start = next_candidate
                end_time = next_start if next_start > start_time else start_time + 1.0
                search_index = next_idx + 1
            else:
                end_time = start_time + 1.0
                search_index = len(words)
            aligned.append(
                AlignedLine(
                    text=line.text,
                    anchor_word=line.anchor_word,
                    start=start_time,
                    end=end_time,
                    match_type="fallback",
                    match_score=None,
                    matched_word_index=None,
                )
            )
            warnings.append(
                {
                    "line_number": line.line_number,
                    "anchor_word": line.anchor_word,
                    "reason": "anchor_not_found",
                    "start": start_time,
                    "end": end_time,
                    "search_from_index": search_index,
                }
            )
            prev_end = end_time

    if warning_log and warnings:
        _persist_warnings(warning_log, warnings)

    return aligned


def _match_anchor(
    *,
    anchor_word: str | None,
    normalized_anchor: str,
    normalized_words: Sequence[str],
    search_index: int,
    thresholds: Sequence[int],
) -> tuple[int, str, float] | None:
    """アンカー語に最も近い単語を検索する。"""
    if not anchor_word:
        return None

    # 完全一致（前方検索）
    for idx in range(search_index, len(normalized_words)):
        if normalized_words[idx] == normalized_anchor:
            return idx, "exact", 100.0

    # RapidFuzz で段階的にスコア評価
    for threshold in thresholds:
        best_idx = None
        best_score = -1.0
        for idx in range(search_index, len(normalized_words)):
            score = fuzz.ratio(normalized_anchor, normalized_words[idx])
            if score > best_score:
                best_score = score
                best_idx = idx
        if best_idx is not None and best_score >= threshold:
            return best_idx, "fuzzy", float(best_score)

    return None


__all__ = ["AlignedLine", "align_formatted_lines"]

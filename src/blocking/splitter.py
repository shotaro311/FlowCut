"""Block splitting logic for LLM整形前のテキスト."""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable, List, Sequence
import unicodedata
import logging

logger = logging.getLogger(__name__)


def _display_width(text: str) -> float:
    """Count characters with East Asian width (全角=1, 半角=0.5)."""
    width = 0.0
    for ch in text:
        width += 1.0 if unicodedata.east_asian_width(ch) in {"W", "F"} else 0.5
    return width


def _find_start(sentences: Sequence["Sentence"]) -> float | None:
    for sent in sentences:
        if sent.start is not None:
            return sent.start
    return None


def _find_end(sentences: Sequence["Sentence"]) -> float | None:
    for sent in reversed(sentences):
        if sent.end is not None:
            return sent.end
    return None


def _split_long_sentence(sentence: "Sentence", *, max_width: float) -> List["Sentence"]:
    """句読点がなく極端に長い文を安全弁で分割する。

    - 全角=1, 半角=0.5 で幅計算し、max_width を超えたら切る
    - 時刻情報は保持できないので start/end は None にする
    """
    pieces: List[str] = []
    buf: List[str] = []
    width = 0.0
    for ch in sentence.text:
        ch_width = 1.0 if unicodedata.east_asian_width(ch) in {"W", "F"} else 0.5
        if buf and width + ch_width > max_width:
            pieces.append("".join(buf))
            buf = []
            width = 0.0
        buf.append(ch)
        width += ch_width
    if buf:
        pieces.append("".join(buf))
    return [Sentence(text=p) for p in pieces if p.strip()]


@dataclass
class Sentence:
    text: str
    start: float | None = None
    end: float | None = None
    overlap: bool = False

    def clone(self, *, overlap: bool | None = None) -> "Sentence":
        return replace(self, overlap=self.overlap if overlap is None else overlap)


@dataclass
class Block:
    sentences: List[Sentence]

    @property
    def text(self) -> str:
        return "".join(sentence.text for sentence in self.sentences).strip()

    @property
    def start(self) -> float | None:
        return _find_start(self.sentences)

    @property
    def end(self) -> float | None:
        return _find_end(self.sentences)

    @property
    def duration(self) -> float | None:
        start = self.start
        end = self.end
        if start is None or end is None:
            return None
        return max(end - start, 0.0)


class BlockSplitter:
    """Split transcripts by文字数/時間 with optional overlap sentences."""

    def __init__(
        self,
        *,
        max_chars: int = 2000,
        max_duration: float | None = None,
        overlap_sentences: int = 2,
    ) -> None:
        if max_chars <= 0:
            raise ValueError("max_chars must be positive")
        if max_duration is not None and max_duration <= 0:
            raise ValueError("max_duration must be positive when specified")
        if overlap_sentences < 0:
            raise ValueError("overlap_sentences must be >= 0")
        self.max_chars = max_chars
        self.max_duration = max_duration
        self.overlap_sentences = overlap_sentences

    def split(self, sentences: Sequence[Sentence]) -> List[Block]:
        blocks: List[Block] = []
        if not sentences:
            return blocks

        current: List[Sentence] = []
        current_chars = 0.0
        overlap_buffer: List[Sentence] = []
        block_start: float | None = None
        block_end: float | None = None

        def seed_overlap() -> None:
            nonlocal current, current_chars, overlap_buffer, block_start, block_end
            if not overlap_buffer:
                return
            logger.debug("Seeding block with %d overlap sentences", len(overlap_buffer))
            for sent in overlap_buffer:
                clone = sent.clone()
                current.append(clone)
                current_chars += _display_width(clone.text)
            block_start = _find_start(current)
            block_end = _find_end(current)
            overlap_buffer = []

        def finalize_block() -> None:
            nonlocal current, current_chars, overlap_buffer, block_start, block_end
            if not current:
                return
            blocks.append(Block(sentences=list(current)))
            if self.overlap_sentences > 0:
                overlap_buffer = [
                    sentence.clone(overlap=True)
                    for sentence in current[-self.overlap_sentences :]
                ]
            else:
                overlap_buffer = []
            current = []
            current_chars = 0.0
            block_start = None
            block_end = None

        for original in sentences:
            if not original.text.strip():
                continue
            sentence = original.clone()
            # 安全弁：1文が max_chars を超える場合は句読点を待たず分割
            sentence_width = _display_width(sentence.text)
            if sentence_width > self.max_chars:
                long_parts = _split_long_sentence(sentence, max_width=self.max_chars)
                for part in long_parts:
                    # 再帰的に処理するため、前段のロジックを流用
                    sentences_to_process = [part]
                    for part_sentence in sentences_to_process:
                        width = _display_width(part_sentence.text)
                        while True:
                            candidate_start = block_start if block_start is not None else part_sentence.start
                            candidate_end = part_sentence.end if part_sentence.end is not None else block_end
                            duration = None
                            if candidate_start is not None and candidate_end is not None:
                                duration = max(candidate_end - candidate_start, 0.0)
                            exceeds_chars = current_chars + width > self.max_chars
                            exceeds_duration = (
                                self.max_duration is not None
                                and duration is not None
                                and duration > self.max_duration
                            )
                            if (exceeds_chars or exceeds_duration) and current:
                                finalize_block()
                                seed_overlap()
                                continue
                            break

                        if not current:
                            seed_overlap()
                            if block_start is None:
                                block_start = part_sentence.start

                        current.append(part_sentence)
                        current_chars += width
                        if part_sentence.end is not None:
                            block_end = part_sentence.end
                        elif block_end is None:
                            block_end = part_sentence.start
                continue
            width = _display_width(sentence.text)

            while True:
                candidate_start = block_start if block_start is not None else sentence.start
                candidate_end = sentence.end if sentence.end is not None else block_end
                duration = None
                if candidate_start is not None and candidate_end is not None:
                    duration = max(candidate_end - candidate_start, 0.0)
                exceeds_chars = current_chars + width > self.max_chars
                exceeds_duration = (
                    self.max_duration is not None and duration is not None and duration > self.max_duration
                )
                if (exceeds_chars or exceeds_duration) and current:
                    finalize_block()
                    seed_overlap()
                    continue
                break

            if not current:
                seed_overlap()
                if block_start is None:
                    block_start = sentence.start

            current.append(sentence)
            current_chars += width
            if sentence.end is not None:
                block_end = sentence.end
            elif block_end is None:
                block_end = sentence.start

        finalize_block()
        return blocks

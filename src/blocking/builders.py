"""Utilities for converting transcription outputs into BlockSplitter inputs."""
from __future__ import annotations

from typing import Iterable, List, Sequence

from ..transcribe.base import WordTimestamp

from .splitter import Sentence

_DEFAULT_DELIMITERS: tuple[str, ...] = (
    "。",
    "？",
    "！",
    "?",
    "!",
    "…",
    "．",
    ".",
)


def _is_sentence_break(token: str, delimiters: Sequence[str]) -> bool:
    stripped = token.strip()
    if not stripped:
        return False
    if stripped in delimiters:
        return True
    return any(stripped.endswith(d) for d in delimiters)


def sentences_from_words(
    words: Sequence[WordTimestamp],
    *,
    delimiters: Sequence[str] = _DEFAULT_DELIMITERS,
    joiner: str = " ",
    fallback_text: str | None = None,
) -> List[Sentence]:
    """Convert word timestamps into Sentence objects for block splitting."""

    sentences: List[Sentence] = []
    buffer: List[str] = []
    start_time: float | None = None
    end_time: float | None = None

    def flush() -> None:
        nonlocal buffer, start_time, end_time
        if not buffer:
            return
        text = joiner.join(buffer).strip()
        if text:
            sentences.append(Sentence(text=text, start=start_time, end=end_time))
        buffer = []
        start_time = None
        end_time = None

    for word in words:
        token = (word.word or "").strip()
        if not token:
            continue
        if start_time is None:
            start_time = word.start
        if word.end is not None:
            end_time = word.end
        buffer.append(token)
        if _is_sentence_break(token, delimiters):
            flush()

    flush()

    if not sentences and fallback_text:
        text = fallback_text.strip()
        if text:
            sentences.append(Sentence(text=text))

    return sentences


__all__ = ["sentences_from_words"]

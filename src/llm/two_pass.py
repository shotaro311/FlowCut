"""Two-pass LLM formatter (Pass1: replace/delete, Pass2: 17-char line splits)."""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import List, Sequence, Dict, Any, Iterable

from src.llm.formatter import FormatterError, get_provider, FormatterRequest
from src.llm.prompts import PromptPayload
from src.transcribe.base import WordTimestamp

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class EditOperation:
    type: str  # "replace" | "delete"
    start_idx: int
    end_idx: int
    text: str | None = None


@dataclass(slots=True)
class LineRange:
    start_idx: int
    end_idx: int
    text: str


@dataclass(slots=True)
class TwoPassResult:
    segments: List["SubtitleSegment"]

    @property
    def srt_text(self) -> str:
        from src.alignment.srt import segments_to_srt

        return segments_to_srt(self.segments)


def _extract_json(text: str) -> Any:
    """Extract JSON object/array from an LLM response, tolerating code fences."""
    # Remove code fences if present
    fenced = re.search(r"```(?:json)?\s*(\{.*\}|\[.*\])\s*```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1)
    # Find first JSON-like substring
    brace = text.find("{")
    bracket = text.find("[")
    start = min([p for p in [brace, bracket] if p != -1], default=-1)
    if start > 0:
        text = text[start:]
    return json.loads(text)


def _parse_operations(raw: Any) -> List[EditOperation]:
    ops = []
    items = raw.get("operations", []) if isinstance(raw, dict) else []
    for item in items:
        try:
            t = item["type"]
            s = int(item["start_idx"])
            e = int(item["end_idx"])
            if t not in {"replace", "delete"}:
                continue
            ops.append(EditOperation(type=t, start_idx=s, end_idx=e, text=item.get("text")))
        except Exception:
            continue
    return ops


def _parse_lines(raw: Any) -> List[LineRange]:
    lines = []
    items = raw.get("lines", []) if isinstance(raw, dict) else []
    for item in items:
        try:
            s = int(item["from"])
            e = int(item["to"])
            txt = str(item.get("text", "")).strip()
            lines.append(LineRange(start_idx=s, end_idx=e, text=txt))
        except Exception:
            continue
    return lines


def _apply_operations(words: Sequence[WordTimestamp], ops: Sequence[EditOperation]) -> List[WordTimestamp]:
    result = list(words)
    # Apply deletes and replaces from the end to keep indices stable
    for op in sorted(ops, key=lambda o: (o.start_idx, o.end_idx), reverse=True):
        if op.start_idx < 0 or op.end_idx >= len(result) or op.start_idx > op.end_idx:
            continue
        if op.type == "delete":
            del result[op.start_idx : op.end_idx + 1]
        elif op.type == "replace":
            new_word = WordTimestamp(
                word=op.text or "",
                start=result[op.start_idx].start,
                end=result[op.end_idx].end,
                confidence=result[op.start_idx].confidence,
            )
            result[op.start_idx : op.end_idx + 1] = [new_word]
    return result


def _build_indexed_words(words: Sequence[WordTimestamp]) -> str:
    return "\n".join(f"{i}: {w.word}" for i, w in enumerate(words))


def _safe_trim_json_response(text: str) -> Any:
    try:
        return _extract_json(text)
    except Exception as exc:  # pragma: no cover - defensive
        raise FormatterError(f"LLM JSONのパースに失敗しました: {exc}") from exc


class TwoPassFormatter:
    """Run two LLM passes and produce SRT without anchor alignment."""

    def __init__(self, llm_provider: str, temperature: float | None = None, timeout: float | None = None) -> None:
        self.provider_slug = llm_provider
        self.temperature = temperature
        self.timeout = timeout

    def _call_llm(self, prompt_text: str) -> str:
        provider = get_provider(self.provider_slug)
        payload = PromptPayload(system_prompt="", user_prompt=prompt_text)
        req = FormatterRequest(
            block_text="",
            provider=self.provider_slug,
            rewrite=False,
            metadata={},
            line_max_chars=17.0,
            max_retries=1,
            temperature=self.temperature,
            timeout=self.timeout,
        )
        return provider.format(prompt=payload, request=req)

    def run(
        self,
        text: str,
        words: Sequence[WordTimestamp],
        *,
        max_chars: float = 17.0,
    ) -> TwoPassResult | None:
        if not words:
            raise FormatterError("wordタイムスタンプが空です")

        logger.info("two-pass: pass1 start (words=%d, provider=%s)", len(words), self.provider_slug)
        # Pass1: replace/delete only
        pass1_prompt = self._build_pass1_prompt(text, words)
        logger.debug("Calling LLM for Pass 1. Prompt length: %d", len(pass1_prompt))
        raw1 = self._call_llm(pass1_prompt)
        parsed1 = _safe_trim_json_response(raw1)
        ops = _parse_operations(parsed1)
        updated_words = _apply_operations(words, ops) if ops else list(words)

        if not updated_words:
            # If all words deleted, return empty result (valid case)
            return TwoPassResult(segments=[])

        logger.info("two-pass: pass2 start (words=%d)", len(updated_words))
        # Pass2: line splits
        pass2_prompt = self._build_pass2_prompt(updated_words, max_chars=max_chars)
        logger.debug("Calling LLM for Pass 2. Prompt length: %d", len(pass2_prompt))
        raw2 = self._call_llm(pass2_prompt)
        parsed2 = _safe_trim_json_response(raw2)
        lines = _parse_lines(parsed2)
        if not lines:
            raise FormatterError("行分割結果が空です")

        segments = self._ranges_to_segments(updated_words, lines)
        logger.info("two-pass: completed (segments=%d)", len(segments))
        return TwoPassResult(segments=segments)

    def _ranges_to_segments(self, words: Sequence[WordTimestamp], lines: Sequence[LineRange]) -> List["SubtitleSegment"]:
        from src.alignment.srt import SubtitleSegment

        segments: List[SubtitleSegment] = []
        last_end_time = 0.0

        for idx, line in enumerate(lines, start=1):
            if line.start_idx < 0 or line.end_idx >= len(words) or line.start_idx > line.end_idx:
                logger.warning("行範囲が不正なためスキップ: %s", line)
                continue
            
            # Check for max chars overflow (fallback for LLM failure)
            current_text = line.text
            if len(current_text) > 17:
                # Local split required
                split_segments = self._split_line_locally(words, line.start_idx, line.end_idx, max_chars=17)
                for seg in split_segments:
                    # Clamp start time to prevent backward jumps
                    if seg.start < last_end_time:
                        logger.warning(
                            "Timestamp backward jump detected (clamped): %.2f -> %.2f", seg.start, last_end_time
                        )
                        seg.start = last_end_time
                    
                    # Ensure end >= start
                    if seg.end < seg.start:
                        seg.end = seg.start + 0.1
                        
                    segments.append(seg)
                    last_end_time = seg.end
            else:
                # Normal case: use exact word timestamps
                start = words[line.start_idx].start or 0.0
                end = words[line.end_idx].end or start
                
                # Clamp start time
                if start < last_end_time:
                    logger.warning(
                        "Timestamp backward jump detected (clamped): %.2f -> %.2f", start, last_end_time
                    )
                    start = last_end_time
                
                # Ensure end >= start
                if end < start:
                    end = start + 0.1
                    
                segments.append(SubtitleSegment(index=0, start=start, end=end, text=current_text))
                last_end_time = end

        # Re-assign indices
        for i, seg in enumerate(segments, start=1):
            seg.index = i
            
        return segments

    def _split_line_locally(
        self, words: Sequence[WordTimestamp], start_idx: int, end_idx: int, max_chars: int
    ) -> List["SubtitleSegment"]:
        from src.alignment.srt import SubtitleSegment
        
        results = []
        current_start_idx = start_idx
        
        while current_start_idx <= end_idx:
            # Build a chunk that fits in max_chars
            current_chars = 0
            chunk_end_idx = current_start_idx
            
            # Try to extend chunk as much as possible
            for i in range(current_start_idx, end_idx + 1):
                word_len = len(words[i].word)
                if current_chars + word_len > max_chars and current_chars > 0:
                    # Stop here, this word makes it too long
                    break
                current_chars += word_len
                chunk_end_idx = i
            
            # Create segment for this chunk
            chunk_text = "".join(w.word for w in words[current_start_idx : chunk_end_idx + 1])
            start_time = words[current_start_idx].start or 0.0
            end_time = words[chunk_end_idx].end or start_time
            if end_time < start_time:
                end_time = start_time + 0.1
                
            results.append(SubtitleSegment(index=0, start=start_time, end=end_time, text=chunk_text))
            
            # Move to next
            current_start_idx = chunk_end_idx + 1
            
        return results

    def _build_pass1_prompt(self, raw_text: str, words: Sequence[WordTimestamp]) -> str:
        indexed = _build_indexed_words(words)
        return (
            "あなたはプロの字幕エディターです。以下の単語列を順番を変えずに最小限の修正だけ加えてください。\n"
            "- 許可される操作: replace, delete（挿入は禁止。音声に無い単語を足さないこと）。\n"
            "- 単語の順序は変えないでください。\n"
            "- 出力は JSON で operations 配列のみを返してください。\n\n"
            f"入力テキスト:\n{raw_text}\n\n"
            f"単語リスト（index:word）:\n{indexed}\n\n"
            '出力フォーマット例:\n{"operations":[{"type":"replace","start_idx":10,"end_idx":11,"text":"カレーライス"},{"type":"delete","start_idx":25,"end_idx":25}]}\n'
            "追加の説明は不要です。"
        )

    def _build_pass2_prompt(self, words: Sequence[WordTimestamp], *, max_chars: float) -> str:
        indexed = _build_indexed_words(words)
        return (
            "# Role\n"
            "あなたは熟練の動画テロップ編集者です。\n"
            "提供されたテキストを、視聴者が最も読みやすいリズムで読めるように、以下の【思考ワークフロー】に従って処理し、行のインデックス範囲を JSON で返してください。\n\n"
            "# Constraints (制約)\n"
            f"- 1行の最大文字数：全角{int(max_chars)}文字\n"
            "- 出力形式：JSON の lines 配列のみ（例を参照）\n"
            "- 禁止事項：行末の句読点（、。）は必ず削除すること。文中の句読点は残してもよい。\n"
            "- 単語の順序を変えない。結合もしない。\n\n"
            "# Thinking Workflow (思考ワークフロー)\n"
            "## Step 1: チャンク分解 (Chunking)\n"
            "入力されたテキストを、文節や意味の最小単位（チャンク）に分解する。句読点（、。）や接続助詞（〜て、〜が、〜ので、〜から）を強い区切りとして扱う。\n\n"
            "## Step 2: 行の構築と決定 (Line Building)\n"
            "チャンクを前から順に追加し、以下の判定を行う。\n"
            f"1. 文字数オーバー: 現バッファ＋次チャンクが{int(max_chars)}文字を超えるなら改行。\n"
            f"2. 文脈区切り: {int(max_chars)}文字以内でも、読点・強い切れ目（〜ます、〜です、〜だ等）・接続助詞で終わるなら改行。\n\n"
            "## Step 3: クリーニング (Cleaning)\n"
            "行末の句読点（、。）を削除。文中の句読点は残してよい。\n\n"
            "# Input\n"
            f"単語リスト（index:word）:\n{indexed}\n\n"
            "# Output\n"
            '以下のJSONだけを返してください:\n{"lines":[{"from":0,"to":10,"text":"私は大学の12月ぐらい"},{"from":11,"to":25,"text":"政治家になろうと決めていて"}]}'
        )


__all__ = ["TwoPassFormatter", "TwoPassResult"]

"""Pass5: 長行改行（SRTテキスト後処理）。"""
from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import re
import time
from typing import Any, Dict, List

from src.llm.formatter import FormatterError, FormatterRequest, get_provider
from src.llm.prompts import PromptPayload
from src.llm.usage_metrics import record_pass_time

logger = logging.getLogger(__name__)

MIN_MAX_CHARS = 8


@dataclass(slots=True)
class SrtEntry:
    index: int
    start_time: str
    end_time: str
    text: str


def parse_srt(srt_text: str) -> List[SrtEntry]:
    entries: List[SrtEntry] = []
    blocks = re.split(r"\n\n+", srt_text.strip())
    for block in blocks:
        lines = block.strip().split("\n")
        if len(lines) < 3:
            continue
        try:
            index = int(lines[0].strip())
        except ValueError:
            continue
        time_match = re.match(
            r"(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2},\d{3})",
            lines[1].strip(),
        )
        if not time_match:
            continue
        entries.append(
            SrtEntry(
                index=index,
                start_time=time_match.group(1),
                end_time=time_match.group(2),
                text="\n".join(lines[2:]),
            )
        )
    return entries


def entries_to_srt(entries: List[SrtEntry]) -> str:
    blocks = []
    for entry in entries:
        blocks.append(f"{entry.index}\n{entry.start_time} --> {entry.end_time}\n{entry.text}")
    return "\n\n".join(blocks) + "\n"


def _extract_json(text: str) -> Any:
    fenced = re.search(r"```(?:json)?\s*(\{.*\}|\[.*\])\s*```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1)
    brace = text.find("{")
    bracket = text.find("[")
    start = min([p for p in [brace, bracket] if p != -1], default=-1)
    if start > 0:
        text = text[start:]
    return json.loads(text)


def _model_metadata_key(provider_slug: str) -> str:
    if provider_slug == "google":
        return "google_model"
    if provider_slug == "openai":
        return "openai_model"
    if provider_slug == "anthropic":
        return "anthropic_model"
    return "model"


class Pass5Processor:
    def __init__(
        self,
        *,
        provider: str,
        max_chars: int = 17,
        model_override: str | None = None,
        run_id: str | None = None,
        source_name: str | None = None,
        temperature: float | None = None,
        timeout: float | None = None,
    ) -> None:
        if max_chars < MIN_MAX_CHARS:
            raise ValueError(f"max_chars は {MIN_MAX_CHARS} 以上である必要があります（指定値: {max_chars}）")
        self.provider_slug = provider
        self.max_chars = max_chars
        self.model_override = model_override
        self.run_id = run_id
        self.source_name = source_name
        self.temperature = temperature
        self.timeout = timeout

    def _build_prompt(self, items: List[Dict[str, Any]]) -> str:
        payload = json.dumps({"lines": items}, ensure_ascii=False, indent=2)
        return (
            "あなたは字幕の長行に改行を入れるアシスタントです。\n"
            f"- 1行あたり全角{self.max_chars}文字を超える場合のみ、意味のまとまりが良い位置で改行を入れてください\n"
            "- 文字の置換・追加・削除は禁止（改行の挿入のみ）\n"
            "- 出力は必ずJSONのみ。改行はJSON文字列内で \\n として表現してください\n"
            "\n"
            "# Input(JSON)\n"
            f"{payload}\n"
            "\n"
            "# Output(JSON)\n"
            '{"lines":[{"index":0,"text":"..."}]}\n'
        )

    def _call_llm(self, prompt: str) -> str:
        provider = get_provider(self.provider_slug)
        metadata: Dict[str, Any] = {"pass_label": "pass5"}
        if self.run_id:
            metadata["run_id"] = self.run_id
        if self.source_name:
            metadata["source_name"] = self.source_name
        if self.model_override:
            metadata[_model_metadata_key(self.provider_slug)] = self.model_override
        if self.provider_slug == "anthropic":
            metadata["anthropic_max_tokens"] = 4096
        request = FormatterRequest(
            block_text="",
            provider=self.provider_slug,
            rewrite=False,
            metadata=metadata,
            line_max_chars=float(self.max_chars),
            max_retries=1,
            temperature=self.temperature,
            timeout=self.timeout,
        )
        return provider.format(prompt=PromptPayload(system_prompt="", user_prompt=prompt), request=request)

    def process(self, srt_text: str) -> str:
        if not srt_text.strip():
            return srt_text

        entries = parse_srt(srt_text)
        if not entries:
            return srt_text

        targets: List[Dict[str, Any]] = []
        target_entry_positions: List[int] = []
        for i, entry in enumerate(entries):
            lines = entry.text.split("\n")
            if not any(len(line) > self.max_chars for line in lines):
                continue
            target_entry_positions.append(i)
            targets.append({"index": len(targets), "text": entry.text.replace("\n", " ")})

        if not targets:
            return srt_text

        t_start = time.perf_counter()
        try:
            raw = self._call_llm(self._build_prompt(targets))
            parsed = _extract_json(raw)
        except (FormatterError, json.JSONDecodeError, TypeError, ValueError) as exc:
            logger.warning("pass5 failed; keeping original SRT: %s", exc)
            return srt_text
        t_end = time.perf_counter()
        record_pass_time(self.run_id, "pass5", t_end - t_start)

        if not isinstance(parsed, dict) or not isinstance(parsed.get("lines"), list):
            return srt_text

        out_lines = parsed["lines"]
        if len(out_lines) != len(targets):
            logger.warning("pass5 output size mismatch; keeping original SRT")
            return srt_text

        for pos, item in zip(target_entry_positions, out_lines):
            if not isinstance(item, dict):
                return srt_text
            text = item.get("text")
            if not isinstance(text, str) or not text.strip():
                return srt_text
            entries[pos].text = text.strip().replace("\\n", "\n")

        return entries_to_srt(entries)


__all__ = ["Pass5Processor", "MIN_MAX_CHARS", "parse_srt", "entries_to_srt"]

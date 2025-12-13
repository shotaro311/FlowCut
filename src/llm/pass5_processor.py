"""Pass5: Claude長行改行処理（Anthropic専用）。

SRT出力後の後処理として、指定文字数を超える行のみをClaudeで改行する。
タイムコードは変更しない。
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import List

from src.config import get_settings
from src.llm.formatter import FormatterError, FormatterRequest
from src.llm.prompts import PromptPayload
from src.llm.providers.anthropic_provider import AnthropicClaudeProvider
from src.llm.usage_metrics import record_pass_time
import time

logger = logging.getLogger(__name__)

# 最小文字数閾値（これ未満は設定不可）
MIN_MAX_CHARS = 8


@dataclass(slots=True)
class SrtEntry:
    """SRTエントリ（1つの字幕ブロック）"""
    index: int
    start_time: str
    end_time: str
    text: str


def parse_srt(srt_text: str) -> List[SrtEntry]:
    """SRTテキストをパースしてエントリリストを返す。"""
    entries: List[SrtEntry] = []
    blocks = re.split(r'\n\n+', srt_text.strip())
    
    for block in blocks:
        lines = block.strip().split('\n')
        if len(lines) < 3:
            continue
        
        try:
            index = int(lines[0].strip())
        except ValueError:
            continue
        
        # タイムコード行をパース
        time_match = re.match(
            r'(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2},\d{3})',
            lines[1].strip()
        )
        if not time_match:
            continue
        
        start_time = time_match.group(1)
        end_time = time_match.group(2)
        text = '\n'.join(lines[2:])
        
        entries.append(SrtEntry(
            index=index,
            start_time=start_time,
            end_time=end_time,
            text=text,
        ))
    
    return entries


def entries_to_srt(entries: List[SrtEntry]) -> str:
    """SrtEntryリストをSRTテキストに変換する。"""
    blocks = []
    for entry in entries:
        block = f"{entry.index}\n{entry.start_time} --> {entry.end_time}\n{entry.text}"
        blocks.append(block)
    return '\n\n'.join(blocks) + '\n'


class Pass5Processor:
    """SRT長行改行処理（Anthropic Claude専用）。
    
    指定文字数を超える行のみをClaudeに渡して改行させる。
    タイムコードは絶対に変更しない。
    """

    def __init__(
        self,
        max_chars: int = 17,
        *,
        run_id: str | None = None,
        source_name: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        timeout: float | None = None,
    ) -> None:
        if max_chars < MIN_MAX_CHARS:
            raise ValueError(f"max_chars は {MIN_MAX_CHARS} 以上である必要があります（指定値: {max_chars}）")
        
        self.max_chars = max_chars
        self.run_id = run_id
        self.source_name = source_name
        self.model = model
        self.temperature = temperature
        self.timeout = timeout
        self._provider = AnthropicClaudeProvider()

    def _build_prompt(self, long_lines: List[str]) -> str:
        """長行改行用のプロンプトを構築する。"""
        lines_text = "\n".join(f"- {line}" for line in long_lines)
        return (
            f"最終出力のsrtファイルの文章から{self.max_chars}文字を超える行のみ、"
            "適切な位置で改行してください。\n\n"
            "※注意事項\n"
            "・改行の位置は意味のまとまりを意識すること\n"
            f"・{self.max_chars}文字を超えない行は一切改行不要！！\n"
            "・タイムコードは絶対に変更しないこと\n\n"
            "【処理対象の行】\n"
            f"{lines_text}\n\n"
            "【出力形式】\n"
            "各行を改行後のテキストで返してください。元の行と同じ順序で、1行ずつ区切って返してください。\n"
            "改行が必要な場所には実際の改行を入れてください。\n"
            "説明文は不要です。処理結果のテキストのみを返してください。"
        )

    def _call_claude(self, prompt: str) -> str:
        """Claudeを呼び出して結果を取得する。"""
        settings = get_settings().llm
        
        metadata = {}
        if self.model:
            metadata["anthropic_model"] = self.model
        if self.run_id:
            metadata["run_id"] = self.run_id
        if self.source_name:
            metadata["source_name"] = self.source_name
        metadata["pass_label"] = "pass5"
        metadata["anthropic_max_tokens"] = 4096
        
        payload = PromptPayload(system_prompt="", user_prompt=prompt)
        request = FormatterRequest(
            block_text="",
            provider="anthropic",
            rewrite=False,
            metadata=metadata,
            line_max_chars=float(self.max_chars),
            max_retries=1,
            temperature=self.temperature,
            timeout=self.timeout,
        )
        
        return self._provider.format(prompt=payload, request=request)

    def process(self, srt_text: str) -> str:
        """SRTテキストを処理し、長行を改行して返す。
        
        Args:
            srt_text: 入力SRTテキスト
            
        Returns:
            処理後のSRTテキスト
        """
        if not srt_text.strip():
            return srt_text
        
        entries = parse_srt(srt_text)
        if not entries:
            logger.warning("Pass5: SRTエントリが見つかりません")
            return srt_text
        
        # 長行を持つエントリを特定
        long_entries_indices: List[int] = []
        long_lines: List[str] = []
        
        for i, entry in enumerate(entries):
            # 各行をチェック（既に改行されている場合は各行個別に）
            lines = entry.text.split('\n')
            has_long_line = any(len(line) > self.max_chars for line in lines)
            if has_long_line:
                long_entries_indices.append(i)
                long_lines.append(entry.text.replace('\n', ' '))  # 改行を一時的にスペースに
        
        if not long_lines:
            logger.info("Pass5: 長行がないためスキップ（max_chars=%d）", self.max_chars)
            return srt_text
        
        logger.info("Pass5: %d件の長行を処理します（max_chars=%d）", len(long_lines), self.max_chars)
        
        # Claudeを呼び出して改行処理
        t_start = time.perf_counter()
        try:
            prompt = self._build_prompt(long_lines)
            result = self._call_claude(prompt)
        except FormatterError as exc:
            logger.warning("Pass5: Claude呼び出しに失敗しました: %s", exc)
            return srt_text
        t_end = time.perf_counter()
        
        if self.run_id:
            record_pass_time(self.run_id, "pass5", t_end - t_start)
        
        # 結果をパース（各行が二重改行で区切られている想定）
        result_lines = [line.strip() for line in result.strip().split('\n\n') if line.strip()]
        
        # 結果が期待数と一致しない場合はフォールバック
        if len(result_lines) != len(long_entries_indices):
            # 単純な改行区切りも試す
            result_lines = [line.strip() for line in result.strip().split('\n') if line.strip()]
            
            # それでも一致しない場合は元のテキストを維持
            if len(result_lines) != len(long_entries_indices):
                logger.warning(
                    "Pass5: 結果の行数が一致しません（期待: %d, 実際: %d）。元のテキストを維持します。",
                    len(long_entries_indices),
                    len(result_lines),
                )
                return srt_text
        
        # 結果を適用
        for idx, result_line in zip(long_entries_indices, result_lines):
            entries[idx].text = result_line
        
        logger.info("Pass5: 処理完了（%.2f秒）", t_end - t_start)
        return entries_to_srt(entries)


__all__ = ["Pass5Processor", "MIN_MAX_CHARS"]

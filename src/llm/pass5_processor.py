"""Pass5: 長行改行処理（マルチプロバイダー対応）。

SRT出力後の後処理として、指定文字数を超える行のみをLLMで改行する。
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


def _get_provider_for_model(model: str) -> str:
    """モデル名からプロバイダーを判定する。"""
    model_lower = model.lower()
    if model_lower.startswith("gemini"):
        return "google"
    elif model_lower.startswith("gpt"):
        return "openai"
    elif model_lower.startswith("claude"):
        return "anthropic"
    return "anthropic"  # デフォルトはanthropic


def _get_llm_provider(provider: str):
    """プロバイダースラグからプロバイダーインスタンスを取得する。"""
    if provider == "google":
        from src.llm.providers.google_genai_provider import GoogleGenAIProvider
        return GoogleGenAIProvider()
    elif provider == "openai":
        from src.llm.providers.openai_provider import OpenAIProvider
        return OpenAIProvider()
    else:  # anthropic
        from src.llm.providers.anthropic_provider import AnthropicClaudeProvider
        return AnthropicClaudeProvider()


class Pass5Processor:
    """SRT長行改行処理（マルチプロバイダー対応）。
    
    指定文字数を超える行のみをLLMに渡して改行させる。
    タイムコードは絶対に変更しない。
    """

    def __init__(
        self,
        max_chars: int = 17,
        *,
        run_id: str | None = None,
        source_name: str | None = None,
        model: str | None = None,
        provider: str | None = None,
        temperature: float | None = None,
        timeout: float | None = None,
    ) -> None:
        if max_chars < MIN_MAX_CHARS:
            raise ValueError(f"max_chars は {MIN_MAX_CHARS} 以上である必要があります（指定値: {max_chars}）")
        
        self.max_chars = max_chars
        self.run_id = run_id
        self.source_name = source_name
        self.model = model or "claude-sonnet-4-5"
        self.temperature = temperature
        self.timeout = timeout
        
        # プロバイダー決定（明示指定 > モデル名から推測 > デフォルト）
        if provider:
            self.provider_slug = provider
        elif model:
            self.provider_slug = _get_provider_for_model(model)
        else:
            self.provider_slug = "anthropic"
        
        self._provider = _get_llm_provider(self.provider_slug)

    def _build_prompt(self, long_lines: List[str]) -> str:
        """長行改行用のプロンプトを構築する。"""
        lines_text = "\n".join(f"- {line}" for line in long_lines)
        return (
            f"指定された文字数（{self.max_chars}文字）を超える行に対し、"
            "意味のまとまりが良い位置で改行を挿入してください。\n\n"
            "# 処理ルール\n"
            f"1. {self.max_chars}文字を超える行のみ改行を入れること\n"
            "2. タイムコードやインデックスは絶対に変更しないこと\n"
            "3. 単なるテキストのみを出力すること（マークダウンや補足説明は不可）\n\n"
            "# 入力例\n"
            "- これはとても長い文章で指定された文字数を大幅に超えているため適切な位置で改行を入れる必要があります\n\n"
            "# 出力例（max_chars=15の場合）\n"
            "これはとても長い文章で\n"
            "指定された文字数を大幅に超えているため\n"
            "適切な位置で改行を入れる必要があります\n"
            "# 処理対象\n"
            f"{lines_text}\n\n"
            "# 出力\n"
            "各行を改行後のテキストで返してください。元の行と同じ順序で出力してください。"
        )

    def _call_llm(self, prompt: str) -> str:
        """LLMを呼び出して結果を取得する。"""
        metadata = {}
        
        # プロバイダー固有のモデル指定
        if self.provider_slug == "anthropic":
            metadata["anthropic_model"] = self.model
            metadata["anthropic_max_tokens"] = 4096
        elif self.provider_slug == "openai":
            metadata["openai_model"] = self.model
        elif self.provider_slug == "google":
            metadata["google_model"] = self.model
            
        if self.run_id:
            metadata["run_id"] = self.run_id
        if self.source_name:
            metadata["source_name"] = self.source_name
        metadata["pass_label"] = "pass5"
        
        system_prompt = (
            "あなたはSRT字幕の長行改行を行う専門のアシスタントです。"
            "指示に従い、正確にフォーマットしてください。余計な説明は不要です。"
        )
        payload = PromptPayload(system_prompt=system_prompt, user_prompt=prompt)
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
        
        logger.info(
            "Pass5: %d件の長行を処理します（max_chars=%d, provider=%s, model=%s）", 
            len(long_lines), self.max_chars, self.provider_slug, self.model
        )
        
        # LLMを呼び出して改行処理
        t_start = time.perf_counter()
        try:
            prompt = self._build_prompt(long_lines)
            result = self._call_llm(prompt)
        except FormatterError as exc:
            logger.warning("Pass5: LLM呼び出しに失敗しました: %s", exc)
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

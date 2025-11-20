"""LLM用のプロンプトテンプレート群。"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

SYSTEM_PROMPT = """あなたはプロの字幕編集者です。\n- 1行あたり17文字以内（全角=1, 半角=0.5）に収める\n- 文脈を保ちつつ冗長なフィラーを取り除く\n- 原文の単語は極力残し、固有名詞は維持する\n- 各行の末尾にその行の最後の単語を `[WORD: 単語]` 形式で必ず付与する\n- 行頭/末尾の空白や句読点を整える\n- JSONや番号付きリストではなく、純粋なテキスト行のみを返す\n"""

USER_PROMPT_TEMPLATE = """以下の文字起こしを日本語字幕用に整形してください。\n\n要件:\n1. 1行あたり17文字以内（全角=1, 半角=0.5）\n2. 自然な改行位置を選び、意味の塊ごとに分割する\n3. 原文の語順・単語を尊重しつつ、読みやすい句読点を追加する\n4. 各行の最後に `[WORD: 最後の単語]` を付与する\n5. {rewrite_instruction}\n6. `[WORD: ]` タグ以外に角括弧を使用しない\n7. 出力例: 設定を開いて[WORD: 開いて]\n\n--- 元テキスト ---\n{transcript}\n------------------\n"""

_REWRITE_ON = "語尾や言い回しを自然な敬体に整える（必要な置換のみ）"
_REWRITE_OFF = "語尾/言い回しは原文を維持し、最低限の整形に留める"


@dataclass(slots=True)
class PromptPayload:
    system_prompt: str
    user_prompt: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def as_messages(self) -> List[Dict[str, str]]:
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self.user_prompt},
        ]


def build_subtitle_prompt(
    transcript: str,
    *,
    rewrite: bool = False,
    metadata: Dict[str, Any] | None = None,
) -> PromptPayload:
    clean_text = transcript.strip()
    instruction = _REWRITE_ON if rewrite else _REWRITE_OFF
    user_prompt = USER_PROMPT_TEMPLATE.format(transcript=clean_text, rewrite_instruction=instruction)
    return PromptPayload(system_prompt=SYSTEM_PROMPT, user_prompt=user_prompt, metadata=metadata or {})


__all__ = ["PromptPayload", "build_subtitle_prompt", "SYSTEM_PROMPT"]

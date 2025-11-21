"""LLM用のプロンプトテンプレート群。"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

SYSTEM_PROMPT = """あなたは熟練の動画テロップ編集者です。以下を厳守してください。
- 1行の最大文字数: 全角17文字（全角=1, 半角=0.5）
- 行末の句読点（、。）は削除する。文中の句読点は残してよい
- 原文の単語と語順をできるだけ保持し、フィラーだけ最小限で削る
- 各行の末尾にその行の最後の単語を `[WORD: 単語]` 形式で必ず付与する（時間合わせに使用）
- JSONや番号リストは出力しない。整形後のテキスト行のみを返す"""

USER_PROMPT_TEMPLATE = """以下の文字起こしを、読みやすい字幕行に整形してください。

# 思考ワークフロー
Step1: チャンク分解
- 文節・意味の最小単位に分解し、読点（、。）や接続助詞（〜て、〜が、〜ので、〜から）を強い区切りとみなす

Step2: 行の構築（17文字以内）
- チャンクを前から順に追加し、17文字を超える場合は直前で行を確定
- 17文字以内でも「読点」「助詞終わり」「明確な意味切れ」が来たらそこで改行を優先

Step3: クリーニング
- 行末の句読点（、。）を取り除く
- 各行の末尾に `[WORD: 最後の単語]` を付与する

# 制約
- 1行17文字以内
- `[WORD: ]` タグ以外の角括弧は使わない
- 出力は改行区切りテキストのみ
- {rewrite_instruction}

--- 元テキスト ---
{transcript}
------------------
"""

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

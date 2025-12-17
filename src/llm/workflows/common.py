from __future__ import annotations

from typing import Sequence

from src.llm.workflows.definition import Pass4PromptFn


def build_indexed_words(words: Sequence) -> str:
    return "\n".join(f"{i}: {w.word}" for i, w in enumerate(words))


def build_pass4_prompt(line, words) -> str:
    indexed = "\n".join(
        f"{i}: {w.word}"
        for i, w in enumerate(words[line.start_idx : line.end_idx + 1], start=line.start_idx)
    )
    return (
        "# Role\n"
        "あなたはテロップ最終チェックの追加ステップ担当です。与えられた行に対してのみ、条件を満たす複数行に必要最小限で分割してください。\n\n"
        "# Constraints\n"
        "- 必ず1行あたり全角5〜17文字に収めること\n"
        "- 語順を変えない、語を追加/削除しない\n"
        "- 要約・翻訳・意訳をしない\n"
        "- 行末の句読点（、。）は削除。文中の句読点は残してよい\n"
        "- 改行の優先度: (1)「。?!」直後 → (2)「、」直後 → (3) 接続助詞・係助詞など自然な切れ目。\n\n"
        "# Input\n"
        f"対象の行テキスト:\n{line.text}\n\n"
        f"単語リスト（index:word）:\n{indexed}\n\n"
        "# Output\n"
        "以下のJSONだけを返してください（説明・コードフェンス禁止）。例:\n"
        '{\n'
        '  "lines": [\n'
        '    {"from": 100, "to": 105, "text": "...."},\n'
        '    {"from": 106, "to": 110, "text": "...."}\n'
        "  ]\n"
        "}\n"
    )


DEFAULT_PASS4_PROMPT: Pass4PromptFn = build_pass4_prompt


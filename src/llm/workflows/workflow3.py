from __future__ import annotations

import json
from typing import Sequence

from src.llm.workflows.common import DEFAULT_PASS4_PROMPT, build_indexed_words
from src.llm.workflows.definition import WorkflowDefinition


def build_pass1_prompt(raw_text: str, words: Sequence) -> str:
    indexed = build_indexed_words(words)
    return (
        "あなたはプロの字幕エディターです。以下の単語列を順番を変えずに、"
        "**明らかな誤変換のみ**最小限の修正を加えてください。\n\n"
        "# 許可される操作\n"
        "- **replace**: 明らかな誤変換を正しい語に置き換え\n"
        "- **delete**: 明らかなノイズ（フィラー、重複）を削除\n"
        "- **禁止**: 挿入（音声に無い単語を追加しない）、並び替え、要約、意訳\n\n"
        "# 入力\n"
        f"元のテキスト:\n{raw_text}\n\n"
        f"単語リスト（index:word）:\n{indexed}\n\n"
        "# 出力\n"
        "以下のJSON形式のみを返してください。説明文・コードフェンスは禁止。\n"
        "{\n"
        '  "operations": [\n'
        '    {"type": "replace", "start_idx": 10, "end_idx": 11, "text": "..."},\n'
        '    {"type": "delete", "start_idx": 25, "end_idx": 25}\n'
        "  ]\n"
        "}\n"
        '操作が不要な場合は空配列を返してください: {"operations": []}\n'
    )


def build_pass2_prompt(words: Sequence, max_chars: float) -> str:
    indexed = build_indexed_words(words)
    return (
        "# Role\n"
        "あなたは熟練の動画テロップ編集者です。\n"
        "以下の単語リストを、視聴者が読みやすいように行分割してください。\n\n"
        "# ルール\n"
        "1. 行頭に助詞・補助表現・小さい文字を置かない\n"
        "2. 行末の句読点（、。）は削除\n"
        "3. 文字数は5〜17文字の範囲を厳守\n"
        f"   - 最大: {int(max_chars)}文字（全角）\n\n"
        "# Input\n"
        f"単語リスト（index:word）:\n{indexed}\n\n"
        "# Output\n"
        "以下のJSONのみを返してください（説明・コードフェンス禁止）:\n"
        "{\n"
        '  "lines": [\n'
        '    {"from": 0, "to": 10, "text": "...."},\n'
        '    {"from": 11, "to": 25, "text": "...."}\n'
        "  ]\n"
        "}\n"
    )


def build_pass3_prompt(lines, words, issues, glossary_terms) -> str:
    if issues:
        issue_text = "\n".join([f"- {issue.description} → {issue.suggested_action}" for issue in issues])
    else:
        issue_text = "（検出された問題はありません）"
    current_lines = json.dumps(
        [{"from": l.start_idx, "to": l.end_idx, "text": l.text} for l in lines],
        ensure_ascii=False,
        indent=2,
    )
    glossary_text = "\n".join(glossary_terms or [])
    return (
        "# Role\n"
        "あなたは字幕の校正者です。以下の行を校正してください。\n\n"
        "# ルール\n"
        "- 誤字・脱字を修正\n"
        "- Glossary がある場合は表記を揃える\n"
        "- 必要なら行範囲（from/to）を調整してよい\n"
        "- JSONのみを返す\n\n"
        "# Glossary\n"
        f"{glossary_text}\n\n"
        "# 検出された問題（参考）\n"
        f"{issue_text}\n\n"
        "# Input\n"
        f"現在の行分割:\n{current_lines}\n\n"
        "# Output\n"
        "以下のJSONのみを返してください（説明・コードフェンス禁止）:\n"
        "{\n"
        '  "lines": [\n'
        '    {"from": 0, "to": 11, "text": "...."}\n'
        "  ]\n"
        "}\n"
    )


WORKFLOW = WorkflowDefinition(
    slug="workflow3",
    label="workflow3: カスタム",
    description="カスタム用（ここを書き換えると workflow3 にだけ反映）",
    wf_env_number=3,
    optimized_pass4=False,
    allow_pass3_range_change=True,
    pass1_fallback_enabled=False,
    pass1_prompt=build_pass1_prompt,
    pass2_prompt=build_pass2_prompt,
    pass3_prompt=build_pass3_prompt,
    pass4_prompt=DEFAULT_PASS4_PROMPT,
)

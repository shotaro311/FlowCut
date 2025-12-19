from __future__ import annotations

from typing import Sequence

from src.llm.workflows.common import DEFAULT_PASS4_PROMPT, build_indexed_words
from src.llm.workflows.definition import WorkflowDefinition


def build_pass1_prompt(raw_text: str, words: Sequence, glossary_terms: Sequence[str]) -> str:
    indexed = build_indexed_words(words)
    glossary_text = "\n".join(glossary_terms or [])
    return (
        "# Role\n"
        "あなたはプロの字幕校正者です。\n"
        "以下の単語列（index付き）を、語順を変えずに**最小限**で校正してください。\n\n"
        "# 目的（重要）\n"
        "- 固有名詞・政治関連用語・誤字脱字の表記を整える（確信がある場合のみ）\n"
        "- 行分割より前の工程なので、ここで表記を確定させる\n\n"
        "# 許可される操作（JSON operations）\n"
        "- replace: 誤変換/誤字を置換（必要なら複数単語を1つにまとめてもよい）\n"
        "- delete: 明らかなノイズ（フィラー・重複）の削除\n\n"
        "# 禁止（厳守）\n"
        "- insert（音声にない語の追加）\n"
        "- 並び替え、要約、意訳\n"
        "- 迷う場合の無理な修正（保守的に）\n\n"
        "# Glossary（最優先）\n"
        "Glossaryにある表記が正解です。該当する場合は必ずGlossary表記に揃えてください。\n"
        "複数単語に分かれていても、連続してGlossary語になる場合はその範囲を replace して1語にまとめてOKです。\n\n"
        f"{glossary_text}\n\n"
        "# Input\n"
        f"単語リスト（index:word）:\n{indexed}\n\n"
        "# Output\n"
        "以下のJSONのみを返してください（説明文・コードフェンス禁止）:\n"
        "{\n"
        '  "operations": [\n'
        '    {"type": "replace", "start_idx": 10, "end_idx": 11, "text": "菅義偉"},\n'
        '    {"type": "delete", "start_idx": 25, "end_idx": 25}\n'
        "  ]\n"
        "}\n"
        '操作が不要なら {"operations": []}\n'
    )


def build_pass2_prompt(words: Sequence, max_chars: float) -> str:
    indexed = build_indexed_words(words)
    return (
        "# Role\n"
        "あなたは熟練の動画テロップ編集者です。\n"
        "以下の単語リスト（index:word）を、字幕用の行に分割してください。\n\n"
        "# 必須条件（最重要）\n"
        "- 単語の順序は変えない。単語を落とさない。重複させない。\n"
        f"- 1行の文字数は必ず 5〜{int(max_chars)} 文字に収める。\n"
        "- from/to は単語indexの範囲（両端含む）。0 から最後の index まで漏れなく連続でカバーする。\n"
        "- 行末の句読点（、。）は削除する（文中は必要なら残してよい）。\n\n"
        "# 分割ルール（優先度高）\n"
        "1. 行頭に助詞・補助表現・小さい文字を置かない（前行に寄せる）\n"
        "2. 1〜4文字の極端に短い行を作らない（必ず統合して5文字以上）\n"
        "3. 引用表現「〜って言う/思う」は分割しない\n"
        "4. 意味のまとまり（文節）を優先して自然に\n\n"
        "# 自己チェック（出力前に必ず確認）\n"
        "- 全行が 5〜17 文字以内か\n"
        "- from/to が 0..last を連続で全カバーしているか（ギャップ/重複なし）\n"
        "- 行頭が助詞だけになっていないか\n\n"
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


WORKFLOW = WorkflowDefinition(
    slug="workflow3",
    label="workflow3: 校正→行分割（Pass3なし）",
    description="固有名詞/用語はPass1で確定し、Pass2で最終行分割（Pass3はスキップ）",
    wf_env_number=3,
    optimized_pass4=False,
    allow_pass3_range_change=False,
    pass3_enabled=False,
    pass1_fallback_enabled=True,
    two_call_enabled=False,
    pass1_prompt=build_pass1_prompt,
    pass2_prompt=build_pass2_prompt,
    pass3_prompt=None,
    pass4_prompt=DEFAULT_PASS4_PROMPT,
    pass2to4_prompt=None,
)

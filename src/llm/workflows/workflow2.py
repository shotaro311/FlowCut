from __future__ import annotations

import json
from typing import Sequence

from src.llm.workflows.common import DEFAULT_PASS4_PROMPT, build_indexed_words
from src.llm.workflows.definition import WorkflowDefinition


def build_pass1_prompt(raw_text: str, words: Sequence, glossary_terms: Sequence[str]) -> str:
    indexed = build_indexed_words(words)
    return (
        "あなたはプロの字幕エディターです。以下の単語列を順番を変えずに、"
        "**明らかな誤変換のみ**最小限の修正を加えてください。\n\n"
        "# 許可される操作\n"
        "- **replace**: 明らかな誤変換を正しい語に置き換え\n"
        "- **delete**: 明らかなノイズ（フィラー、重複）を削除\n"
        "- **禁止**: 挿入（音声に無い単語を追加しない）、並び替え、要約、意訳\n\n"
        "# 特に注意すべき誤変換（文脈が明確な場合のみ修正）\n"
        "1. **有名な固有名詞**（政治家・地名・企業名・政党名）\n"
        "   - 文脈から明らかに誤変換と判断できる場合のみ修正\n"
        "   - 例: 政治の話で「石川」→「石破」、「安政党」→「参政党」\n"
        "   - 判断が難しい場合は修正しない（人名の誤認識リスクを避ける）\n\n"
        "2. **一般的な誤変換パターン**\n"
        "   - 同音異義語の明らかな誤り（例: 「会う」→「合う」）\n"
        "   - カタカナ語の誤変換（例: 「コンピューター」→「コンピュータ」）\n\n"
        "# 禁止事項（厳守）\n"
        "- 意訳・要約・言い換えは絶対に禁止\n"
        "- 音声に無い単語を追加しない\n"
        "- 単語の順序を変えない\n"
        "- 判断が難しい場合は修正しない（保守的に）\n\n"
        "# 入力\n"
        f"元のテキスト:\n{raw_text}\n\n"
        f"単語リスト（index:word）:\n{indexed}\n\n"
        "# 出力\n"
        "以下のJSON形式のみを返してください。説明文・コードフェンスは禁止。\n"
        "{\n"
        '  "operations": [\n'
        '    {"type": "replace", "start_idx": 10, "end_idx": 11, "text": "石破"},\n'
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
        "# 最重要ルール（これを破ったら即アウト）\n"
        "1. **行頭に助詞・補助表現・小さい文字を置かない**\n"
        "   - 「が」「は」「を」「に」「で」「と」「も」「から」「まで」「よ」「ね」「な」「わ」や、\n"
        "     「んじゃない」「が必要」「と思って」、および「ぁぃぅぇぉゃゅょっァィゥェォャュョッ」「ん」「ン」などで行を *始めない*（前の行とひとまとまりにする）\n\n"
        "2. **行末の句読点（、。）は削除**\n"
        "   - 文中の句読点は残してよい\n\n"
        "3. **助詞だけ／短すぎる助詞行を作らない**\n"
        "   - 「が」「に」「を」「んで」など助詞を含む行が1〜4文字程度しかない場合は、必ず前後の行と統合して5文字以上にする\n\n"
        "4. **文字数は5〜17文字の範囲を厳守**\n"
        f"   - 1行の最大: {int(max_chars)}文字（全角）← 厳守\n"
        "   - 1行の最小: 5文字（全角）← 厳守\n"
        "   - 17文字を超える場合は、必ず前後の文脈を見て分割できる場所を探す\n"
        "   - 17文字超えは絶対に許容しない\n\n"
        "# 自然な分割ルール\n"
        "- 意味のまとまり（文節・フレーズ）を優先\n"
        "- 助詞での分割は許容されるが、「に」「が」「を」などが行頭に単独で残る分割はしない\n"
        "  （例: 「政策に」7文字 → OK、「に」1文字 → NG）\n"
        "- 引用表現「〜って言う/思う」は分割しない\n\n"
        "# 禁止事項\n"
        "- 単語の順序を変えない、結合しない\n"
        "- 要約・意訳・言い換えをしない\n"
        "- 1〜4文字の極端に短い行を作らない\n\n"
        "# 良い例\n"
        "✅ 「私は大学の時の」→「12月ぐらいかなと思って」\n"
        "   理由: 意味のまとまり、5文字以上、17文字以内\n\n"
        "# 悪い例\n"
        "❌ 「私は大学の時の」→「に」（1文字）\n"
        "   理由: 極端に短い\n"
        "❌ 「何するんだって」→「言うから」\n"
        "   理由: 引用表現の分割\n\n"
        "# Input\n"
        f"単語リスト（index:word）:\n{indexed}\n\n"
        "# Output\n"
        "以下のJSONのみを返してください（説明・コードフェンス禁止）:\n"
        "{\n"
        '  "lines": [\n'
        '    {"from": 0, "to": 10, "text": "私は大学の12月ぐらい"},\n'
        '    {"from": 11, "to": 25, "text": "政治家になろうと決めていて"}\n'
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
        "あなたはプロの字幕校正者です。\n"
        "以下の字幕行（SRT生成前の行情報）を校正してください。\n\n"
        "# 校正ルール（重要）\n"
        "1. 誤字・脱字を修正する\n"
        "2. 固有名詞（人名・地名・組織名）は、必ずGlossaryの正しい表記に揃える\n"
        "3. 政治関連用語（政党名・法案名・政策名など）は、一般に使われる公式表記に統一する\n"
        "   - ただし確信がない場合は変更しない（誤修正を避ける）\n"
        "4. from/to（範囲）は一切変更しない（タイムコードに影響するため）\n"
        "5. 行の順序と行数を変更しない\n"
        "6. 余計な説明は禁止。JSONのみを返す（コードフェンスも禁止）\n\n"
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
        '    {"from": 0, "to": 11, "text": "私は大学の時の12月ぐらいかな"},\n'
        '    {"from": 12, "to": 18, "text": "4年生の12月には"},\n'
        '    {"from": 19, "to": 28, "text": "政治家になろうという腹を決めていて"}\n'
        "  ]\n"
        "}\n"
    )


WORKFLOW = WorkflowDefinition(
    slug="workflow2",
    label="workflow2: 校正（最適化版）",
    description="誤字/固有名詞/政治用語の校正（Pass3で範囲変更なし）",
    wf_env_number=2,
    optimized_pass4=True,
    allow_pass3_range_change=False,
    pass1_fallback_enabled=True,
    pass1_prompt=build_pass1_prompt,
    pass2_prompt=build_pass2_prompt,
    pass3_prompt=build_pass3_prompt,
    pass4_prompt=DEFAULT_PASS4_PROMPT,
)

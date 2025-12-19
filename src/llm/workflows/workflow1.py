from __future__ import annotations

import json
from typing import Sequence

from src.llm.workflows.common import DEFAULT_PASS4_PROMPT, build_indexed_words
from src.llm.workflows.definition import WorkflowDefinition


def build_pass1_prompt(raw_text: str, words: Sequence, glossary_terms: Sequence[str]) -> str:
    indexed = build_indexed_words(words)
    glossary_text = "\n".join(glossary_terms or [])
    return (
        "# Role\n"
        "あなたはプロの字幕エディターです。\n"
        "以下の単語列（index付き）を、語順を変えずに**最小限**で校正してください。\n\n"
        "# 目的（この順で優先）\n"
        "1. 誤字・脱字を修正\n"
        "2. 固有名詞（人名・地名・組織名）を、Glossary と照らし合わせて正しい表記に揃える\n"
        "3. 政治関連用語（政党名・法案名・政策名など）は、一般に使われる公式表記に統一（例: Wikipedia等）\n"
        "   - ただし確信がない場合は変更しない（誤修正を避ける）\n\n"
        "# 許可される操作（JSON operations）\n"
        "- replace: 誤変換/誤字を正しい表記に置換（必要なら複数単語を1つにまとめて置換してよい）\n"
        "- delete: 明らかなノイズ（フィラー・重複）を削除\n\n"
        "# 禁止（厳守）\n"
        "- insert（音声に無い単語を追加しない）\n"
        "- 並び替え、要約、意訳\n\n"
        "# Glossary（最優先）\n"
        "Glossary にある表記が正解です。該当する場合は必ず Glossary 表記に揃えてください。\n"
        f"{glossary_text}\n\n"
        "# Input\n"
        f"元のテキスト:\n{raw_text}\n\n"
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
        "提供されたテキストを、視聴者が最も読みやすいリズムで読めるように、以下の【思考ワークフロー】に従って処理し、行のインデックス範囲を JSON で返してください。\n\n"
        "# Constraints (制約)\n"
        f"- 1行の最大文字数：全角{int(max_chars)}文字\n"
        "- 出力形式：JSON の lines 配列のみ（例を参照）\n"
        "- 単語の順序を変えない。結合もしない。\n\n"
        "# 自然な分割ルール（最優先）\n"
        "**以下のルールは文字数制約よりも優先度が高い：**\n"
        "1. **行頭に助詞・補助表現・小さい文字を置かない**: 「が」「は」「を」「に」「で」「と」「も」「から」「まで」「よ」「ね」「な」「わ」や、「んじゃない」「が必要」「と思って」、および「ぁぃぅぇぉゃゅょっァィゥェォャュョッ」「ん」「ン」などで行を *始めない*（前の行とひとまとまりにする）\n"
        "2. **接続表現・接続詞で分割しない**: 〜と思って、〜ものの、〜たら、〜ので、〜けど、〜けれど、んで、それで、そして 等で文を切らない\n"
        "3. **助詞だけ／短すぎる助詞行を作らない**: 「が」「に」「を」「んで」など助詞を含む行が1〜4文字程度しかない場合は必ず前後の行と統合し、5文字以上のまとまりにする\n"
        "4. **活用語尾の保持**: 〜てた、〜だった、〜たと 等の活用形は分割せずひとまとまりに\n"
        "5. **引用表現の保持**: 〜って言う、〜って思う、〜ってこと 等は分割しない\n\n"
        "# 分割の良い例・悪い例\n"
        "❌ 悪い例:\n"
        "  - 「考えた」→「ことがあったから」 （助詞「が」で分断）\n"
        "  - 「なった時」→「に」 （1文字のみの行）\n"
        "  - 「目指す」→「ものの」 （接続助詞で分断）\n"
        "  - 「何するんだ」→「って言うから」 （引用「って」で分断）\n"
        "  - 「思います」→「よ」 （終助詞「よ」が単独・1文字）\n"
        "  - 「怖かった」→「んで」 （接続詞「んで」が単独・2文字）\n\n"
        "✅ 良い例:\n"
        "  - 「考えたことが」→「あったから」 （助詞を含めてひとまとまり）\n"
        "  - 「なった時に」→「いやしないと」 （最小4文字以上）\n"
        "  - 「目指すものの」→「ダメな場合も」 （接続表現を保持）\n"
        "  - 「何するんだって言うから」→「家の手伝いを」 （引用表現を保持）\n"
        "  - 「思いますよ」→「いらっしゃって」 （終助詞を含めて4文字以上）\n"
        "  - 「怖かったんで」→「父がすっかり」 （接続詞を含めて4文字以上）\n\n"
        "# 禁止事項\n"
        "- 行末の句読点（、。）は必ず削除すること。文中の句読点は、読みやすさのために残してもよい。\n"
        "- 助詞・接続詞・活用語尾・終助詞での不自然な分割（上記ルール参照）\n"
        "- 1〜4文字のみの極端に短い行を残さない（5文字以上必須）\n"
        "- 引用表現「〜って言う」「〜って思う」の分割\n\n"
        "# Thinking Workflow (思考ワークフロー)\n"
        "## Step 1: チャンク分解 (Chunking)\n"
        "入力されたテキストを、文節や意味の最小単位（チャンク）に分解する。句読点（、。）や接続助詞（〜て、〜が、〜ので、〜から）を強い区切りとして扱う。\n\n"
        "## Step 2: 行の構築と決定 (Line Building)\n"
        "チャンクを前から順に追加し、以下の **優先順位** で判定を行う。\n"
        "**【最優先】自然な意味のまとまり（フレーズ境界）** を保持する\n"
        "1. 助詞・接続表現・終助詞での分割を避ける（上記ルール参照）\n"
        f"2. 文字数オーバー: 現バッファ＋次チャンクが{int(max_chars)}文字を超える場合のみ改行\n"
        f"3. 文脈区切り: {int(max_chars)}文字以内でも、読点・強い切れ目（〜ます、〜です、〜だ等）で終わるなら改行を検討\n"
        "4. 最小行長チェック: 分割後の行が5文字未満にならないか確認（4文字以下は禁止）\n\n"
        "## Step 3: クリーニング (Cleaning)\n"
        "行末の句読点（、。）を削除。文中の句読点は残してよい。\n\n"
        "# Input\n"
        f"単語リスト（index:word）:\n{indexed}\n\n"
        "# Output\n"
        "以下のJSONだけを返してください（説明・コードフェンス禁止）。例:\n"
        "{\n"
        '  "lines": [\n'
        '    {"from": 0, "to": 10, "text": "私は大学の12月ぐらい"},\n'
        '    {"from": 11, "to": 25, "text": "政治家になろうと決めていて"}\n'
        "  ]\n"
        "}\n"
    )


def build_pass3_prompt(lines, words, issues, glossary_terms) -> str:
    indexed = build_indexed_words(words)
    if issues:
        issue_text = "\n".join([f"- {issue.description} → {issue.suggested_action}" for issue in issues])
    else:
        issue_text = "問題は検出されませんでした。全行を確認し、以下のルールに従って最小限の修正を行ってください。"
    current_lines = json.dumps(
        [{"from": l.start_idx, "to": l.end_idx, "text": l.text} for l in lines],
        ensure_ascii=False,
        indent=2,
    )
    return (
        "# Role\n"
        "あなたはテロップの最終チェック担当の熟練編集者です。\n"
        "Pass 2で作成された字幕の行分割に問題がないか確認し、必要最小限の修正を行ってください。\n\n"
        "# 検出された問題\n"
        f"{issue_text}\n\n"
        "# 修正ルール\n"
        "1. **1-4文字の極端に短い行**: 前行または次行と統合し、結合後に全行がルールに沿っているか再確認\n"
        "2. **引用表現の分割「〜って言う/思う」**: 統合して1行に\n"
        "3. **修正後も制約を維持**: 17文字以内・5文字以上\n"
        "4. **最小限の修正**: 問題箇所のみ修正（全体を作り直さない）\n"
        "5. **要約・翻訳・意訳をしない**。語句の追加・削除もしない\n"
        "6. **語の途中で切れている箇所は必ず連結**（分断された語を統合）\n"
        "7. **改行の優先度**: (1)「。?!」直後 → (2)「、」直後 → (3) 接続助詞・係助詞など句が自然に切れる後ろ。名詞句/動詞句の途中は切らない。迷う場合は改行しない\n"
        "8. **元の語順と文脈を保つ**。句読点がない場合も上記7に沿って自然に整形する\n\n"
        "- 必ず1件以上の行を `lines` 配列で返してください（空配列やnullは禁止）\n\n"
        "# Input\n"
        f"単語リスト（index:word）:\n{indexed}\n\n"
        f"現在の行分割:\n{current_lines}\n\n"
        "# Output\n"
        "以下のJSONのみを返してください。説明文・コードフェンス・前後のテキストを含めることは禁止です。\n"
        "{\n"
        '  "lines": [\n'
        '    {"from": 0, "to": 11, "text": "私は大学の時の12月ぐらいかな"},\n'
        '    {"from": 12, "to": 18, "text": "4年生の12月には"},\n'
        '    {"from": 19, "to": 28, "text": "政治家になろうという腹を決めていて"},\n'
        '    {"from": 29, "to": 33, "text": "1月ぐらいから"},\n'
        '    {"from": 34, "to": 45, "text": "司法試験予備校に申し込んで"}\n'
        "  ]\n"
        "}\n"
    )


WORKFLOW = WorkflowDefinition(
    slug="workflow1",
    label="workflow1: 標準",
    description="従来の行分割 + 修正（Pass3で範囲変更あり）",
    wf_env_number=None,
    optimized_pass4=False,
    allow_pass3_range_change=True,
    pass1_fallback_enabled=False,
    pass1_prompt=build_pass1_prompt,
    pass2_prompt=build_pass2_prompt,
    pass3_prompt=build_pass3_prompt,
    pass4_prompt=DEFAULT_PASS4_PROMPT,
)

from __future__ import annotations

from typing import Sequence

from src.llm.workflows.definition import Pass4PromptFn


MAX_LINE_DURATION_SEC = 10.0

# fill_gapsで埋める最大ギャップ秒数
# この値を超えるギャップは埋めず、字幕間の自然なポーズを保持する
MAX_GAP_DURATION_SEC = MAX_LINE_DURATION_SEC


def build_indexed_words(words: Sequence) -> str:
    return "\n".join(f"{i}: {w.word}" for i, w in enumerate(words))


def build_pass4_prompt(line, words, max_chars: int) -> str:
    def _format_sec(value: object) -> str:
        try:
            return f"{float(value):.2f}"
        except (TypeError, ValueError):
            return "?"

    indexed = "\n".join(
        f"[{i}] {w.word} (time: {_format_sec(getattr(w, 'start', None))}-{_format_sec(getattr(w, 'end', None))}s)"
        for i, w in enumerate(words[line.start_idx : line.end_idx + 1], start=line.start_idx)
    )
    start = None
    end = None
    if 0 <= line.start_idx < len(words):
        start = getattr(words[line.start_idx], "start", None)
    if 0 <= line.end_idx < len(words):
        end = getattr(words[line.end_idx], "end", None)
    duration = None
    if start is not None and end is not None:
        try:
            duration = float(end) - float(start)
        except (TypeError, ValueError):
            duration = None
    line_time = f"{_format_sec(start)}-{_format_sec(end)} (duration={_format_sec(duration)}s)"
    return (
        "# Role\n"
        "あなたはテロップ最終チェックの追加ステップ担当です。与えられた行に対してのみ、条件を満たす複数行に必要最小限で分割してください。\n\n"
        "# Constraints\n"
        f"- 必ず1行あたり全角5〜{int(max_chars)}文字に収めること\n"
        f"- 1行の時間幅（end-start）が{MAX_LINE_DURATION_SEC:.1f}秒を超える場合は必ず分割\n"
        "- 語順を変えない、語を追加/削除しない\n"
        "- 要約・翻訳・意訳をしない\n"
        "- 行末の句読点（、。）は削除。文中の句読点は残してよい\n"
        "- 改行の優先度: (1)「。?!」直後 → (2)「、」直後 → (3) 接続助詞・係助詞など自然な切れ目。\n\n"
        "# Input\n"
        f"対象行のインデックス範囲: from={line.start_idx}, to={line.end_idx}\n"
        f"対象の行テキスト:\n{line.text}\n\n"
        f"対象行の時間情報（参考）:\n{line_time}\n\n"
        f"単語リスト（[インデックス] 単語 (time: 開始-終了)）:\n{indexed}\n\n"
        "# Output\n"
        "以下のJSONだけを返してください（説明・コードフェンス禁止）。\n"
        "**重要**: from/toは単語リストのインデックス番号（上記の[角括弧内の数字]）を指定してください。時間（秒）ではありません。\n\n"
        f"例（対象行が from={line.start_idx}, to={line.end_idx} の場合）:\n"
        '{\n'
        '  "lines": [\n'
        f'    {{"from": {line.start_idx}, "to": {line.start_idx + 5}, "text": "...."}},\n'
        f'    {{"from": {line.start_idx + 6}, "to": {line.end_idx}, "text": "...."}}\n'
        "  ]\n"
        "}\n"
    )


DEFAULT_PASS4_PROMPT: Pass4PromptFn = build_pass4_prompt

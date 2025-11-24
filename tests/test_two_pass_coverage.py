import json
from typing import List

from src.llm.two_pass import TwoPassFormatter
from src.transcribe.base import WordTimestamp


def _words_with_tail() -> List[WordTimestamp]:
    # シンプルな 6 語のダミーデータ（末尾側がフォールバック対象になる想定）
    return [
        WordTimestamp(word="ワン", start=0.0, end=0.5),
        WordTimestamp(word="ツー", start=0.5, end=1.0),
        WordTimestamp(word="スリー", start=1.0, end=1.5),
        WordTimestamp(word="フォー", start=1.5, end=2.0),
        WordTimestamp(word="ファイブ", start=2.0, end=2.5),
        WordTimestamp(word="シックス", start=2.5, end=3.0),
    ]


def test_trailing_words_are_covered_by_fallback(monkeypatch):
    """
    Pass2/Pass3 が先頭側の語（0〜3）だけに行を生成し、
    末尾の語（4〜5）に対応する行が無い場合でも、
    フォールバックにより全文が SRT に含まれることを確認する。
    """

    calls: list[str] = []

    def fake_call_llm(self, payload: str, model_override=None, pass_label=None):
        calls.append(pass_label or "")
        # Pass1: 変更なし
        if pass_label == "pass1":
            return json.dumps({"operations": []})
        # Pass2 / Pass3: 先頭 0〜3 だけをカバーする行を返す（末尾はあえて欠落させる）
        if pass_label in {"pass2", "pass3"}:
            return json.dumps(
                {
                    "lines": [
                        {"from": 0, "to": 1, "text": "ワンツー"},
                        {"from": 2, "to": 3, "text": "スリーフォー"},
                    ]
                },
                ensure_ascii=False,
            )
        # Pass4 は今回は呼ばれない想定だが、防御的に同じ行を返す
        return json.dumps(
            {
                "lines": [
                    {"from": 0, "to": 1, "text": "ワンツー"},
                    {"from": 2, "to": 3, "text": "スリーフォー"},
                ]
            },
            ensure_ascii=False,
        )

    # 検証の目的を単純化するため、issues は常に 0 件とする
    monkeypatch.setattr("src.llm.validators.detect_issues", lambda lines, words: [])
    monkeypatch.setattr(TwoPassFormatter, "_call_llm", fake_call_llm)

    words = _words_with_tail()
    formatter = TwoPassFormatter(llm_provider="google")

    result = formatter.run(
        text="".join(w.word for w in words),
        words=words,
    )

    assert result is not None
    # 末尾の字幕セグメントが、最後の語の終了時刻以降までカバーしていること
    assert result.segments[-1].end >= words[-1].end
    # 末尾の語のテキストが、少なくともどこかの字幕に含まれていること
    tail_word = words[-1].word
    assert any(tail_word in seg.text for seg in result.segments)


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


def test_invalid_pass2_ranges_are_repaired(monkeypatch):
    """
    Pass2 が `from/to` の範囲を不正に返しても（例: 1始まり）、
    index表現の正規化により処理が停止しないことを確認する。
    """

    words = [
        WordTimestamp(word="あいう", start=0.0, end=0.5),
        WordTimestamp(word="えお", start=0.5, end=1.0),
        WordTimestamp(word="かき", start=1.0, end=1.5),
        WordTimestamp(word="くけこ", start=1.5, end=2.0),
        WordTimestamp(word="さしす", start=2.0, end=2.5),
        WordTimestamp(word="せそ", start=2.5, end=3.0),
    ]

    def fake_call_llm(self, payload: str, model_override=None, pass_label=None):
        # Pass1: 変更なし
        if pass_label == "pass1":
            return json.dumps({"operations": []}, ensure_ascii=False)
        # Pass2: 1始まり想定の `from/to` を返してしまうケース（先頭が 0 でない）
        if pass_label == "pass2":
            return json.dumps(
                {
                    "lines": [
                        {"from": 1, "to": 2, "text": "あいうえお"},
                        {"from": 3, "to": 4, "text": "かきくけこ"},
                    ]
                },
                ensure_ascii=False,
            )
        # Pass3/4: 今回は影響させない（そのまま返す）
        return json.dumps(
            {
                "lines": [
                    {"from": 1, "to": 2, "text": "あいうえお"},
                    {"from": 3, "to": 4, "text": "かきくけこ"},
                ]
            },
            ensure_ascii=False,
        )

    monkeypatch.setattr("src.llm.validators.detect_issues", lambda lines, words: [])
    monkeypatch.setattr(TwoPassFormatter, "_call_llm", fake_call_llm)

    formatter = TwoPassFormatter(llm_provider="google")
    result = formatter.run(text="".join(w.word for w in words), words=words)

    assert result is not None
    assert result.segments[-1].end >= words[-1].end


def test_invalid_pass2_ranges_are_fixed_by_llm_repair(monkeypatch):
    """
    Pass2 がギャップ/オーバーラップを含む不正な範囲を返し、単純な正規化で直せない場合でも、
    「行テキストを変えずに範囲だけ再生成」することで処理が進むことを確認する。
    """

    words = [
        WordTimestamp(word="ワン", start=0.0, end=0.5),
        WordTimestamp(word="ツー", start=0.5, end=1.0),
        WordTimestamp(word="スリー", start=1.0, end=1.5),
        WordTimestamp(word="フォー", start=1.5, end=2.0),
        WordTimestamp(word="ファイブ", start=2.0, end=2.5),
        WordTimestamp(word="シックス", start=2.5, end=3.0),
    ]

    calls: list[str] = []

    def fake_call_llm(self, payload: str, model_override=None, pass_label=None):
        calls.append(pass_label or "")
        if pass_label == "pass1":
            return json.dumps({"operations": []}, ensure_ascii=False)
        if pass_label == "pass2":
            # ギャップのある不正な範囲（正規化では直せない想定）
            return json.dumps(
                {
                    "lines": [
                        {"from": 0, "to": 0, "text": "ワンツースリー"},
                        {"from": 2, "to": 5, "text": "フォーファイブシックス"},
                    ]
                },
                ensure_ascii=False,
            )
        if pass_label == "pass2_repair":
            # textはそのまま、範囲だけを修正した結果を返す
            return json.dumps(
                {
                    "lines": [
                        {"from": 0, "to": 2, "text": "ワンツースリー"},
                        {"from": 3, "to": 5, "text": "フォーファイブシックス"},
                    ]
                },
                ensure_ascii=False,
            )
        # Pass3/Pass4: 範囲は維持
        return json.dumps(
            {
                "lines": [
                    {"from": 0, "to": 2, "text": "ワンツースリー"},
                    {"from": 3, "to": 5, "text": "フォーファイブシックス"},
                ]
            },
            ensure_ascii=False,
        )

    monkeypatch.setattr("src.llm.validators.detect_issues", lambda lines, words: [])
    monkeypatch.setattr(TwoPassFormatter, "_call_llm", fake_call_llm)

    formatter = TwoPassFormatter(llm_provider="google")
    result = formatter.run(text="".join(w.word for w in words), words=words)

    assert result is not None
    assert "pass2_repair" in calls
    assert any("ワンツースリー" == seg.text for seg in result.segments)


def _words_with_internal_gap() -> List[WordTimestamp]:
    # 単語間に明確な時間ギャップがあるデータ
    return [
        WordTimestamp(word="前半", start=0.0, end=0.5),
        WordTimestamp(word="後半", start=2.0, end=2.5),
    ]


def test_segments_have_no_time_gaps(monkeypatch):
    """
    TwoPassFormatter が生成する SubtitleSegment 間に
    タイムコード上の空白が残らないことを確認する。
    """

    def fake_call_llm(self, payload: str, model_override=None, pass_label=None):
        # Pass1: 変更なし
        if pass_label == "pass1":
            return json.dumps({"operations": []})
        # Pass2 / Pass3 / Pass4: 2行の単純な行分割を返す
        return json.dumps(
            {
                "lines": [
                    {"from": 0, "to": 0, "text": "前半"},
                    {"from": 1, "to": 1, "text": "後半"},
                ]
            },
            ensure_ascii=False,
        )

    monkeypatch.setattr("src.llm.validators.detect_issues", lambda lines, words: [])
    monkeypatch.setattr(TwoPassFormatter, "_call_llm", fake_call_llm)

    words = _words_with_internal_gap()
    formatter = TwoPassFormatter(llm_provider="google")

    result = formatter.run(
        text="".join(w.word for w in words),
        words=words,
    )

    assert result is not None
    # 連続するすべてのセグメント間にギャップがないこと
    segments = result.segments
    assert len(segments) >= 2
    for prev, nxt in zip(segments, segments[1:]):
        assert prev.end == nxt.start

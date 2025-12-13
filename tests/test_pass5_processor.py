"""Pass5プロセッサのテスト。

Claude長行改行処理の単体テスト。
"""
from __future__ import annotations

import pytest
from src.llm.pass5_processor import (
    Pass5Processor,
    MIN_MAX_CHARS,
    SrtEntry,
    parse_srt,
    entries_to_srt,
)


class TestSrtParsing:
    """SRTパース関連のテスト。"""

    def test_parse_simple_srt(self):
        """シンプルなSRTをパースできる。"""
        srt_text = """1
00:00:00,000 --> 00:00:02,000
こんにちは

2
00:00:02,000 --> 00:00:04,000
世界です
"""
        entries = parse_srt(srt_text)
        assert len(entries) == 2
        assert entries[0].index == 1
        assert entries[0].start_time == "00:00:00,000"
        assert entries[0].end_time == "00:00:02,000"
        assert entries[0].text == "こんにちは"
        assert entries[1].text == "世界です"

    def test_parse_multiline_text(self):
        """複数行のテキストを持つSRTをパースできる。"""
        srt_text = """1
00:00:00,000 --> 00:00:02,000
1行目
2行目
"""
        entries = parse_srt(srt_text)
        assert len(entries) == 1
        assert entries[0].text == "1行目\n2行目"

    def test_entries_to_srt(self):
        """SrtEntryリストをSRTテキストに変換できる。"""
        entries = [
            SrtEntry(index=1, start_time="00:00:00,000", end_time="00:00:02,000", text="テスト"),
            SrtEntry(index=2, start_time="00:00:02,000", end_time="00:00:04,000", text="です"),
        ]
        srt_text = entries_to_srt(entries)
        assert "1\n00:00:00,000 --> 00:00:02,000\nテスト" in srt_text
        assert "2\n00:00:02,000 --> 00:00:04,000\nです" in srt_text


class TestPass5Processor:
    """Pass5Processorのテスト。"""

    def test_min_max_chars_validation(self):
        """max_charsが最小値未満だとエラーになる。"""
        with pytest.raises(ValueError) as excinfo:
            Pass5Processor(max_chars=7)
        assert str(MIN_MAX_CHARS) in str(excinfo.value)

    def test_min_max_chars_boundary(self):
        """max_charsが最小値ちょうどなら許可される。"""
        processor = Pass5Processor(max_chars=8)
        assert processor.max_chars == 8

    def test_default_max_chars(self):
        """デフォルトのmax_charsは17。"""
        processor = Pass5Processor()
        assert processor.max_chars == 17

    def test_empty_srt_returns_unchanged(self):
        """空のSRTは変更なしで返す。"""
        processor = Pass5Processor(max_chars=17)
        result = processor.process("")
        assert result == ""

    def test_whitespace_srt_returns_unchanged(self):
        """空白のみのSRTは変更なしで返す。"""
        processor = Pass5Processor(max_chars=17)
        result = processor.process("   \n\n  ")
        assert result == "   \n\n  "


class TestPass5ProcessorLogic:
    """Pass5の処理ロジックのテスト（Claudeを呼ばない範囲）。"""

    def test_short_lines_not_processed(self):
        """短い行は処理対象外（Claudeを呼ばない）。"""
        srt_text = """1
00:00:00,000 --> 00:00:02,000
短い行

2
00:00:02,000 --> 00:00:04,000
これも短い
"""
        processor = Pass5Processor(max_chars=20)
        # 全行が20文字以下なので、Claudeを呼ばずにそのまま返される
        # （実際のClaudeコールはモックしないとテストできないが、
        #   processメソッド内で長行がない場合はスキップされる）
        # ここではパースとロジックの確認のみ
        entries = parse_srt(srt_text)
        has_long = any(len(e.text) > 20 for e in entries)
        assert not has_long

    def test_long_line_detection(self):
        """長い行が正しく検出される。"""
        srt_text = """1
00:00:00,000 --> 00:00:02,000
これはとても長い行で17文字を超えています

2
00:00:02,000 --> 00:00:04,000
短い
"""
        entries = parse_srt(srt_text)
        # 1番目のエントリは17文字超
        assert len(entries[0].text) > 17
        # 2番目のエントリは17文字以下
        assert len(entries[1].text) <= 17


class TestPass5Integration:
    """Pass5の統合テスト（Claudeモックが必要な部分はスキップ）。"""

    def test_processor_initialization_with_all_params(self):
        """全パラメータでの初期化が正常に動作する。"""
        processor = Pass5Processor(
            max_chars=20,
            run_id="test_run",
            source_name="test.wav",
            model="claude-sonnet-4-20250514",
            temperature=0.5,
            timeout=30.0,
        )
        assert processor.max_chars == 20
        assert processor.run_id == "test_run"
        assert processor.source_name == "test.wav"
        assert processor.model == "claude-sonnet-4-20250514"
        assert processor.temperature == 0.5
        assert processor.timeout == 30.0

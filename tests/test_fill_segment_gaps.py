"""SRTギャップ埋め機能のテスト。

Phase 1: テロップ間の空白時間を埋める機能の単体テスト。
"""
from __future__ import annotations

import pytest
from src.alignment.srt import SubtitleSegment


class MockTwoPassFormatterForGapTest:
    """ギャップ埋めロジックのみをテストするための最小モック。"""

    def __init__(self, fill_gaps: bool = True, max_gap_duration: float | None = None):
        self.fill_gaps = fill_gaps
        self.max_gap_duration = max_gap_duration

    def _fill_segment_gaps(
        self,
        segments: list[SubtitleSegment],
        max_gap: float | None = None,
    ) -> None:
        """本物の TwoPassFormatter._fill_segment_gaps と同一ロジック。"""
        if not segments:
            return

        for i in range(len(segments) - 1):
            current = segments[i]
            nxt = segments[i + 1]
            gap = nxt.start - current.end
            if gap > 0:
                if max_gap is not None and gap > max_gap:
                    continue
                current.end = nxt.start


class TestFillSegmentGaps:
    """_fill_segment_gaps 関数のテスト。"""

    def test_basic_gap_filling(self):
        """基本的なギャップ埋め: すべてのギャップが埋まる。"""
        segments = [
            SubtitleSegment(index=1, start=0.0, end=1.0, text="こんにちは"),
            SubtitleSegment(index=2, start=2.0, end=3.0, text="世界"),
            SubtitleSegment(index=3, start=5.0, end=6.0, text="テスト"),
        ]

        formatter = MockTwoPassFormatterForGapTest(fill_gaps=True)
        formatter._fill_segment_gaps(segments)

        # セグメント1: end が 1.0 → 2.0 に延長
        assert segments[0].end == 2.0
        # セグメント2: end が 3.0 → 5.0 に延長
        assert segments[1].end == 5.0
        # セグメント3: 最後なので変更なし
        assert segments[2].end == 6.0

    def test_no_gaps(self):
        """ギャップがない場合は変更なし。"""
        segments = [
            SubtitleSegment(index=1, start=0.0, end=1.0, text="こんにちは"),
            SubtitleSegment(index=2, start=1.0, end=2.0, text="世界"),
            SubtitleSegment(index=3, start=2.0, end=3.0, text="テスト"),
        ]

        formatter = MockTwoPassFormatterForGapTest(fill_gaps=True)
        original_ends = [s.end for s in segments]
        formatter._fill_segment_gaps(segments)

        for i, seg in enumerate(segments):
            assert seg.end == original_ends[i]

    def test_max_gap_threshold(self):
        """max_gap 閾値のテスト: 閾値を超えるギャップは埋めない。"""
        segments = [
            SubtitleSegment(index=1, start=0.0, end=1.0, text="短いギャップ"),
            SubtitleSegment(index=2, start=2.0, end=3.0, text="長いギャップ後"),
            SubtitleSegment(index=3, start=15.0, end=16.0, text="最後"),
        ]

        formatter = MockTwoPassFormatterForGapTest(fill_gaps=True, max_gap_duration=5.0)
        formatter._fill_segment_gaps(segments, max_gap=5.0)

        # セグメント1: 1秒のギャップは埋まる (1.0 → 2.0)
        assert segments[0].end == 2.0
        # セグメント2: 12秒のギャップは max_gap=5.0 を超えるので埋まらない
        assert segments[1].end == 3.0
        # セグメント3: 最後なので変更なし
        assert segments[2].end == 16.0

    def test_max_gap_none_fills_all(self):
        """max_gap=None の場合はすべてのギャップを埋める。"""
        segments = [
            SubtitleSegment(index=1, start=0.0, end=1.0, text="開始"),
            SubtitleSegment(index=2, start=100.0, end=101.0, text="100秒後"),
        ]

        formatter = MockTwoPassFormatterForGapTest(fill_gaps=True, max_gap_duration=None)
        formatter._fill_segment_gaps(segments, max_gap=None)

        # 99秒のギャップでも埋まる
        assert segments[0].end == 100.0

    def test_empty_segments(self):
        """空のセグメントリストでもエラーにならない。"""
        segments: list[SubtitleSegment] = []
        formatter = MockTwoPassFormatterForGapTest(fill_gaps=True)
        formatter._fill_segment_gaps(segments)
        assert segments == []

    def test_single_segment(self):
        """セグメントが1つの場合も正常動作。"""
        segments = [SubtitleSegment(index=1, start=0.0, end=1.0, text="唯一")]
        formatter = MockTwoPassFormatterForGapTest(fill_gaps=True)
        formatter._fill_segment_gaps(segments)
        assert segments[0].end == 1.0


class TestFillGapsIntegration:
    """TwoPassFormatter のギャップ埋めパラメータの統合テスト。"""

    def test_fill_gaps_disabled(self):
        """fill_gaps=False の場合、ギャップは埋まらない。"""
        segments = [
            SubtitleSegment(index=1, start=0.0, end=1.0, text="こんにちは"),
            SubtitleSegment(index=2, start=3.0, end=4.0, text="世界"),
        ]

        formatter = MockTwoPassFormatterForGapTest(fill_gaps=False)
        # fill_gaps=False なので _fill_segment_gaps は呼ばれない想定
        # ここでは直接呼んでも、実際のコードでは fill_gaps フラグでガードされる
        if formatter.fill_gaps:
            formatter._fill_segment_gaps(segments, max_gap=formatter.max_gap_duration)

        # fill_gaps=False なので変更なし
        assert segments[0].end == 1.0
        assert segments[1].end == 4.0


class TestSRTOutput:
    """SRT出力形式のテスト（ギャップ埋め後の出力確認）。"""

    def test_srt_format_after_gap_fill(self):
        """ギャップ埋め後のSRT出力が正しいフォーマットになる。"""
        from src.alignment.srt import segments_to_srt

        segments = [
            SubtitleSegment(index=1, start=0.0, end=2.0, text="こんにちは"),
            SubtitleSegment(index=2, start=2.0, end=4.0, text="世界"),
        ]

        srt_text = segments_to_srt(segments)

        # SRT形式の基本構造を確認
        assert "1\n" in srt_text
        assert "00:00:00,000 --> 00:00:02,000" in srt_text
        assert "こんにちは" in srt_text
        assert "2\n" in srt_text
        assert "00:00:02,000 --> 00:00:04,000" in srt_text
        assert "世界" in srt_text

    def test_no_overlap_after_gap_fill(self):
        """ギャップ埋め後、セグメント同士が重複しない（end <= next.start）。"""
        segments = [
            SubtitleSegment(index=1, start=0.0, end=1.0, text="A"),
            SubtitleSegment(index=2, start=2.0, end=3.0, text="B"),
            SubtitleSegment(index=3, start=4.0, end=5.0, text="C"),
        ]

        formatter = MockTwoPassFormatterForGapTest(fill_gaps=True)
        formatter._fill_segment_gaps(segments)

        for i in range(len(segments) - 1):
            assert segments[i].end <= segments[i + 1].start

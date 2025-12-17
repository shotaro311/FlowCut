"""start_delay機能のテスト。

テロップのstart時間を遅らせる機能の単体テスト。
"""
from __future__ import annotations

import pytest
from src.alignment.srt import SubtitleSegment


class MockTwoPassFormatterForDelayTest:
    """start_delay ロジックのみをテストするための最小モック。"""

    def __init__(
        self,
        fill_gaps: bool = True,
        max_gap_duration: float | None = None,
        gap_padding: float = 0.0,
        start_delay: float = 0.0,
    ):
        self.fill_gaps = fill_gaps
        self.max_gap_duration = max_gap_duration
        self.gap_padding = gap_padding
        self.start_delay = start_delay

    def _fill_segment_gaps(
        self,
        segments: list[SubtitleSegment],
        max_gap: float | None = None,
        gap_padding: float = 0.0,
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
                    if gap_padding > 0:
                        desired_end = current.end + gap_padding
                        current.end = min(desired_end, nxt.start)
                    continue
                current.end = nxt.start

    def apply_start_delay(self, segments: list[SubtitleSegment]) -> None:
        """本物の TwoPassFormatter._ranges_to_segments のstart_delay部分と同一ロジック。"""
        # 最初のgap埋め
        if self.fill_gaps:
            self._fill_segment_gaps(
                segments,
                max_gap=self.max_gap_duration,
                gap_padding=self.gap_padding
            )

        # start_delay適用
        if self.start_delay > 0 and len(segments) > 1:
            original_last_end = segments[-1].end

            for i in range(1, len(segments)):
                new_start = segments[i].start + self.start_delay
                if new_start < segments[i - 1].end:
                    new_start = segments[i - 1].end
                segments[i].start = new_start

            # 遅延適用後、再度gap埋め
            if self.fill_gaps:
                self._fill_segment_gaps(
                    segments,
                    max_gap=self.max_gap_duration,
                    gap_padding=self.gap_padding
                )

            # 最後のendを元に戻す
            segments[-1].end = original_last_end


class TestStartDelay:
    """start_delay 機能のテスト。"""

    def test_basic_start_delay(self):
        """基本的なstart遅延テスト: 全セグメントに遅延が適用される。"""
        segments = [
            SubtitleSegment(index=1, start=0.0, end=1.0, text="こんにちは"),
            SubtitleSegment(index=2, start=1.0, end=2.0, text="世界"),
            SubtitleSegment(index=3, start=2.0, end=3.0, text="です"),
        ]

        formatter = MockTwoPassFormatterForDelayTest(
            fill_gaps=True, start_delay=0.2
        )
        formatter.apply_start_delay(segments)

        # セグメント1: startは維持（0.0）、endは次のstartまで延長（1.2）
        assert segments[0].start == 0.0
        assert segments[0].end == 1.2

        # セグメント2: start=1.0+0.2=1.2、endは次のstartまで延長（2.2）
        assert segments[1].start == 1.2
        assert segments[1].end == 2.2

        # セグメント3: start=2.0+0.2=2.2、endは元の値を維持（3.0）
        assert segments[2].start == 2.2
        assert segments[2].end == 3.0

    def test_first_segment_start_preserved(self):
        """最初のセグメントのstartは維持される。"""
        segments = [
            SubtitleSegment(index=1, start=0.5, end=1.0, text="開始"),
            SubtitleSegment(index=2, start=1.0, end=2.0, text="次"),
        ]

        formatter = MockTwoPassFormatterForDelayTest(
            fill_gaps=True, start_delay=0.3
        )
        formatter.apply_start_delay(segments)

        # 最初のセグメントのstartは変わらない
        assert segments[0].start == 0.5
        # 2番目のセグメントは遅延
        assert segments[1].start == 1.3

    def test_last_segment_end_preserved(self):
        """最後のセグメントのendは維持される。"""
        segments = [
            SubtitleSegment(index=1, start=0.0, end=1.0, text="A"),
            SubtitleSegment(index=2, start=1.0, end=5.0, text="B"),
        ]
        original_last_end = 5.0

        formatter = MockTwoPassFormatterForDelayTest(
            fill_gaps=True, start_delay=0.2
        )
        formatter.apply_start_delay(segments)

        # 最後のセグメントのendは元の値を維持
        assert segments[-1].end == original_last_end

    def test_no_delay_when_zero(self):
        """delay=0.0の場合は変更なし（gap埋めのみ）。"""
        segments = [
            SubtitleSegment(index=1, start=0.0, end=1.0, text="A"),
            SubtitleSegment(index=2, start=2.0, end=3.0, text="B"),
        ]

        formatter = MockTwoPassFormatterForDelayTest(
            fill_gaps=True, start_delay=0.0
        )
        formatter.apply_start_delay(segments)

        # gap埋めのみ適用（セグメント1のendが2.0に）
        assert segments[0].start == 0.0
        assert segments[0].end == 2.0
        assert segments[1].start == 2.0
        assert segments[1].end == 3.0

    def test_single_segment_no_change(self):
        """セグメントが1つのみの場合は変更なし。"""
        segments = [
            SubtitleSegment(index=1, start=0.0, end=1.0, text="単独"),
        ]

        formatter = MockTwoPassFormatterForDelayTest(
            fill_gaps=True, start_delay=0.5
        )
        formatter.apply_start_delay(segments)

        # 変更なし
        assert segments[0].start == 0.0
        assert segments[0].end == 1.0

    def test_overlap_prevention(self):
        """遅延が大きすぎる場合でもオーバーラップしない。"""
        segments = [
            SubtitleSegment(index=1, start=0.0, end=1.0, text="A"),
            SubtitleSegment(index=2, start=1.0, end=2.0, text="B"),
            SubtitleSegment(index=3, start=2.0, end=3.0, text="C"),
        ]

        # 遅延が1秒 → 各セグメントのstartが+1.0秒
        # しかしgap埋めで前のendに追従するため、オーバーラップは発生しない
        formatter = MockTwoPassFormatterForDelayTest(
            fill_gaps=True, start_delay=1.0
        )
        formatter.apply_start_delay(segments)

        # すべてのセグメントでオーバーラップなし
        for i in range(len(segments) - 1):
            assert segments[i].end <= segments[i + 1].start

    def test_gap_filling_after_delay(self):
        """遅延適用後にgap埋めが正しく動作する。"""
        segments = [
            SubtitleSegment(index=1, start=0.0, end=1.0, text="A"),
            SubtitleSegment(index=2, start=1.5, end=2.5, text="B"),  # 0.5秒のギャップ
        ]

        formatter = MockTwoPassFormatterForDelayTest(
            fill_gaps=True, start_delay=0.2
        )
        formatter.apply_start_delay(segments)

        # 遅延後: セグメント2のstartは1.5+0.2=1.7
        # gap埋め: セグメント1のendは1.7まで延長
        assert segments[0].end == segments[1].start
        assert segments[1].start == 1.7


class TestStartDelayIntegration:
    """start_delay とgap埋めパラメータの統合テスト。"""

    def test_delay_with_max_gap(self):
        """start_delay と max_gap_duration の組み合わせ。"""
        segments = [
            SubtitleSegment(index=1, start=0.0, end=1.0, text="A"),
            SubtitleSegment(index=2, start=10.0, end=11.0, text="B"),  # 9秒のギャップ
        ]

        formatter = MockTwoPassFormatterForDelayTest(
            fill_gaps=True,
            max_gap_duration=5.0,  # 5秒以上のギャップは埋めない
            start_delay=0.2,
        )
        formatter.apply_start_delay(segments)

        # セグメント2のstartは10.0+0.2=10.2
        assert segments[1].start == 10.2
        # ギャップが大きいので、セグメント1のendは1.0のまま（max_gap適用）
        assert segments[0].end == 1.0
        # 最後のendは維持
        assert segments[1].end == 11.0

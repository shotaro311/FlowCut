"""Workflow2 optimized TwoPassFormatter.

This module intentionally reuses `src.llm.two_pass` to avoid duplicated logic.
Only Pass4 (length fix) strategy is overridden to try a fast model first and
fallback to another model when needed.
"""
from __future__ import annotations

import logging
import time
from typing import List, Sequence

from src.llm.two_pass import (
    LineRange,
    TwoPassFormatter as BaseTwoPassFormatter,
    TwoPassResult,
    _call_llm_with_parse,
    _parse_lines,
)
from src.llm.usage_metrics import record_pass_time
from src.transcribe.base import WordTimestamp

logger = logging.getLogger(__name__)


class TwoPassFormatter(BaseTwoPassFormatter):
    def __init__(
        self,
        llm_provider: str,
        temperature: float | None = None,
        timeout: float | None = None,
        pass1_model: str | None = None,
        pass2_model: str | None = None,
        pass3_model: str | None = None,
        pass4_model: str | None = None,
        workflow: str = "workflow2",
        glossary_terms: Sequence[str] | None = None,
        *,
        run_id: str | None = None,
        source_name: str | None = None,
        fill_gaps: bool = True,
        max_gap_duration: float | None = None,
        gap_padding: float = 0.15,
        start_delay: float = 0.0,
        pass4_fast_model: str | None = None,
        pass4_fallback_model: str | None = None,
    ) -> None:
        super().__init__(
            llm_provider=llm_provider,
            temperature=temperature,
            timeout=timeout,
            pass1_model=pass1_model,
            pass2_model=pass2_model,
            pass3_model=pass3_model,
            pass4_model=pass4_model,
            workflow=workflow,
            glossary_terms=glossary_terms,
            run_id=run_id,
            source_name=source_name,
            fill_gaps=fill_gaps,
            max_gap_duration=max_gap_duration,
            gap_padding=gap_padding,
            start_delay=start_delay,
        )
        # Pass4の高速/フォールバックモデル（レイテンシ削減用）
        # - fast: まず軽量モデルで試行（未指定なら Pass2 と同じモデル）
        # - fallback: fast で失敗した場合に本命モデルへフォールバック
        self.pass4_fast_model = pass4_fast_model or self.pass2_model or self.pass4_model
        self.pass4_fallback_model = pass4_fallback_model or self.pass4_model

    def _validate_pass4_replacement(self, repl: Sequence[LineRange], orig: LineRange, n_words: int) -> bool:
        """Validate Pass4 replacement.

        - すべての行が 5〜17 文字に収まっていること
        - インデックスが orig の範囲内で、orig の範囲を完全にカバーすること
        """
        if not repl:
            return False
        for lr in repl:
            if len(lr.text) > 17 or len(lr.text) < 5:
                return False
            if lr.start_idx < 0 or lr.end_idx >= n_words or lr.start_idx > lr.end_idx:
                return False
            if lr.start_idx < orig.start_idx or lr.end_idx > orig.end_idx:
                return False
        covered = set()
        for lr in repl:
            covered.update(range(lr.start_idx, lr.end_idx + 1))
        return covered == set(range(orig.start_idx, orig.end_idx + 1))

    def _run_pass4_fix_once(
        self,
        line: LineRange,
        words: Sequence[WordTimestamp],
        *,
        model_override: str | None,
        pass_label: str,
    ) -> List[LineRange]:
        prompt = self._build_pass4_prompt(line, words)
        raw, parsed = _call_llm_with_parse(
            self._call_llm,
            pass_label=pass_label,
            prompt=prompt,
            model_override=model_override or self.pass4_model,
            retries=1,
            soft_fail=True,
            log_sink=self,
        )
        if parsed:
            repl = _parse_lines(parsed)
            if repl and self._validate_pass4_replacement(repl, line, len(words)):
                return repl
        logger.warning(
            "pass4 failed/invalid; keeping original line (%d-%d)",
            line.start_idx,
            line.end_idx,
        )
        return [line]

    def _run_pass4_fix(self, line: LineRange, words: Sequence[WordTimestamp]) -> List[LineRange]:
        """Try fast model first, then fallback model if needed."""
        t_fast_start = time.perf_counter()
        repl = self._run_pass4_fix_once(
            line,
            words,
            model_override=self.pass4_fast_model,
            pass_label="pass4_fast",
        )
        t_fast_end = time.perf_counter()
        record_pass_time(self.run_id, "pass4_fast", t_fast_end - t_fast_start)

        if (
            repl == [line]
            and self.pass4_fallback_model
            and self.pass4_fallback_model != self.pass4_fast_model
        ):
            t_fb_start = time.perf_counter()
            repl = self._run_pass4_fix_once(
                line,
                words,
                model_override=self.pass4_fallback_model,
                pass_label="pass4_fallback",
            )
            t_fb_end = time.perf_counter()
            record_pass_time(self.run_id, "pass4_fallback", t_fb_end - t_fb_start)

        return repl


__all__ = [
    "LineRange",
    "TwoPassFormatter",
    "TwoPassResult",
]


"""Two-pass LLM formatter (Pass1: replace/delete, Pass2: 17-char line splits)."""
from __future__ import annotations

import json
import logging
import os
import re
import uuid
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Sequence, Dict, Any, Iterable, Callable

from src.config import get_settings
from src.llm.formatter import FormatterError, get_provider, FormatterRequest
from src.llm.prompts import PromptPayload
from src.llm.workflows.registry import get_workflow
from src.transcribe.base import WordTimestamp
from src.llm.usage_metrics import record_pass_time
from src.utils.glossary import DEFAULT_GLOSSARY_TERMS, normalize_glossary_terms

logger = logging.getLogger(__name__)
RAW_LOG_DIR = Path("logs/llm_raw")


@dataclass(slots=True)
class EditOperation:
    type: str  # "replace" | "delete"
    start_idx: int
    end_idx: int
    text: str | None = None


@dataclass(slots=True)
class LineRange:
    start_idx: int
    end_idx: int
    text: str


@dataclass(slots=True)
class TwoPassResult:
    segments: List["SubtitleSegment"]

    @property
    def srt_text(self) -> str:
        from src.alignment.srt import segments_to_srt

        return segments_to_srt(self.segments)


def _extract_json(text: str) -> Any:
    """Extract JSON object/array from an LLM response, tolerating code fences."""
    # Remove code fences if present
    fenced = re.search(r"```(?:json)?\s*(\{.*\}|\[.*\])\s*```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1)
    # Find first JSON-like substring
    brace = text.find("{")
    bracket = text.find("[")
    start = min([p for p in [brace, bracket] if p != -1], default=-1)
    if start > 0:
        text = text[start:]
    return json.loads(text)


def _log_raw_response(pass_label: str, raw: str) -> None:
    """Persist raw LLM response for debugging.

    NOTE: この関数は旧実装との後方互換用。現在の実際のログは
    TwoPassFormatter 内の run_id / source_name 単位で集約される。
    """
    try:
        RAW_LOG_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        fname = RAW_LOG_DIR / f"{pass_label}_{ts}_{uuid.uuid4().hex[:8]}.txt"
        fname.write_text(raw, encoding="utf-8")
        logger.debug("Saved raw LLM response for %s to %s (len=%d)", pass_label, fname, len(raw))
    except Exception as exc:  # pragma: no cover - ログ失敗は致命的でない
        logger.warning("Failed to save raw LLM response for %s: %s", pass_label, exc)


def _parse_operations(raw: Any) -> List[EditOperation]:
    ops = []
    items = raw.get("operations", []) if isinstance(raw, dict) else []
    for item in items:
        try:
            t = item["type"]
            s = int(item["start_idx"])
            e = int(item["end_idx"])
            if t not in {"replace", "delete"}:
                continue
            ops.append(EditOperation(type=t, start_idx=s, end_idx=e, text=item.get("text")))
        except Exception:
            continue
    return ops


def _parse_lines(raw: Any) -> List[LineRange]:
    lines = []
    items = raw.get("lines", []) if isinstance(raw, dict) else []
    for item in items:
        try:
            s = int(item["from"])
            e = int(item["to"])
            txt = str(item.get("text", "")).strip()
            lines.append(LineRange(start_idx=s, end_idx=e, text=txt))
        except Exception:
            continue
    return lines


def _apply_operations(words: Sequence[WordTimestamp], ops: Sequence[EditOperation]) -> List[WordTimestamp]:
    result = list(words)
    # Apply deletes and replaces from the end to keep indices stable
    for op in sorted(ops, key=lambda o: (o.start_idx, o.end_idx), reverse=True):
        if op.start_idx < 0 or op.end_idx >= len(result) or op.start_idx > op.end_idx:
            continue
        if op.type == "delete":
            del result[op.start_idx : op.end_idx + 1]
        elif op.type == "replace":
            new_word = WordTimestamp(
                word=op.text or "",
                start=result[op.start_idx].start,
                end=result[op.end_idx].end,
                confidence=result[op.start_idx].confidence,
            )
            result[op.start_idx : op.end_idx + 1] = [new_word]
    return result


def _build_indexed_words(words: Sequence[WordTimestamp]) -> str:
    return "\n".join(f"{i}: {w.word}" for i, w in enumerate(words))


def _safe_trim_json_response(text: str) -> Any:
    try:
        return _extract_json(text)
    except Exception as exc:  # pragma: no cover - defensive
        raise FormatterError(f"LLM JSONのパースに失敗しました: {exc}") from exc


def _call_llm_with_parse(
    call_fn,
    *,
    pass_label: str,
    prompt: str,
    model_override: str | None,
    retries: int = 2,
    soft_fail: bool = False,
    log_sink: "TwoPassFormatter | None" = None,
) -> tuple[str | None, Any | None]:
    """
    Call LLM, log raw, parse JSON with limited retries.
    - soft_fail=True: return (None, None) instead of raising after retries.
    """
    last_exc: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            raw = call_fn(prompt, model_override=model_override, pass_label=pass_label)
        except FormatterError as exc:
            # API呼び出しレベルのエラー（HTTP 503など）
            last_exc = exc
            message = str(exc)
            # 単純な文字列マッチで「一時的エラー」っぽいものだけリトライ対象にする
            retryable = any(code in message for code in [" 500", " 502", " 503", " 504", " 429"]) or (
                "API request failed" in message or "temporarily unavailable" in message
            )
            if retryable and attempt < retries:
                logger.warning(
                    "%s API error (retryable, attempt %d/%d): %s",
                    pass_label,
                    attempt,
                    retries,
                    exc,
                )
                continue

            # リトライしない（or 最終試行）場合は、raw ログにエラー内容も残す。
            err_text = f"[API ERROR] pass={pass_label} model={model_override or '-'}: {exc}\n"
            if log_sink is not None:
                log_sink._append_raw_log(pass_label, err_text)
            else:
                _log_raw_response(pass_label, err_text)

            # リトライ不可または最終試行で失敗
            if soft_fail:
                logger.error("%s failed with non-retryable error: %s", pass_label, exc)
                return None, None
            raise FormatterError(f"{pass_label} failed (model={model_override or '-'}): {exc}") from exc

        # ログ集約先があればそこへ、なければ旧方式で単発ログ
        if log_sink is not None:
            log_sink._append_raw_log(pass_label, raw)
        else:
            _log_raw_response(pass_label, raw)
        try:
            parsed = _safe_trim_json_response(raw)
            return raw, parsed
        except FormatterError as exc:
            last_exc = exc
            logger.warning("%s parse failed (attempt %d/%d): %s", pass_label, attempt, retries, exc)
    if soft_fail:
        logger.error("%s parse failed after %d attempts; falling back", pass_label, retries)
        return None, None
    raise FormatterError(f"{pass_label} parse failed (model={model_override or '-'}): {last_exc}") from last_exc  # type: ignore[misc]


def _resolve_workflow_pass_model(wf_env_number: int | None, pass_num: int) -> str | None:
    if not wf_env_number:
        return None
    key = f"LLM_WF{wf_env_number}_PASS{pass_num}_MODEL"
    value = os.getenv(key)
    if not value:
        return None
    value = value.strip()
    return value or None


def _looks_like_context_limit_error(message: str) -> bool:
    msg = message.lower()
    return any(
        token in msg
        for token in [
            "context length",
            "maximum context",
            "context_length_exceeded",
            "too many tokens",
            "token limit",
            "prompt is too long",
            "request too large",
            "exceeds the maximum",
            "max tokens",
        ]
    )


def _looks_like_model_not_found_error(message: str) -> bool:
    msg = message.lower()
    if "model" not in msg:
        return False
    return any(token in msg for token in ["not found", "does not exist", "unknown model", "model_not_found"])


def _has_full_word_coverage(lines: Sequence["LineRange"], words: Sequence[WordTimestamp]) -> bool:
    """
    与えられた行集合が、words 全体（0〜len(words)-1）を
    1つ以上の行で完全にカバーしているかを確認するユーティリティ。

    - どこかのインデックスが1度も含まれていない場合は False
    - 不正なインデックス範囲（負数や末尾超過など）がある場合も False
    """
    if not lines:
        return False
    n = len(words)
    if n == 0:
        return False
    covered = [False] * n
    for line in lines:
        if line.start_idx < 0 or line.end_idx >= n or line.start_idx > line.end_idx:
            return False
        for idx in range(line.start_idx, line.end_idx + 1):
            covered[idx] = True
    return all(covered)


def _has_same_line_ranges(before: Sequence["LineRange"], after: Sequence["LineRange"]) -> bool:
    """Return True if line ranges (from/to) are identical and in the same order."""
    if len(before) != len(after):
        return False
    for b, a in zip(before, after):
        if b.start_idx != a.start_idx or b.end_idx != a.end_idx:
            return False
    return True


class TwoPassFormatter:
    """Run two LLM passes and produce SRT without anchor alignment."""

    def __init__(
        self,
        llm_provider: str,
        temperature: float | None = None,
        timeout: float | None = None,
        pass1_model: str | None = None,
        pass2_model: str | None = None,
        pass3_model: str | None = None,
        pass4_model: str | None = None,
        workflow: str = "workflow1",
        glossary_terms: Sequence[str] | None = None,
        *,
        run_id: str | None = None,
        source_name: str | None = None,
        raw_log_dir: Path | None = None,
        fill_gaps: bool = True,
        max_gap_duration: float | None = None,
        gap_padding: float = 0.15,
        start_delay: float = 0.0,
    ) -> None:
        settings = get_settings().llm
        self.provider_slug = llm_provider
        self.temperature = temperature
        self.timeout = timeout
        self.workflow_def = get_workflow(workflow)
        self.workflow = self.workflow_def.slug

        wf_num = self.workflow_def.wf_env_number
        wf_p1 = _resolve_workflow_pass_model(wf_num, 1)
        wf_p2 = _resolve_workflow_pass_model(wf_num, 2)
        wf_p3 = _resolve_workflow_pass_model(wf_num, 3)
        wf_p4 = _resolve_workflow_pass_model(wf_num, 4)

        self.pass1_model = pass1_model or wf_p1 or settings.pass1_model
        self.pass2_model = pass2_model or wf_p2 or settings.pass2_model
        self.pass3_model = pass3_model or wf_p3 or settings.pass3_model
        self.pass4_model = pass4_model or wf_p4 or settings.pass4_model
        self._pass1_fallback_model = settings.pass1_model if self.workflow_def.pass1_fallback_enabled else None
        # 辞書（Glossary）: None の場合はデフォルト辞書を利用する
        self.glossary_terms = (
            list(DEFAULT_GLOSSARY_TERMS)
            if glossary_terms is None
            else normalize_glossary_terms(glossary_terms)
        )
        # ログ用コンテキスト（処理単位で1ファイルにまとめる）
        self.run_id = run_id
        self.source_name = source_name
        self.raw_log_dir = raw_log_dir or RAW_LOG_DIR
        self._log_buffer: Dict[str, str] = {}
        self._log_date_str = datetime.now(timezone.utc).strftime("%Y%m%d")
        self._log_written = False
        # SRTギャップ埋め設定
        # fill_gaps: True の場合、連続するセグメント間の空白時間を埋める
        # max_gap_duration: 埋める最大ギャップ秒数。None の場合は上限なし
        self.fill_gaps = fill_gaps
        self.max_gap_duration = max_gap_duration
        self.gap_padding = gap_padding
        # テロップ開始時間遅延（秒）。2番目以降のセグメントのstartをこの秒数だけ遅らせる。
        # 最初のセグメントのstartと最後のセグメントのendは維持される。
        self.start_delay = start_delay

    # --- logging helpers -------------------------------------------------

    def _append_raw_log(self, pass_label: str, raw: str) -> None:
        # パスごとに区切りを付けて1ファイルにまとめるためのバッファ
        prefix = f"\n\n===== {pass_label} =====\n"
        existing = self._log_buffer.get(pass_label, "")
        self._log_buffer[pass_label] = existing + prefix + raw

    def _flush_logs(self) -> None:
        """現在の run 内で蓄積した LLM 生ログを1ファイルにまとめて書き出す。"""
        if self._log_written or not self._log_buffer:
            return
        try:
            self.raw_log_dir.mkdir(parents=True, exist_ok=True)
            base_name = (self.source_name or self.run_id or "llm_run").replace("/", "_")
            # yyyymmdd_連番 の連番部分は run_id があればそれを使い、無ければ uuid 短縮
            suffix = self.run_id or uuid.uuid4().hex[:8]
            fname = self.raw_log_dir / f"{base_name}_{self._log_date_str}_{suffix}.txt"
            # パス順で安定した並びにして書き出す
            ordered = []
            for label in sorted(self._log_buffer.keys()):
                ordered.append(self._log_buffer[label])
            fname.write_text("".join(ordered), encoding="utf-8")
            logger.debug("Saved aggregated raw LLM response to %s", fname)
            self._log_written = True
        except Exception as exc:  # pragma: no cover - ログ失敗は致命的でない
            logger.warning("Failed to save aggregated LLM raw log: %s", exc)

    def _call_llm(self, prompt_text: str, model_override: str | None = None, pass_label: str | None = None) -> str:
        provider = get_provider(self.provider_slug)
        payload = PromptPayload(system_prompt="", user_prompt=prompt_text)
        
        # Add model override to metadata if specified
        metadata: Dict[str, Any] = {}
        if model_override:
            if self.provider_slug == "google":
                metadata["google_model"] = model_override
            elif self.provider_slug == "openai":
                metadata["openai_model"] = model_override
            elif self.provider_slug == "anthropic":
                metadata["anthropic_model"] = model_override
            else:
                metadata["model"] = model_override
        # run_id / source_name / pass_label をメタデータに載せておくと、
        # プロバイダー側でトークン使用量をパス別に集計できる。
        if self.run_id:
            metadata["run_id"] = self.run_id
        if self.source_name:
            metadata["source_name"] = self.source_name
        if pass_label:
            metadata["pass_label"] = pass_label
        
        req = FormatterRequest(
            block_text="",
            provider=self.provider_slug,
            rewrite=False,
            metadata=metadata,
            line_max_chars=17.0,
            max_retries=1,
            temperature=self.temperature,
            timeout=self.timeout,
        )
        return provider.format(prompt=payload, request=req)

    def run(
        self,
        text: str,
        words: Sequence[WordTimestamp],
        *,
        max_chars: float = 17.0,
        enable_pass3: bool = True,
        progress_callback: Callable[[str, int], None] | None = None,
    ) -> TwoPassResult | None:
        if not words:
            raise FormatterError("wordタイムスタンプが空です")
        try:
            logger.info("two-pass: pass1 start (words=%d, provider=%s)", len(words), self.provider_slug)
            # Pass1: replace/delete only
            pass1_prompt = self._build_pass1_prompt(text, words)
            logger.debug("Calling LLM for Pass 1. Prompt length: %d", len(pass1_prompt))
            t_p1_start = time.perf_counter()
            try:
                raw1, parsed1 = _call_llm_with_parse(
                    self._call_llm,
                    pass_label="pass1",
                    prompt=pass1_prompt,
                    model_override=self.pass1_model,
                    retries=2,
                    soft_fail=False,
                    log_sink=self,
                )
            except FormatterError as exc:
                fallback_model = self._pass1_fallback_model
                if (
                    self.workflow_def.pass1_fallback_enabled
                    and fallback_model
                    and fallback_model != self.pass1_model
                    and (
                        _looks_like_context_limit_error(str(exc))
                        or _looks_like_model_not_found_error(str(exc))
                    )
                ):
                    logger.warning(
                        "pass1 failed with model=%s; retrying with fallback model=%s: %s",
                        self.pass1_model,
                        fallback_model,
                        exc,
                    )
                    if progress_callback:
                        progress_callback("LLM Pass 1（フォールバック）", 45)
                    raw1, parsed1 = _call_llm_with_parse(
                        self._call_llm,
                        pass_label="pass1",
                        prompt=pass1_prompt,
                        model_override=fallback_model,
                        retries=1,
                        soft_fail=False,
                        log_sink=self,
                    )
                else:
                    raise
            t_p1_end = time.perf_counter()
            record_pass_time(self.run_id, "pass1", t_p1_end - t_p1_start)
            ops = _parse_operations(parsed1)
            updated_words = _apply_operations(words, ops) if ops else list(words)

            if not updated_words:
                # If all words deleted, return empty result (valid case)
                return TwoPassResult(segments=[])

            logger.info("two-pass: pass2 start (words=%d)", len(updated_words))
            # Pass2: line splits
            if progress_callback:
                progress_callback("LLM Pass 2", 60)
            pass2_prompt = self._build_pass2_prompt(updated_words, max_chars=max_chars)
            logger.debug("Calling LLM for Pass 2. Prompt length: %d", len(pass2_prompt))
            t_p2_start = time.perf_counter()
            raw2, parsed2 = _call_llm_with_parse(
                self._call_llm,
                pass_label="pass2",
                prompt=pass2_prompt,
                model_override=self.pass2_model,
                retries=2,
                soft_fail=False,
                log_sink=self,
            )
            t_p2_end = time.perf_counter()
            record_pass_time(self.run_id, "pass2", t_p2_end - t_p2_start)
            lines = _parse_lines(parsed2)
            if not lines:
                raise FormatterError("行分割結果が空です")
            pass2_lines = lines

            # Pass3: Validation (now always executed)
            from src.llm.validators import detect_issues
            if not enable_pass3:
                logger.warning("enable_pass3=False は非推奨になりました。Pass3は常に実行されます。")

            issues = detect_issues(lines, updated_words)
            logger.info("two-pass: pass3 start (%d issues detected)", len(issues))
            for issue in issues:
                logger.debug("  - %s", issue.description)

            if progress_callback:
                progress_callback("LLM Pass 3", 80)
            pass3_prompt = self._build_pass3_prompt(lines, updated_words, issues)
            logger.debug("Calling LLM for Pass 3. Prompt length: %d", len(pass3_prompt))
            t_p3_start = time.perf_counter()
            raw3, parsed3 = _call_llm_with_parse(
                self._call_llm,
                pass_label="pass3",
                prompt=pass3_prompt,
                model_override=self.pass3_model,
                retries=2,
                soft_fail=True,
                log_sink=self,
            )
            t_p3_end = time.perf_counter()
            record_pass_time(self.run_id, "pass3", t_p3_end - t_p3_start)
            if parsed3:
                pass3_lines = _parse_lines(parsed3)
                if pass3_lines and _has_full_word_coverage(pass3_lines, updated_words):
                    if (
                        not self.workflow_def.allow_pass3_range_change
                        and not _has_same_line_ranges(pass2_lines, pass3_lines)
                    ):
                        logger.warning("Pass 3 changed line ranges; using Pass 2 output instead")
                    else:
                        lines = pass3_lines
                elif pass3_lines:
                    logger.warning(
                        "Pass 3 returned partial coverage (words=%d, lines=%d); using Pass 2 output instead",
                        len(updated_words),
                        len(pass3_lines),
                    )
                else:
                    logger.warning("Pass 3 returned empty lines, using Pass 2 output")
            else:
                logger.warning("Pass 3 parsing failed; using Pass 2 output")

            # Pass4: re-run only lines that violate length bounds
            fixed_lines: List[LineRange] = []
            pass4_needed = any(self._needs_pass4(line) for line in lines)
            if pass4_needed and progress_callback:
                progress_callback("LLM Pass 4", 95)
            for line in lines:
                if self._needs_pass4(line):
                    logger.info("pass4: line length %d out of bounds, retrying LLM", len(line.text))
                    t_p4_start = time.perf_counter()
                    repl = self._run_pass4_fix(line, updated_words)
                    t_p4_end = time.perf_counter()
                    record_pass_time(self.run_id, "pass4", t_p4_end - t_p4_start)
                    fixed_lines.extend(repl)
                else:
                    fixed_lines.append(line)
            lines = fixed_lines

            # 一部のプロバイダ（特に OpenAI）では、Pass2/Pass3 の lines が
            # 先頭付近だけに偏り、後半の単語がまったく行に割り当てられないケースがある。
            # その場合は、残りの単語を元の WordTimestamp から単純な行に起こす
            # フォールバックを行い、SRT が音声全体をカバーするようにする。
            lines = self._ensure_trailing_coverage(lines, updated_words)

            segments = self._ranges_to_segments(updated_words, lines)
            logger.info("two-pass: completed (segments=%d)", len(segments))
            return TwoPassResult(segments=segments)
        finally:
            # 一回の run ごとに raw LLM 応答を1ファイルにまとめてフラッシュ
            self._flush_logs()

    def _ranges_to_segments(self, words: Sequence[WordTimestamp], lines: Sequence[LineRange]) -> List["SubtitleSegment"]:
        from src.alignment.srt import SubtitleSegment

        segments: List[SubtitleSegment] = []
        last_end_time = 0.0

        for idx, line in enumerate(lines, start=1):
            if line.start_idx < 0 or line.end_idx >= len(words) or line.start_idx > line.end_idx:
                logger.warning("行範囲が不正なためスキップ: %s", line)
                continue

            # LLM（Pass2/Pass3/Pass4）が決めた行分割をそのまま採用する。
            current_text = line.text
            start = words[line.start_idx].start or 0.0
            end = words[line.end_idx].end or start

            # タイムスタンプの巻き戻りだけはローカルで補正する
            if start < last_end_time:
                logger.warning(
                    "Timestamp backward jump detected (clamped): %.2f -> %.2f", start, last_end_time
                )
                start = last_end_time

            # end < start になってしまった場合の最小補正
            if end < start:
                end = start + 0.1

            segments.append(SubtitleSegment(index=0, start=start, end=end, text=current_text))
            last_end_time = end

        # セグメント間の「タイムコードの空白時間」を埋めるため、
        # 次のセグメントの開始時刻まで前のセグメントの end を延長する。
        if self.fill_gaps:
            self._fill_segment_gaps(
                segments,
                max_gap=self.max_gap_duration,
                gap_padding=self.gap_padding
            )

        # start_delay: 2番目以降のセグメントの開始時間を遅らせる
        # 最初のセグメントのstartと最後のセグメントのendは維持
        if self.start_delay > 0 and len(segments) > 1:
            original_last_end = segments[-1].end

            for i in range(1, len(segments)):
                # 遅延を適用（ただし、次のセグメントの元startを超えないよう制限）
                new_start = segments[i].start + self.start_delay
                # オーバーラップ防止: 前のセグメントのendより後ろになるよう制限
                if new_start < segments[i - 1].end:
                    new_start = segments[i - 1].end
                segments[i].start = new_start

            # 遅延適用後、再度gap埋めを実行して隙間を埋める
            if self.fill_gaps:
                self._fill_segment_gaps(
                    segments,
                    max_gap=self.max_gap_duration,
                    gap_padding=self.gap_padding
                )

            # 最後のセグメントのendを元の値に戻す
            segments[-1].end = original_last_end

        # Re-assign indices
        for i, seg in enumerate(segments, start=1):
            seg.index = i

        return segments

    def _fill_segment_gaps(
        self,
        segments: List["SubtitleSegment"],
        max_gap: float | None = None,
        gap_padding: float = 0.0,
    ) -> None:
        """
        連続する SubtitleSegment 間に存在するタイムコードの空白時間を埋める。

        具体的には、次のセグメントの start が現在の end より後ろにある場合、
        現在の end を次の start まで延長し、画面上のテロップが途切れないようにする。

        さらに gap_padding (秒) が指定されている場合、現在の end に余韻（パディング）を追加する。
        ただし、パディング追加後の end が次の start を超えてしまう（オーバーラップする）場合は、
        次の start で止める（次の字幕の開始時刻を優先し、遅らせない）。

        Args:
            segments: 対象の SubtitleSegment リスト（in-place で更新）
            max_gap: 埋める最大ギャップ秒数。None の場合は上限なし（すべてのギャップを埋める）。
                     例えば 10.0 を指定すると、10秒以上のギャップは埋めない。
            gap_padding: 各セグメントの末尾に追加する余韻時間（秒）。デフォルト0.0。
                         例えば 0.15 を指定すると、最低でも0.15秒の余韻を確保しようとする。
                         （ただし次の字幕開始まで）
        """
        if not segments:
            return

        if not segments:
            return

        for i in range(len(segments) - 1):
            current = segments[i]
            nxt = segments[i + 1]

            # ギャップがある場合 -> 埋めるかどうか判定
            gap = nxt.start - current.end

            if gap > 0:
                # max_gap判定: ギャップが大きすぎる場合は完全に埋めない
                if max_gap is not None and gap > max_gap:
                    logger.debug(
                        "Gap %.2fs exceeds max_gap %.2fs; not filling (segment %d)",
                        gap,
                        max_gap,
                        i + 1,
                    )
                    # ただし、gap_padding 分だけは「余韻」として確保したい
                    if gap_padding > 0:
                        desired_end = current.end + gap_padding
                        # ただし絶対に次の開始を超えてはいけない
                        current.end = min(desired_end, nxt.start)
                    continue

                # ギャップが許容範囲内なら、次の開始位置まで完全に埋める（隙間ゼロ）
                current.end = nxt.start

    def _ensure_trailing_coverage(
        self,
        lines: Sequence[LineRange],
        words: Sequence[WordTimestamp],
    ) -> List[LineRange]:
        """
        LLM が先頭側だけの行しか返さず、末尾の単語に対応する行が生成されなかった場合に、
        残りの単語を元にシンプルな行を追加して「音声全体をカバーする」ことを保証する。

        既に最後の行が末尾の単語までカバーしている場合は、入力の lines をそのまま返す。
        """
        if not lines or not words:
            return list(lines)

        max_idx = len(words) - 1
        # lines のうち、最も大きい end_idx を持つものを取得
        last_line = max(lines, key=lambda l: l.end_idx)
        if last_line.end_idx >= max_idx:
            # すでに末尾までカバーされているのでフォールバック不要
            return list(lines)

        # 末尾側に未カバー領域がある場合のみ、簡易な行分割で補完する
        gap_start = max(last_line.end_idx + 1, 0)
        if gap_start > max_idx:
            return list(lines)

        fallback_lines: List[LineRange] = list(lines)

        idx = gap_start
        while idx <= max_idx:
            line_start = idx
            text_parts: List[str] = []
            current_len = 0

            # 最低 1 つは単語を含める & 17 文字程度を目安に分割（既存仕様を緩く模倣）
            while idx <= max_idx:
                word = words[idx].word or ""
                # 日本語前提のため、ここでは単純に連結（スペースは挿入しない）
                next_len = current_len + len(word)
                # すでにある程度の長さがあり、これ以上繋ぐと 17 文字を大きく超える場合は改行
                if text_parts and next_len > 17 and current_len >= 5:
                    break
                text_parts.append(word)
                current_len = next_len
                idx += 1

                # ちょうど良い長さになったら一旦区切る
                if current_len >= 10 and (idx > max_idx or current_len >= 17):
                    break

            line_end = idx - 1
            if not text_parts:
                # 万一 word が空文字のみだった場合でも進捗を進めて無限ループを防ぐ
                idx += 1
                continue

            fallback_lines.append(
                LineRange(
                    start_idx=line_start,
                    end_idx=line_end,
                    text="".join(text_parts),
                )
            )

        # 元の行とフォールバック行を start_idx でソートして返す
        return sorted(fallback_lines, key=lambda l: (l.start_idx, l.end_idx))

    def _build_pass1_prompt(self, raw_text: str, words: Sequence[WordTimestamp]) -> str:
        if self.workflow_def.pass1_prompt is None:
            raise FormatterError(f"未設定のワークフローです: {self.workflow}")
        return self.workflow_def.pass1_prompt(raw_text, words)

    def _build_pass2_prompt(self, words: Sequence[WordTimestamp], *, max_chars: float) -> str:
        if self.workflow_def.pass2_prompt is None:
            raise FormatterError(f"未設定のワークフローです: {self.workflow}")
        return self.workflow_def.pass2_prompt(words, max_chars)

    def _build_pass3_prompt(self, lines: Sequence[LineRange], words: Sequence[WordTimestamp], issues) -> str:
        if self.workflow_def.pass3_prompt is None:
            raise FormatterError(f"未設定のワークフローです: {self.workflow}")
        return self.workflow_def.pass3_prompt(lines, words, issues, self.glossary_terms)

    def _needs_pass4(self, line: LineRange) -> bool:
        return len(line.text) > 17 or len(line.text) < 5

    def _build_pass4_prompt(self, line: LineRange, words: Sequence[WordTimestamp]) -> str:
        if self.workflow_def.pass4_prompt is None:
            raise FormatterError(f"未設定のワークフローです: {self.workflow}")
        return self.workflow_def.pass4_prompt(line, words)

    def _run_pass4_fix(self, line: LineRange, words: Sequence[WordTimestamp]) -> List[LineRange]:
        prompt = self._build_pass4_prompt(line, words)
        raw, parsed = _call_llm_with_parse(
            self._call_llm,
            pass_label="pass4",
            prompt=prompt,
            model_override=self.pass4_model,
            retries=1,
            soft_fail=True,
            log_sink=self,
        )
        if parsed:
            repl = _parse_lines(parsed)
            if repl:
                return repl
        logger.warning("pass4 failed or empty; keeping original line (%d-%d)", line.start_idx, line.end_idx)
        return [line]


__all__ = ["TwoPassFormatter", "TwoPassResult"]

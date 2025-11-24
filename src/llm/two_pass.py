"""Two-pass LLM formatter (Pass1: replace/delete, Pass2: 17-char line splits)."""
from __future__ import annotations

import json
import logging
import re
import uuid
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Sequence, Dict, Any, Iterable

from src.config import get_settings
from src.llm.formatter import FormatterError, get_provider, FormatterRequest
from src.llm.prompts import PromptPayload
from src.transcribe.base import WordTimestamp
from src.llm.usage_metrics import record_pass_time

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
        ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
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
        raw = call_fn(prompt, model_override=model_override, pass_label=pass_label)
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
    raise last_exc  # type: ignore[misc]


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
        *,
        run_id: str | None = None,
        source_name: str | None = None,
    ) -> None:
        settings = get_settings().llm
        self.provider_slug = llm_provider
        self.temperature = temperature
        self.timeout = timeout
        # Per-pass model selection
        self.pass1_model = pass1_model or settings.pass1_model
        self.pass2_model = pass2_model or settings.pass2_model
        self.pass3_model = pass3_model or settings.pass3_model  # Default to Flash for cost efficiency
        # Pass4 は settings.pass4_model（= LLM_PASS4_MODEL or LLM_PASS3_MODEL）を尊重しつつ、
        # プロファイルやCLIからの引数で上書きできるようにする。
        self.pass4_model = pass4_model or settings.pass4_model
        # ログ用コンテキスト（処理単位で1ファイルにまとめる）
        self.run_id = run_id
        self.source_name = source_name
        self._log_buffer: Dict[str, str] = {}
        self._log_date_str = datetime.utcnow().strftime("%Y%m%d")
        self._log_written = False

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
            RAW_LOG_DIR.mkdir(parents=True, exist_ok=True)
            base_name = (self.source_name or self.run_id or "llm_run").replace("/", "_")
            # yyyymmdd_連番 の連番部分は run_id があればそれを使い、無ければ uuid 短縮
            suffix = self.run_id or uuid.uuid4().hex[:8]
            fname = RAW_LOG_DIR / f"{base_name}_{self._log_date_str}_{suffix}.txt"
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
    ) -> TwoPassResult | None:
        if not words:
            raise FormatterError("wordタイムスタンプが空です")
        try:
            logger.info("two-pass: pass1 start (words=%d, provider=%s)", len(words), self.provider_slug)
            # Pass1: replace/delete only
            pass1_prompt = self._build_pass1_prompt(text, words)
            logger.debug("Calling LLM for Pass 1. Prompt length: %d", len(pass1_prompt))
            t_p1_start = time.perf_counter()
            raw1, parsed1 = _call_llm_with_parse(
                self._call_llm,
                pass_label="pass1",
                prompt=pass1_prompt,
                model_override=self.pass1_model,
                retries=2,
                soft_fail=False,
                log_sink=self,
            )
            t_p1_end = time.perf_counter()
            record_pass_time(self.run_id, "pass1", t_p1_end - t_p1_start)
            ops = _parse_operations(parsed1)
            updated_words = _apply_operations(words, ops) if ops else list(words)

            if not updated_words:
                # If all words deleted, return empty result (valid case)
                return TwoPassResult(segments=[])

            logger.info("two-pass: pass2 start (words=%d)", len(updated_words))
            # Pass2: line splits
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

            # Pass3: Validation (now always executed)
            from src.llm.validators import detect_issues
            if not enable_pass3:
                logger.warning("enable_pass3=False は非推奨になりました。Pass3は常に実行されます。")

            issues = detect_issues(lines, updated_words)
            logger.info("two-pass: pass3 start (%d issues detected)", len(issues))
            for issue in issues:
                logger.debug("  - %s", issue.description)

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
                if pass3_lines:
                    lines = pass3_lines
                else:
                    logger.warning("Pass 3 returned empty lines, using Pass 2 output")
            else:
                logger.warning("Pass 3 parsing failed; using Pass 2 output")

            # Pass4: re-run only lines that violate length bounds
            fixed_lines: List[LineRange] = []
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

        # Re-assign indices
        for i, seg in enumerate(segments, start=1):
            seg.index = i
            
        return segments

    def _build_pass1_prompt(self, raw_text: str, words: Sequence[WordTimestamp]) -> str:
        indexed = _build_indexed_words(words)
        return (
            "あなたはプロの字幕エディターです。以下の単語列を順番を変えずに最小限の修正だけ加えてください。\n"
            "- 許可される操作: replace, delete（挿入は禁止。音声に無い単語を足さないこと）。\n"
            "- 単語の順序は変えないでください。\n"
            "- 出力は JSON の operations 配列のみ。説明文・コードフェンスは禁止。\n\n"
            f"入力テキスト:\n{raw_text}\n\n"
            f"単語リスト（index:word）:\n{indexed}\n\n"
            "出力フォーマット例:\n"
            '{\n'
            '  "operations": [\n'
            '    {"type": "replace", "start_idx": 10, "end_idx": 11, "text": "カレーライス"},\n'
            '    {"type": "delete", "start_idx": 25, "end_idx": 25}\n'
            '  ]\n'
            '}\n'
            "追加の説明・前後文字列・コードフェンスは一切不要です。"
        )

    def _build_pass2_prompt(self, words: Sequence[WordTimestamp], *, max_chars: float) -> str:
        indexed = _build_indexed_words(words)
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
            "1. **助詞で分割しない**: が、を、に、で、の、は、も、から、まで、よ、ね、な、わ、ぞ、ぜ、さ 等で行を終わらせない\n"
            "2. **接続表現・接続詞で分割しない**: 〜と思って、〜ものの、〜たら、〜ので、〜けど、〜けれど、んで、それで、そして 等で文を切らない\n"
            "3. **最小行長の確保**: 1行は最低でも5文字以上必要（1〜4文字のみの行は絶対に禁止）\n"
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
            f"1. 助詞・接続表現・終助詞での分割を避ける（上記ルール参照）\n"
            f"2. 文字数オーバー: 現バッファ＋次チャンクが{int(max_chars)}文字を超える場合のみ改行\n"
            f"3. 文脈区切り: {int(max_chars)}文字以内でも、読点・強い切れ目（〜ます、〜です、〜だ等）で終わるなら改行を検討\n"
            "4. 最小行長チェック: 分割後の行が4文字未満にならないか確認（3文字以下は禁止）\n\n"
            "## Step 3: クリーニング (Cleaning)\n"
            "行末の句読点（、。）を削除。文中の句読点は残してよい。\n\n"
            "# Input\n"
            f"単語リスト（index:word）:\n{indexed}\n\n"
            "# Output\n"
            "以下のJSONだけを返してください（説明・コードフェンス禁止）。例:\n"
            '{\n'
            '  "lines": [\n'
            '    {"from": 0, "to": 10, "text": "私は大学の12月ぐらい"},\n'
            '    {"from": 11, "to": 25, "text": "政治家になろうと決めていて"}\n'
            '  ]\n'
            '}\n'
        )

    def _build_pass3_prompt(self, lines: Sequence[LineRange], words: Sequence[WordTimestamp], issues) -> str:
        """Build Pass 3 prompt for fixing detected issues."""
        indexed = _build_indexed_words(words)

        # Format issues for LLM
        if issues:
            issue_text = "\n".join(
                [f"- {issue.description} → {issue.suggested_action}" for issue in issues]
            )
        else:
            issue_text = "問題は検出されませんでした。全行を確認し、以下のルールに従って最小限の修正を行ってください。"

        # Format current lines as JSON
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
            '{\n'
            '  "lines": [\n'
            '    {"from": 0, "to": 11, "text": "私は大学の時の12月ぐらいかな"},\n'
            '    {"from": 12, "to": 18, "text": "4年生の12月には"},\n'
            '    {"from": 19, "to": 28, "text": "政治家になろうという腹を決めていて"},\n'
            '    {"from": 29, "to": 33, "text": "1月ぐらいから"},\n'
            '    {"from": 34, "to": 45, "text": "司法試験予備校に申し込んで"}\n'
            '  ]\n'
            '}\n'
        )

    def _needs_pass4(self, line: LineRange) -> bool:
        return len(line.text) > 17 or len(line.text) < 5

    def _build_pass4_prompt(self, line: LineRange, words: Sequence[WordTimestamp]) -> str:
        indexed = "\n".join(
            f"{i}: {w.word}"
            for i, w in enumerate(words[line.start_idx : line.end_idx + 1], start=line.start_idx)
        )
        return (
            "# Role\n"
            "あなたはテロップ最終チェックの追加ステップ担当です。与えられた行に対してのみ、条件を満たす複数行に必要最小限で分割してください。\n\n"
            "# Constraints\n"
            "- 必ず1行あたり全角5〜17文字に収めること\n"
            "- 語順を変えない、語を追加/削除しない\n"
            "- 要約・翻訳・意訳をしない\n"
            "- 行末の句読点（、。）は削除。文中の句読点は残してよい\n"
            "- 改行の優先度: (1)「。?!」直後 → (2)「、」直後 → (3) 接続助詞・係助詞など自然な切れ目。\n\n"
            "# Input\n"
            f"対象の行テキスト:\n{line.text}\n\n"
            f"単語リスト（index:word）:\n{indexed}\n\n"
            "# Output\n"
            "以下のJSONだけを返してください（説明・コードフェンス禁止）。例:\n"
            '{\n'
            '  "lines": [\n'
            '    {"from": 100, "to": 105, "text": "...."},\n'
            '    {"from": 106, "to": 110, "text": "...."}\n'
            '  ]\n'
            '}\n'
        )

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

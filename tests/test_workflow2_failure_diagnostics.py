import json
from pathlib import Path

import pytest

from src.config import reload_settings
from src.llm.formatter import BaseLLMProvider, FormatterError, register_provider
from src.llm.two_pass import TwoPassFormatter
from src.pipeline.poc import PocRunOptions, execute_poc_run
from src.transcribe.base import WordTimestamp


@register_provider
class DummyAlwaysFailProvider(BaseLLMProvider):
    slug = "test-workflow2-always-fail-provider"
    display_name = "Dummy Always Fail Provider"

    def format(self, prompt, request):
        pass_label = (request.metadata or {}).get("pass_label") or ""
        if pass_label == "pass1":
            raise FormatterError("Google Gemini API Error: 400 Request too large")
        return json.dumps({"operations": []}, ensure_ascii=False)


def test_workflow2_writes_llm_raw_on_api_error(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    audio_path = Path("audio.m4a")
    audio_path.write_bytes(b"dummy")

    options = PocRunOptions(
        subtitle_dir=Path("output"),
        output_dir=Path("temp/poc_samples"),
        progress_dir=Path("temp/progress"),
        simulate=True,
        llm_provider=DummyAlwaysFailProvider.slug,
        save_logs=True,
        workflow="workflow2",
        timestamp="20251215T000000",
    )

    with pytest.raises(FormatterError):
        execute_poc_run([audio_path], ["mlx"], options)

    run_id = f"{audio_path.stem}_mlx_{options.timestamp}"
    run_dir = Path("output") / f"{audio_path.stem}_{options.timestamp}"
    raw_dir = run_dir / "logs/llm_raw"
    raw_files = list(raw_dir.glob(f"{audio_path.name}_*_{run_id}.txt"))
    assert len(raw_files) == 1
    assert "[API ERROR]" in raw_files[0].read_text(encoding="utf-8")


def test_workflow2_pass1_model_fallback(monkeypatch):
    monkeypatch.setenv("LLM_PASS1_MODEL", "pro-model")
    monkeypatch.setenv("LLM_WF2_PASS1_MODEL", "flash-model")
    reload_settings()

    calls: list[tuple[str | None, str | None]] = []

    def fake_call_llm(self, payload: str, model_override=None, pass_label=None):
        calls.append((pass_label, model_override))
        if pass_label == "pass1" and model_override == "flash-model":
            raise FormatterError("Google Gemini API Error: 400 Request too large")
        if pass_label == "pass1":
            return json.dumps({"operations": []}, ensure_ascii=False)
        if pass_label in {"pass2", "pass3"}:
            return json.dumps({"lines": [{"from": 0, "to": 2, "text": "テストです"}]}, ensure_ascii=False)
        raise AssertionError(f"unexpected pass_label={pass_label}")

    monkeypatch.setattr("src.llm.validators.detect_issues", lambda lines, words: [])
    monkeypatch.setattr(TwoPassFormatter, "_call_llm", fake_call_llm)

    formatter = TwoPassFormatter(
        llm_provider="google",
        workflow="workflow2",
        run_id="test_run",
        source_name="test_source",
    )
    words = [
        WordTimestamp(word="テスト", start=0.0, end=0.2),
        WordTimestamp(word="です", start=0.2, end=0.4),
        WordTimestamp(word="ね", start=0.4, end=0.6),
    ]
    result = formatter.run(text="テストですね", words=words, progress_callback=None)

    assert result is not None
    assert calls[0] == ("pass1", "flash-model")
    assert calls[1] == ("pass1", "pro-model")

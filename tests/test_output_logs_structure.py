import json
from pathlib import Path

from src.llm.formatter import BaseLLMProvider, register_provider
from src.pipeline.poc import PocRunOptions, execute_poc_run


@register_provider
class DummyJsonLlmProvider(BaseLLMProvider):
    slug = "test-dummy-json-llm-provider"
    display_name = "Dummy JSON LLM Provider"

    def format(self, prompt, request):
        pass_label = (request.metadata or {}).get("pass_label") or ""
        if pass_label == "pass1":
            return json.dumps({"operations": []}, ensure_ascii=False)
        if pass_label in {"pass2", "pass3", "pass4"}:
            return json.dumps(
                {"lines": [{"from": 0, "to": 4, "text": "テストです"}]},
                ensure_ascii=False,
            )
        return json.dumps({"lines": []}, ensure_ascii=False)


def test_execute_poc_run_save_logs_writes_under_output_dir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    audio_path = Path("audio.m4a")
    audio_path.write_bytes(b"dummy")

    options = PocRunOptions(
        subtitle_dir=Path("output"),
        output_dir=Path("temp/poc_samples"),
        progress_dir=Path("temp/progress"),
        simulate=True,
        llm_provider=DummyJsonLlmProvider.slug,
        save_logs=True,
        timestamp="20251215T000000",
    )
    saved_paths = execute_poc_run([audio_path], ["whisper-local"], options)

    run_id = f"{audio_path.stem}_whisper-local_{options.timestamp}"
    run_dir = Path("output") / f"{audio_path.stem}_{options.timestamp}"
    assert (run_dir / f"{run_id}.srt").exists()
    assert (run_dir / "logs/poc_samples" / f"{run_id}.json").exists()
    assert (run_dir / "logs/progress" / f"{run_id}.json").exists()

    raw_dir = run_dir / "logs/llm_raw"
    raw_files = list(raw_dir.glob(f"{audio_path.name}_*_{run_id}.txt"))
    assert len(raw_files) == 1
    raw_text = raw_files[0].read_text(encoding="utf-8")
    assert "===== pass1 =====" in raw_text
    assert "===== pass2 =====" in raw_text
    assert "===== pass3 =====" in raw_text

    metrics_dir = run_dir / "logs/metrics"
    metrics_files = list(metrics_dir.glob(f"{audio_path.name}_*_{run_id}_metrics.json"))
    assert len(metrics_files) == 1

    assert any(path.name == f"{run_id}.srt" for path in saved_paths)

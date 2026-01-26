from __future__ import annotations

from pathlib import Path
import sys

test_dir = Path(__file__).resolve().parent
project_root = test_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import src.llm.formatter as fmt
from src.pipeline.poc import PocRunOptions, execute_poc_run


def test_execute_poc_run_passes_temperature_to_formatter(tmp_path):
    registry_backup = fmt._PROVIDER_REGISTRY.copy()

    captured = {}

    class DummyProvider(fmt.BaseLLMProvider):
        slug = "tempcheck"
        display_name = "TempCheck"
        def __init__(self):
            self.calls = 0

        def format(self, prompt, request):
            captured["temperature"] = request.temperature
            if "operations" in prompt.user_prompt:
                return '{"operations": []}'
            return '{"lines":[{"from":0,"to":0,"text":"hello"}]}'

    fmt.register_provider(DummyProvider)

    audio = tmp_path / "audio.m4a"
    audio.write_text("dummy")

    options = PocRunOptions(
        output_dir=tmp_path / "out",
        progress_dir=tmp_path / "prog",
        subtitle_dir=tmp_path / "subs",
        simulate=True,
        llm_provider="tempcheck",
        llm_temperature=0.42,
        rewrite=False,
        timestamp="20250101T000000",
    )

    try:
        execute_poc_run([audio], ["whisper-local"], options)
    finally:
        fmt._PROVIDER_REGISTRY.clear()
        fmt._PROVIDER_REGISTRY.update(registry_backup)

    assert captured["temperature"] == 0.42

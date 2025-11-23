from __future__ import annotations

from pathlib import Path
import sys

test_dir = Path(__file__).resolve().parent
project_root = test_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import src.llm.formatter as fmt
from src.pipeline.poc import PocRunOptions, execute_poc_run


def test_execute_poc_run_generates_srt_with_dummy_provider(tmp_path):
    # preserve registry
    registry_backup = fmt._PROVIDER_REGISTRY.copy()

    class DummyProvider(fmt.BaseLLMProvider):
        slug = "dummy"
        display_name = "Dummy Provider"
        def __init__(self):
            self.calls = 0

        def format(self, prompt, request):
            # Pass1: operations / Pass2+: lines
            if "operations" in prompt.user_prompt:
                return '{"operations": []}'
            return '{"lines":[{"from":0,"to":0,"text":"こんにちは世界"}]}'

    fmt.register_provider(DummyProvider)

    audio = tmp_path / "audio.m4a"
    audio.write_text("dummy-audio")

    options = PocRunOptions(
        output_dir=tmp_path / "out",
        progress_dir=tmp_path / "prog",
        subtitle_dir=tmp_path / "subs",
        simulate=True,
        llm_provider="dummy",
        rewrite=False,
        timestamp="20250101T000000",
    )

    try:
        execute_poc_run([audio], ["openai"], options)
    finally:
        fmt._PROVIDER_REGISTRY.clear()
        fmt._PROVIDER_REGISTRY.update(registry_backup)

    srt_files = list((tmp_path / "subs").glob("*.srt"))
    assert len(srt_files) == 1
    content = srt_files[0].read_text()
    assert "こんにちは" in content
    assert "世界" in content

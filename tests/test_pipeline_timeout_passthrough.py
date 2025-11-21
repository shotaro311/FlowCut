from __future__ import annotations

from pathlib import Path
import sys

test_dir = Path(__file__).resolve().parent
project_root = test_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import src.llm.formatter as fmt
from src.pipeline.poc import PocRunOptions, execute_poc_run


def test_execute_poc_run_passes_timeout(tmp_path):
    registry_backup = fmt._PROVIDER_REGISTRY.copy()
    seen = {}

    class DummyProvider(fmt.BaseLLMProvider):
        slug = "timeoutcheck"
        display_name = "TimeoutCheck"

        def format(self, prompt, request):
            seen["timeout"] = request.timeout
            return "hi[WORD: hi]"

    fmt.register_provider(DummyProvider)

    audio = tmp_path / "audio.m4a"
    audio.write_text("dummy")

    options = PocRunOptions(
        output_dir=tmp_path / "out",
        progress_dir=tmp_path / "prog",
        subtitle_dir=tmp_path / "subs",
        simulate=True,
        llm_provider="timeoutcheck",
        llm_timeout=12.5,
        timestamp="20250101T000000",
    )

    try:
        execute_poc_run([audio], ["openai"], options, formatter=fmt.LLMFormatter(strict_validation=False))
    finally:
        fmt._PROVIDER_REGISTRY.clear()
        fmt._PROVIDER_REGISTRY.update(registry_backup)

    assert seen["timeout"] == 12.5

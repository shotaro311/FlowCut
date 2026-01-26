from __future__ import annotations

from pathlib import Path
import sys
import json

test_dir = Path(__file__).resolve().parent
project_root = test_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import src.llm.formatter as fmt
from src.pipeline.poc import PocRunOptions, execute_poc_run


def test_execute_poc_run_uses_pass5_provider_when_overridden(tmp_path):
    registry_backup = fmt._PROVIDER_REGISTRY.copy()

    class DummyMainProvider(fmt.BaseLLMProvider):
        slug = "dummy-main"
        display_name = "Dummy Main Provider"

        def format(self, prompt, request):
            if "operations" in prompt.user_prompt:
                return '{"operations": []}'
            return '{"lines":[{"from":0,"to":0,"text":"こんにちは世界こんにちは"}]}'

    class DummyPass5Provider(fmt.BaseLLMProvider):
        slug = "dummy-pass5"
        display_name = "Dummy Pass5 Provider"

        def format(self, prompt, request):
            assert (request.metadata or {}).get("pass_label") == "pass5"
            marker_in = "# Input(JSON)\n"
            marker_out = "\n\n# Output(JSON)"
            start = prompt.user_prompt.find(marker_in)
            end = prompt.user_prompt.find(marker_out)
            payload = json.loads(prompt.user_prompt[start + len(marker_in) : end].strip())
            items = payload.get("lines", [])
            out = []
            for item in items:
                text = str(item.get("text") or "")
                if text == "こんにちは世界こんにちは":
                    text = "こんにちは世界\\nこんにちは"
                out.append({"index": int(item.get("index") or 0), "text": text})
            return json.dumps({"lines": out}, ensure_ascii=False)

    fmt.register_provider(DummyMainProvider)
    fmt.register_provider(DummyPass5Provider)

    audio = tmp_path / "audio.m4a"
    audio.write_text("dummy-audio")

    options = PocRunOptions(
        output_dir=tmp_path / "out",
        progress_dir=tmp_path / "prog",
        subtitle_dir=tmp_path / "subs",
        simulate=True,
        llm_provider=DummyMainProvider.slug,
        enable_pass5=True,
        pass5_max_chars=8,
        pass5_provider=DummyPass5Provider.slug,
        timestamp="20250101T000000",
    )

    try:
        execute_poc_run([audio], ["whisper-local"], options)
    finally:
        fmt._PROVIDER_REGISTRY.clear()
        fmt._PROVIDER_REGISTRY.update(registry_backup)

    srt_files = list((tmp_path / "subs").glob("*.srt"))
    assert len(srt_files) == 1
    content = srt_files[0].read_text(encoding="utf-8")
    assert "こんにちは世界\nこんにちは" in content

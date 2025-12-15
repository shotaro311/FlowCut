import json

from src.llm.formatter import BaseLLMProvider, register_provider
from src.llm.pass5_processor import Pass5Processor


@register_provider
class DummyPass5Provider(BaseLLMProvider):
    slug = "test-pass5-provider"
    display_name = "Dummy Pass5 Provider"

    def format(self, prompt, request):
        assert (request.metadata or {}).get("pass_label") == "pass5"
        return json.dumps({"lines": [{"index": 0, "text": "長い文章を\\n改行しました"}]}, ensure_ascii=False)


def test_pass5_processor_rewrites_only_long_entries():
    srt = (
        "1\n"
        "00:00:00,000 --> 00:00:01,000\n"
        "長い文章を改行する必要があります\n\n"
        "2\n"
        "00:00:01,000 --> 00:00:02,000\n"
        "短い\n"
    )
    out = Pass5Processor(provider=DummyPass5Provider.slug, max_chars=8, run_id="run1").process(srt)
    assert "長い文章を\n改行しました" in out
    assert "短い" in out

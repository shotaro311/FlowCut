from __future__ import annotations

from pathlib import Path
import sys
import pytest

test_dir = Path(__file__).resolve().parent
project_root = test_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.llm.two_pass import _extract_json as extract_json_two_pass
from src.llm.pass5_processor import _extract_json as extract_json_pass5


@pytest.mark.parametrize(
    "extractor",
    [extract_json_two_pass, extract_json_pass5],
)
def test_extract_json_accepts_plain_json(extractor):
    assert extractor('{"operations": []}') == {"operations": []}
    assert extractor('{"lines":[{"from":0,"to":0,"text":"hello"}]}') == {
        "lines": [{"from": 0, "to": 0, "text": "hello"}]
    }


@pytest.mark.parametrize(
    "extractor",
    [extract_json_two_pass, extract_json_pass5],
)
def test_extract_json_tolerates_code_fences_and_trailing_text(extractor):
    raw = "Here is the result:\n```json\n{\"operations\": []}\n```\nThanks!"
    assert extractor(raw) == {"operations": []}

    raw2 = "prefix...\n{\"lines\":[{\"from\":0,\"to\":0,\"text\":\"hello\"}]}\n(suffix)"
    assert extractor(raw2) == {"lines": [{"from": 0, "to": 0, "text": "hello"}]}


@pytest.mark.parametrize(
    "extractor",
    [extract_json_two_pass, extract_json_pass5],
)
def test_extract_json_errors_on_empty_response(extractor):
    with pytest.raises(ValueError):
        extractor("")
    with pytest.raises(ValueError):
        extractor("   \n\t")


@pytest.mark.parametrize(
    "extractor",
    [extract_json_two_pass, extract_json_pass5],
)
def test_extract_json_errors_when_no_json_present(extractor):
    with pytest.raises(ValueError):
        extractor("not json")


from src.llm.workflows.workflow1 import WORKFLOW as WORKFLOW1
from src.llm.workflows.workflow2 import WORKFLOW as WORKFLOW2


def test_workflow2_prompts_match_workflow1():
    assert WORKFLOW2.pass1_prompt is WORKFLOW1.pass1_prompt
    assert WORKFLOW2.pass2_prompt is WORKFLOW1.pass2_prompt
    assert WORKFLOW2.pass3_prompt is WORKFLOW1.pass3_prompt
    assert WORKFLOW2.pass4_prompt is WORKFLOW1.pass4_prompt

    assert WORKFLOW2.optimized_pass4 == WORKFLOW1.optimized_pass4
    assert WORKFLOW2.allow_pass3_range_change == WORKFLOW1.allow_pass3_range_change
    assert WORKFLOW2.pass1_fallback_enabled == WORKFLOW1.pass1_fallback_enabled

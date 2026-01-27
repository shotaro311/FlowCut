from src.llm.workflows.workflow2 import WORKFLOW


def test_workflow2_definition_is_complete():
    assert WORKFLOW.slug == "workflow2"
    assert WORKFLOW.wf_env_number == 2
    assert WORKFLOW.pass1_prompt is not None
    assert WORKFLOW.pass2_prompt is not None
    assert WORKFLOW.pass3_prompt is not None
    assert WORKFLOW.pass4_prompt is not None

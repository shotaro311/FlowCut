from src.llm.workflows.registry import get_workflow, is_known_workflow, list_workflows


def test_workflow_registry_lists_workflows():
    slugs = [wf.slug for wf in list_workflows()]
    assert "workflow2" in slugs
    assert slugs == ["workflow2"]


def test_unknown_workflow_falls_back_to_workflow2():
    assert get_workflow("unknown").slug == "workflow2"
    assert is_known_workflow("unknown") is False
    assert is_known_workflow("workflow2") is True

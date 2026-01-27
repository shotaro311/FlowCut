from __future__ import annotations

from src.llm.workflows.definition import WorkflowDefinition
from src.llm.workflows.workflow2 import WORKFLOW as WORKFLOW2

DEFAULT_WORKFLOW_SLUG = WORKFLOW2.slug

_WORKFLOWS: list[WorkflowDefinition] = [WORKFLOW2]

_BY_SLUG = {wf.slug: wf for wf in _WORKFLOWS}


def list_workflows() -> list[WorkflowDefinition]:
    return list(_WORKFLOWS)


def is_known_workflow(slug: str | None) -> bool:
    if slug is None:
        return False
    return slug.strip().lower() in _BY_SLUG


def get_workflow(slug: str | None) -> WorkflowDefinition:
    if slug is None:
        return WORKFLOW2
    key = slug.strip().lower()
    if not key:
        return WORKFLOW2
    return _BY_SLUG.get(key, WORKFLOW2)

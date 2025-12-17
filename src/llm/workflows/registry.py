from __future__ import annotations

from src.llm.workflows.definition import WorkflowDefinition
from src.llm.workflows.workflow1 import WORKFLOW as WORKFLOW1

try:  # pragma: no cover - workflowは削除される可能性がある
    from src.llm.workflows.workflow2 import WORKFLOW as WORKFLOW2  # type: ignore
except Exception:  # pragma: no cover
    WORKFLOW2 = None

try:  # pragma: no cover - workflowは削除される可能性がある
    from src.llm.workflows.workflow3 import WORKFLOW as WORKFLOW3  # type: ignore
except Exception:  # pragma: no cover
    WORKFLOW3 = None

DEFAULT_WORKFLOW_SLUG = WORKFLOW1.slug

_WORKFLOWS: list[WorkflowDefinition] = [WORKFLOW1]
for wf in [WORKFLOW2, WORKFLOW3]:
    if isinstance(wf, WorkflowDefinition):
        _WORKFLOWS.append(wf)

_BY_SLUG = {wf.slug: wf for wf in _WORKFLOWS}


def list_workflows() -> list[WorkflowDefinition]:
    return list(_WORKFLOWS)


def is_known_workflow(slug: str | None) -> bool:
    if slug is None:
        return False
    return slug.strip().lower() in _BY_SLUG


def get_workflow(slug: str | None) -> WorkflowDefinition:
    if slug is None:
        return WORKFLOW1
    key = slug.strip().lower()
    if not key:
        return WORKFLOW1
    return _BY_SLUG.get(key, WORKFLOW1)

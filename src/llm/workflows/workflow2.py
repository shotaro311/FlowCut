from __future__ import annotations

from src.llm.workflows.common import DEFAULT_PASS4_PROMPT
from src.llm.workflows.definition import WorkflowDefinition
from src.llm.workflows.workflow1 import build_pass1_prompt, build_pass2_prompt, build_pass3_prompt


WORKFLOW = WorkflowDefinition(
    slug="workflow2",
    label="workflow2: 標準（分割並列）",
    description="workflow1と同等のプロンプト/処理（Pass3で範囲変更あり）。長尺は約5分で分割し並列処理。",
    wf_env_number=2,
    optimized_pass4=False,
    allow_pass3_range_change=True,
    pass1_fallback_enabled=False,
    pass1_prompt=build_pass1_prompt,
    pass2_prompt=build_pass2_prompt,
    pass3_prompt=build_pass3_prompt,
    pass4_prompt=DEFAULT_PASS4_PROMPT,
)

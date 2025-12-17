from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence, TYPE_CHECKING

if TYPE_CHECKING:
    from src.llm.two_pass import LineRange
    from src.transcribe.base import WordTimestamp


Pass1PromptFn = Callable[[str, Sequence["WordTimestamp"]], str]
Pass2PromptFn = Callable[[Sequence["WordTimestamp"], float], str]
Pass3PromptFn = Callable[
    [Sequence["LineRange"], Sequence["WordTimestamp"], Any, Sequence[str]], str
]
Pass4PromptFn = Callable[["LineRange", Sequence["WordTimestamp"]], str]


@dataclass(frozen=True, slots=True)
class WorkflowDefinition:
    slug: str
    label: str
    description: str
    wf_env_number: int | None = None
    optimized_pass4: bool = False
    allow_pass3_range_change: bool = True
    pass1_fallback_enabled: bool = False
    pass1_prompt: Pass1PromptFn | None = None
    pass2_prompt: Pass2PromptFn | None = None
    pass3_prompt: Pass3PromptFn | None = None
    pass4_prompt: Pass4PromptFn | None = None


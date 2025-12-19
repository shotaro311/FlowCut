from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence, TYPE_CHECKING

if TYPE_CHECKING:
    from src.llm.two_pass import LineRange
    from src.transcribe.base import WordTimestamp


Pass1PromptFn = Callable[[str, Sequence["WordTimestamp"], Sequence[str]], str]
Pass2PromptFn = Callable[[Sequence["WordTimestamp"], float], str]
Pass3PromptFn = Callable[
    [Sequence["LineRange"], Sequence["WordTimestamp"], Any, Sequence[str]], str
]
Pass4PromptFn = Callable[["LineRange", Sequence["WordTimestamp"]], str]
Pass2to4PromptFn = Callable[[Sequence["WordTimestamp"], float, Sequence[str]], str]


@dataclass(frozen=True, slots=True)
class WorkflowDefinition:
    slug: str
    label: str
    description: str
    wf_env_number: int | None = None
    optimized_pass4: bool = False
    allow_pass3_range_change: bool = True
    pass3_enabled: bool = True
    pass1_fallback_enabled: bool = False
    two_call_enabled: bool = False
    pass1_prompt: Pass1PromptFn | None = None
    pass2_prompt: Pass2PromptFn | None = None
    pass3_prompt: Pass3PromptFn | None = None
    pass4_prompt: Pass4PromptFn | None = None
    pass2to4_prompt: Pass2to4PromptFn | None = None

    def is_two_call_mode(self) -> bool:
        return self.two_call_enabled and self.pass2to4_prompt is not None

    def active_pass_model_keys(self) -> list[str]:
        if self.is_two_call_mode():
            return ["pass1", "pass2"]
        keys = ["pass1", "pass2"]
        if self.pass3_enabled:
            keys.append("pass3")
        keys.append("pass4")
        return keys

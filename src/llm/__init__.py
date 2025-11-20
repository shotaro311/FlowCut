"""LLM整形モジュールの公開API。"""
from .formatter import (
    BaseLLMProvider,
    FormatterError,
    FormatterRequest,
    FormatterResult,
    FormattedLine,
    FormatValidationError,
    LineValidationIssue,
    LLMFormatter,
    available_providers,
    get_provider,
    register_provider,
)
from .prompts import PromptPayload, build_subtitle_prompt
from . import providers as _providers  # noqa: F401  # side-effect: register providers

__all__ = [
    "BaseLLMProvider",
    "FormatterError",
    "FormatterRequest",
    "FormatterResult",
    "FormattedLine",
    "FormatValidationError",
    "LineValidationIssue",
    "LLMFormatter",
    "available_providers",
    "get_provider",
    "register_provider",
    "PromptPayload",
    "build_subtitle_prompt",
]

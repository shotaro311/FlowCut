"""LLM整形処理のインターフェースとバリデーション。"""
from __future__ import annotations

from dataclasses import dataclass, field
import logging
import re
from typing import Any, Callable, Dict, List, Sequence, Tuple, Type
import unicodedata

from .prompts import PromptPayload, build_subtitle_prompt

logger = logging.getLogger(__name__)

_LINE_PATTERN = re.compile(r"^(?P<text>.*?)\s*\[WORD:\s*(?P<anchor>[^\]]+)\]\s*$")


def _display_width(text: str) -> float:
    width = 0.0
    for ch in text:
        width += 1.0 if unicodedata.east_asian_width(ch) in {"W", "F"} else 0.5
    return width


class FormatterError(RuntimeError):
    """LLM整形に関連するベース例外。"""


class ProviderNotRegisteredError(FormatterError):
    """指定されたプロバイダーが未登録の場合に送出。"""


class FormatValidationError(FormatterError):
    """出力行に問題があった場合の例外。"""

    def __init__(self, issues: Sequence["LineValidationIssue"]) -> None:
        self.issues = list(issues)
        super().__init__(f"LLM出力に{len(self.issues)}件のバリデーションエラー")


@dataclass(slots=True)
class LineValidationIssue:
    line_number: int
    text: str
    reason: str


@dataclass(slots=True)
class FormattedLine:
    text: str
    anchor_word: str | None
    raw: str
    line_number: int

    @property
    def display_width(self) -> float:
        return _display_width(self.text)


@dataclass(slots=True)
class FormatterRequest:
    block_text: str
    provider: str
    rewrite: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    line_max_chars: float = 17.0
    max_retries: int = 3
    temperature: float | None = None  # None の場合はプロバイダー規定値を使用


@dataclass(slots=True)
class FormatterResult:
    provider: str
    prompt: PromptPayload
    raw_output: str
    lines: List[FormattedLine]
    issues: List[LineValidationIssue] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_valid(self) -> bool:
        return not self.issues


class BaseLLMProvider:
    """LLMプロバイダーの共通IF。"""

    slug: str = "base"
    display_name: str = "Base LLM Provider"

    def format(self, prompt: PromptPayload, request: FormatterRequest) -> str:
        raise NotImplementedError


_PROVIDER_REGISTRY: Dict[str, Type[BaseLLMProvider]] = {}


def register_provider(cls: Type[BaseLLMProvider]) -> Type[BaseLLMProvider]:
    if not cls.slug or cls.slug == "base":
        raise ValueError("slug は一意な値を指定してください")
    if cls.slug in _PROVIDER_REGISTRY:
        logger.warning("LLMプロバイダー %s は上書き登録されます", cls.slug)
    _PROVIDER_REGISTRY[cls.slug] = cls
    return cls


def available_providers() -> List[str]:
    return sorted(_PROVIDER_REGISTRY.keys())


def get_provider(slug: str) -> BaseLLMProvider:
    try:
        provider_cls = _PROVIDER_REGISTRY[slug]
    except KeyError as exc:  # pragma: no cover - エラーハンドリングのみ
        raise ProviderNotRegisteredError(f"プロバイダー '{slug}' は未登録です: {available_providers()}") from exc
    return provider_cls()


class LLMFormatter:
    """ブロック単位でLLM整形を実行し、行を構造化する。"""

    def __init__(
        self,
        *,
        prompt_builder: Callable[[FormatterRequest], PromptPayload] | None = None,
        strict_validation: bool = True,
    ) -> None:
        self.prompt_builder = prompt_builder or (lambda req: build_subtitle_prompt(req.block_text, rewrite=req.rewrite, metadata=req.metadata))
        self.strict_validation = strict_validation

    def format_block(self, request: FormatterRequest) -> FormatterResult:
        if not request.block_text.strip():
            raise FormatterError("block_text が空です")
        if request.line_max_chars <= 0:
            raise FormatterError("line_max_chars には正の値を指定してください")
        provider = get_provider(request.provider)
        prompt = self.prompt_builder(request)
        raw_output = provider.format(prompt, request)
        lines, parse_issues = self._parse_output(raw_output)
        validation_issues = self._validate_lines(lines, request.line_max_chars)
        issues = [*parse_issues, *validation_issues]
        if self.strict_validation and issues:
            raise FormatValidationError(issues)
        metadata = {"rewrite": request.rewrite, **request.metadata}
        return FormatterResult(
            provider=provider.slug,
            prompt=prompt,
            raw_output=raw_output,
            lines=lines,
            issues=issues,
            metadata=metadata,
        )

    def _parse_output(self, raw_output: str) -> Tuple[List[FormattedLine], List[LineValidationIssue]]:
        lines: List[FormattedLine] = []
        issues: List[LineValidationIssue] = []
        stripped = raw_output.strip("\n")
        for idx, raw_line in enumerate(stripped.splitlines(), start=1):
            normalized = raw_line.strip()
            if not normalized:
                continue
            match = _LINE_PATTERN.match(normalized)
            if match:
                text = match.group("text").strip()
                anchor = match.group("anchor").strip()
            else:
                text = normalized
                anchor = None
                issues.append(LineValidationIssue(line_number=idx, text=normalized, reason="missing_word_tag"))
            lines.append(FormattedLine(text=text, anchor_word=anchor, raw=normalized, line_number=idx))
        if not lines:
            issues.append(LineValidationIssue(line_number=0, text="", reason="empty_output"))
        return lines, issues

    def _validate_lines(
        self,
        lines: Sequence[FormattedLine],
        max_chars: float,
    ) -> List[LineValidationIssue]:
        issues: List[LineValidationIssue] = []
        for line in lines:
            if not line.text:
                issues.append(LineValidationIssue(line_number=line.line_number, text=line.raw, reason="empty_text"))
            width = line.display_width
            if width > max_chars:
                issues.append(
                    LineValidationIssue(
                        line_number=line.line_number,
                        text=line.raw,
                        reason="exceeds_length",
                    )
                )
            if line.anchor_word is None:
                issues.append(
                    LineValidationIssue(
                        line_number=line.line_number,
                        text=line.raw,
                        reason="missing_anchor",
                    )
                )
        return issues


__all__ = [
    "BaseLLMProvider",
    "LLMFormatter",
    "FormatterError",
    "FormatterRequest",
    "FormatterResult",
    "FormatValidationError",
    "LineValidationIssue",
    "FormattedLine",
    "register_provider",
    "get_provider",
    "available_providers",
]

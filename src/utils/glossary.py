"""Glossary utilities (GUI editable dictionary for proofreading)."""
from __future__ import annotations

from typing import Iterable, List, Sequence


DEFAULT_GLOSSARY_TERMS: List[str] = [
    "菅義偉",
    "岸田文雄",
    "安倍晋三",
    "小池百合子",
    "立花孝志",
    "石破茂",
    "松野博一",
    "神谷宗幣",
    "小泉進次郎",
    "榛葉賀津也",
    "木原誠二",
    "高市早苗",
    "河合ゆうすけ",
    "大津力",
    "門田隆将",
    "北野裕子",
    "北村晴男",
    "公明党",
]


def normalize_glossary_terms(terms: Iterable[str] | None) -> List[str]:
    """Strip/normalize and de-duplicate glossary terms while keeping order."""
    if terms is None:
        return []
    seen: set[str] = set()
    normalized: List[str] = []
    for raw in terms:
        term = str(raw).strip()
        if not term:
            continue
        if term in seen:
            continue
        seen.add(term)
        normalized.append(term)
    return normalized


def parse_glossary_text(text: str) -> List[str]:
    """Parse 'one term per line' text into a normalized list."""
    return normalize_glossary_terms(text.splitlines())


def format_glossary_text(terms: Sequence[str]) -> str:
    """Format glossary terms as 'one term per line' text."""
    return "\n".join(normalize_glossary_terms(terms))


__all__ = [
    "DEFAULT_GLOSSARY_TERMS",
    "format_glossary_text",
    "normalize_glossary_terms",
    "parse_glossary_text",
]


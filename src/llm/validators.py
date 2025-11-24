"""Validators for detecting issues in Pass 2 output."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Sequence

from src.transcribe.base import WordTimestamp


@dataclass
class LineRange:
    """Represents a line range from Pass 2 output."""
    start_idx: int
    end_idx: int
    text: str


@dataclass
class ValidationIssue:
    """Represents a detected issue in the output."""
    type: Literal["short_particle_line", "split_quotation"]
    line_idx: int
    severity: Literal["high", "medium"]
    description: str
    suggested_action: str


def detect_issues(
    lines: Sequence[LineRange], 
    words: Sequence[WordTimestamp]
) -> List[ValidationIssue]:
    """
    Detect validation issues in Pass 2 output.
    
    Args:
        lines: Line ranges from Pass 2
        words: Word timestamps from Pass 1
        
    Returns:
        List of detected issues
    """
    issues = []
    
    # Rule 1: Short line (< 5 chars) ending with particle
    PARTICLES = ["を", "に", "で", "が", "は", "も", "から", "まで", "へ", "と"]
    for i, line in enumerate(lines):
        if len(line.text) < 5 and line.text and line.text[-1] in PARTICLES:
            issues.append(ValidationIssue(
                type="short_particle_line",
                line_idx=i,
                severity="high",
                description=f"行{i+1}は{len(line.text)}文字で助詞「{line.text[-1]}」で終わる",
                suggested_action="前行または次行と統合"
            ))
    
    # Rule 2: Split quotation expressions
    for i in range(len(lines) - 1):
        current = lines[i]
        next_line = lines[i + 1]
        
        if not current.text or not next_line.text:
            continue
            
        # Check if current line ends with "って" and next starts with "言" or "思"
        if current.text.endswith("って") and next_line.text.startswith(("言", "思")):
            issues.append(ValidationIssue(
                type="split_quotation",
                line_idx=i,
                severity="medium",
                description=f"行{i+1}-{i+2}で引用表現「〜って言う/思う」が分割されている",
                suggested_action="引用表現を統合"
            ))
    
    return issues


__all__ = ["LineRange", "ValidationIssue", "detect_issues"]

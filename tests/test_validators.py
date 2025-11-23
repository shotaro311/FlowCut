"""Tests for validators module."""
import pytest

from src.llm.validators import LineRange, ValidationIssue, detect_issues
from src.transcribe.base import WordTimestamp


def test_detect_short_particle_line():
    """Test detection of short lines ending with particles."""
    lines = [
        LineRange(0, 15, "1億円近い借金ができてるということ"),
        LineRange(16, 16, "を"),  # Problem: 1 char, ends with particle
        LineRange(17, 20, "カミングアウトされて"),
    ]
    words = []  # Not used in this detection
    
    issues = detect_issues(lines, words)
    
    assert len(issues) == 1
    assert issues[0].type == "short_particle_line"
    assert issues[0].line_idx == 1
    assert issues[0].severity == "high"


def test_detect_split_quotation():
    """Test detection of split quotation expressions."""
    lines = [
        LineRange(0, 10, "はい何するんだって"),
        LineRange(11, 15, "言うから"),  # Problem: quoted "って言う" split
    ]
    words = []
    
    issues = detect_issues(lines, words)
    
    assert len(issues) == 1
    assert issues[0].type == "split_quotation"
    assert issues[0].line_idx == 0
    assert issues[0].severity == "medium"


def test_no_issues():
    """Test when there are no issues."""
    lines = [
        LineRange(0, 10, "私は大学の12月ぐらい"),
        LineRange(11, 25, "政治家になろうと決めていて"),
    ]
    words = []
    
    issues = detect_issues(lines, words)
    
    assert len(issues) == 0


def test_short_line_without_particle():
    """Test that short lines without particles are not flagged."""
    lines = [
        LineRange(0, 5, "これ"),  # 2 chars, doesn't end with particle
        LineRange(6, 10, "テストです"),
    ]
    words = []
    
    issues = detect_issues(lines, words)
    
    assert len(issues) == 0


def test_particle_line_with_sufficient_length():
    """Test that lines with particles but sufficient length are not flagged."""
    lines = [
        LineRange(0, 10, "学校に行く"),  # 5 chars, ends with particle but >= 4
        LineRange(11, 15, "ことにした"),
    ]
    words = []
    
    issues = detect_issues(lines, words)
    
    assert len(issues) == 0

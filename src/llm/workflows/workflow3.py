"""workflow3: Whisper + Gemini ハイブリッド文字起こしワークフロー。

Whisperの単語タイムスタンプとGemini 3 Flash Previewの高精度テキストを
組み合わせて、精度の高い字幕データを生成する。
"""
from __future__ import annotations

from src.llm.workflows.common import DEFAULT_PASS4_PROMPT
from src.llm.workflows.definition import WorkflowDefinition
from src.llm.workflows.workflow1 import (
    build_pass1_prompt,
    build_pass2_prompt,
    build_pass3_prompt,
)


WORKFLOW = WorkflowDefinition(
    slug="workflow3",
    label="workflow3: Whisper+Geminiハイブリッド",
    description="Whisperの文字起こしをGemini 3 Flash音声認識で補正。長尺動画の認識漏れ対策に推奨。",
    wf_env_number=3,
    optimized_pass4=False,
    allow_pass3_range_change=True,
    pass3_enabled=True,
    pass1_fallback_enabled=False,
    two_call_enabled=False,
    # ハイブリッド処理を有効化
    hybrid_enabled=True,
    hybrid_thinking_level="medium",  # minimal, low, medium, high
    hybrid_similarity_threshold=0.8,  # この閾値未満の類似度でGeminiテキストを採用
    # LLMプロンプトはworkflow1と同じ
    pass1_prompt=build_pass1_prompt,
    pass2_prompt=build_pass2_prompt,
    pass3_prompt=build_pass3_prompt,
    pass4_prompt=DEFAULT_PASS4_PROMPT,
)

# Whisper + LLM ハイブリッド文字起こし提案書

**作成日**: 2024-12-24
**バージョン**: 1.1
**ステータス**: 提案段階

---

## 1. 背景と課題

### 1.1 現状の課題

現在のFlowCutでは、Whisper（ローカル実行）による文字起こしを行い、その結果をLLMで整形してSRTファイルを生成している。しかし、以下の課題が確認されている：

1. **Whisperの認識漏れ**: 長尺動画（30分以上）において、特定の区間で文字起こしが欠落する
2. **モデル性能の限界**: `large-v3`、`mlx-whisper`、`kotoba`など複数モデルを検証したが、根本的な改善には至らなかった
3. **低品質音声への脆弱性**: 背景ノイズ、マイク品質、話者の滑舌などにより精度が大幅に低下する

### 1.2 提案の目的

**Gemini**のマルチモーダル音声処理能力を活用し、Whisperとの**ハイブリッド方式**で文字起こし精度を向上させる。本提案では**Gemini 2.5 Flash**と**Gemini 3 Flash Preview**の両モデルを比較検討する。

---

## 2. Gemini音声処理機能の調査結果

### 2.1 モデル比較: Gemini 2.5 Flash vs Gemini 3 Flash Preview

| 項目 | Gemini 2.5 Flash | Gemini 3 Flash Preview |
|------|------------------|------------------------|
| **モデルID** | `gemini-2.5-flash-preview-native-audio-dialog` | `gemini-3-flash-preview` |
| **コンテキスト長** | 1M tokens | 1M tokens |
| **音声トークン消費** | 32 tokens/秒（1分 = 1,920 tokens） | 32 tokens/秒（1分 = 1,920 tokens） |
| **最大入力長** | 約15時間 | 約15時間 |
| **対応フォーマット** | WAV, MP3, AIFF, AAC, OGG, FLAC | WAV, MP3, AIFF, AAC, OGG, FLAC |
| **対応言語** | 24言語（日本語含む） | 24言語（日本語含む） |
| **入力コスト** | $0.50/1M tokens（テキスト）| $0.50/1M tokens（テキスト）|
| **音声入力コスト** | 〜$0.50/1M tokens | **$1.00/1M tokens** |
| **出力コスト** | $1.50/1M tokens | **$3.00/1M tokens** |
| **Thinking機能** | なし | **thinking_level対応**（minimal/low/medium/high） |
| **推論性能** | 高速・安定 | **2.5 Proを超える精度**、高速 |
| **コンテキストキャッシュ** | あり | あり（90%コスト削減可能） |
| **ステータス** | GA（一般提供） | Preview |

### 2.2 Gemini 3 Flash Previewの特徴

**強み**:
- **最新の推論性能**: Gemini 2.5 Proを多くのベンチマークで上回る
- **thinking_level制御**: 推論の深さを調整可能（minimal〜high）
  - `minimal`: 最速、シンプルな文字起こしに最適
  - `high`: 高精度、複雑な音声（専門用語、方言）に最適
- **エージェント向け最適化**: マルチターン会話、複雑なワークフローに強い
- **コンテキストキャッシュ**: 繰り返し処理で90%コスト削減

**弱み**:
- Previewステータス（仕様変更の可能性）
- 出力コストが2.5 Flashの2倍
- 音声入力コストも2倍

### 2.3 Gemini 2.5 Flashの特徴

**強み**:
- GA（一般提供）で安定性が高い
- コストパフォーマンスに優れる
- Native Audio機能による高品質な音声認識
- 30種類のHDボイス対応（TTS用途も可）

**弱み**:
- 3 Flash Previewと比較すると推論性能が劣る
- thinking機能なし

### 2.4 共通の特徴

**強み**:
- 高精度な音声認識（特に文脈理解に優れる）
- 話者識別（Speaker Diarization）対応
- セグメント単位のタイムスタンプ提供可能
- 感情検出、多言語切り替え対応

**弱み**:
- **単語レベルのタイムスタンプは非対応**（Whisperの`WordTimestamp`相当が得られない）
- リアルタイム処理非対応（バッチ処理のみ、Live APIは別）

### 2.5 モデル選択の推奨

| ユースケース | 推奨モデル | 理由 |
|-------------|-----------|------|
| **通常の動画字幕** | Gemini 2.5 Flash | コスト効率、安定性 |
| **長尺・複雑な音声** | Gemini 3 Flash Preview | 高精度推論、thinking_level=high |
| **専門用語が多い** | Gemini 3 Flash Preview | 文脈理解の深さ |
| **コスト重視** | Gemini 2.5 Flash | 出力コスト半額 |
| **プロダクション運用** | Gemini 2.5 Flash | GA安定性 |
| **実験・検証** | Gemini 3 Flash Preview | 最新性能の評価 |

### 2.6 参考情報

- [Audio understanding | Gemini API](https://ai.google.dev/gemini-api/docs/audio)
- [Gemini 2.5 Flash Native Audio upgrade](https://blog.google/products/gemini/gemini-audio-model-updates/)
- [Introducing Gemini 3 Flash](https://blog.google/products/gemini/gemini-3-flash/)
- [Build with Gemini 3 Flash](https://blog.google/technology/developers/build-with-gemini-3-flash/)
- [Gemini 3 Flash | Vertex AI Documentation](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/models/gemini/3-flash)
- [Gemini 3 Flash Preview - OpenRouter](https://openrouter.ai/google/gemini-3-flash-preview)

---

## 3. アーキテクチャ提案

### 3.1 提案A: シーケンシャル補正方式（推奨）

```
┌─────────────────────────────────────────────────────────────────┐
│                      Phase 0: 入力                               │
│  動画/音声ファイル → 音声抽出（必要に応じて）                      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Phase 1: 並列処理                             │
│                                                                   │
│  ┌──────────────────────┐    ┌──────────────────────┐           │
│  │  Whisper文字起こし   │    │  Gemini文字起こし     │           │
│  │  (WordTimestamp付)   │    │  (セグメント単位)     │           │
│  └──────────────────────┘    └──────────────────────┘           │
│            │                           │                         │
│            ▼                           ▼                         │
│    words: [                    gemini_text:                      │
│      {word: "私", start: 0.1, ...}    "私は大学の時に..."        │
│      {word: "は", start: 0.2, ...}                               │
│      ...                                                         │
│    ]                                                             │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                  Phase 2: テキストアライメント                    │
│                                                                   │
│  Whisperの単語列 ↔ Geminiのテキストを照合                         │
│                                                                   │
│  アルゴリズム:                                                    │
│  1. difflib.SequenceMatcher で類似度計算                         │
│  2. 低スコア区間（< 0.8）を「不一致区間」として検出               │
│  3. 不一致区間では Gemini のテキストを採用                        │
│  4. Whisper のタイムスタンプは維持                                │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│              Phase 3: 統合 WordTimestamp 生成                    │
│                                                                   │
│  merged_words: [                                                  │
│    {word: "私", start: 0.1, end: 0.2, source: "whisper"}         │
│    {word: "は", start: 0.2, end: 0.3, source: "gemini"}          │
│    ...                                                           │
│  ]                                                               │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                Phase 4-7: 既存LLMパイプライン                     │
│                                                                   │
│  Pass1: 校正 → Pass2: 行分割 → Pass3: 検証 → Pass4: 再分割       │
│                                                                   │
│                         ↓                                         │
│                   SRT ファイル出力                                │
└─────────────────────────────────────────────────────────────────┘
```

**メリット**:
- 既存のワークフロー（Pass1-4）をそのまま活用可能
- Whisperのタイムスタンプを維持しつつ、テキスト精度を向上
- 実装コストが比較的低い

**デメリット**:
- アライメント処理の精度がボトルネックになる可能性
- Whisperで完全に欠落した区間はタイムスタンプ推定が必要

---

### 3.2 提案B: ギャップ検出・補完方式

```
┌───────────────────────────────────────────────────────────────┐
│                    Phase 1: Whisper処理                        │
└───────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌───────────────────────────────────────────────────────────────┐
│                  Phase 2: ギャップ検出                          │
│                                                                 │
│  検出条件:                                                      │
│  - 連続する単語間の時間差が2秒以上                              │
│  - confidence < 0.5 の単語が連続                                │
│  - 文の途中で不自然に途切れている                               │
│                                                                 │
│  gap_segments: [(10.5, 15.2), (45.0, 48.5), ...]               │
└───────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌───────────────────────────────────────────────────────────────┐
│              Phase 3: ギャップ区間のGemini処理                   │
│                                                                 │
│  for each gap in gap_segments:                                  │
│      audio_segment = extract_audio(gap.start, gap.end)          │
│      gemini_text = gemini_transcribe(audio_segment)             │
│      gemini_words = estimate_word_timestamps(gemini_text, gap)  │
└───────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌───────────────────────────────────────────────────────────────┐
│                   Phase 4: 結果マージ                           │
│                                                                 │
│  Whisper結果 + Gemini補完 → 統合WordTimestamp                   │
└───────────────────────────────────────────────────────────────┘
```

**メリット**:
- Gemini APIコールを最小限に抑えられる（コスト効率）
- Whisperが正常に動作している区間はそのまま使用

**デメリット**:
- ギャップ検出のヒューリスティクスが複雑
- Gemini区間のタイムスタンプ推定精度に依存

---

### 3.3 提案C: LLMプライマリ方式（高精度志向）

```
┌───────────────────────────────────────────────────────────────┐
│                    Phase 1: 並列処理                           │
│                                                                 │
│  Gemini: 主たる文字起こし（高精度テキスト）                     │
│  Whisper: タイムスタンプ抽出専用                                │
└───────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌───────────────────────────────────────────────────────────────┐
│             Phase 2: Forced Alignment                           │
│                                                                 │
│  Geminiテキストを音声波形に対してアライメント                   │
│  (Montreal Forced Aligner または類似ツール)                     │
│                                                                 │
│  入力: gemini_text + audio_file                                 │
│  出力: word_timestamps[]                                        │
└───────────────────────────────────────────────────────────────┘
```

**メリット**:
- 最高精度のテキスト（Geminiの強み）をフル活用
- Whisperの認識ミスに依存しない

**デメリット**:
- Forced Alignerの導入が必要（追加依存）
- 処理時間が増加
- アライメント失敗時のフォールバック処理が必要

---

## 4. 推奨アプローチ: 提案A（シーケンシャル補正方式）

### 4.1 推奨理由

1. **既存資産の活用**: Pass1-4の既存ロジックをそのまま使用可能
2. **段階的導入**: まずは補正のみ、後から拡張可能
3. **コスト効率**: 必要な区間のみGemini補正を適用
4. **フォールバック安全性**: Gemini失敗時はWhisper結果で継続

### 4.2 実装構成

#### 4.2.1 新規ワークフロー `workflow4` の追加

```python
# src/llm/workflows/workflow4.py

WORKFLOW = WorkflowDefinition(
    slug="workflow4",
    label="workflow4: Whisper+Geminiハイブリッド",
    description="Whisperの文字起こしをGemini音声認識で補正。長尺動画の認識漏れ対策。",
    wf_env_number=4,
    optimized_pass4=False,
    allow_pass3_range_change=True,
    pass1_fallback_enabled=False,
    # Pass0として音声アライメント処理を追加
    pass0_enabled=True,
    pass0_prompt=None,  # プロンプト不要（音声処理）
    pass1_prompt=build_pass1_prompt,
    pass2_prompt=build_pass2_prompt,
    pass3_prompt=build_pass3_prompt,
    pass4_prompt=DEFAULT_PASS4_PROMPT,
)
```

#### 4.2.2 新規モジュール構成

```
src/
├── transcribe/
│   └── hybrid/
│       ├── __init__.py
│       ├── gemini_transcriber.py   # Gemini音声認識
│       ├── aligner.py              # テキストアライメント
│       └── merger.py               # 結果マージ
└── llm/
    └── workflows/
        └── workflow4.py            # ハイブリッドワークフロー
```

#### 4.2.3 GeminiTranscriber クラス設計

```python
# src/transcribe/hybrid/gemini_transcriber.py

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List, Literal, Optional
import base64

class GeminiModel(Enum):
    """利用可能なGeminiモデル"""
    FLASH_2_5 = "gemini-2.5-flash-preview-native-audio-dialog"
    FLASH_3_PREVIEW = "gemini-3-flash-preview"

class ThinkingLevel(Enum):
    """Gemini 3 Flash Preview用のthinking_level"""
    MINIMAL = "minimal"  # 最速、シンプルなタスク向け
    LOW = "low"          # 軽い推論
    MEDIUM = "medium"    # バランス型
    HIGH = "high"        # 最高精度、複雑なタスク向け

@dataclass
class GeminiSegment:
    text: str
    start_sec: float
    end_sec: float
    confidence: float = 1.0
    source_model: str = ""

class GeminiTranscriber:
    """Gemini APIを使用した音声文字起こし（2.5 Flash / 3 Flash Preview対応）"""

    # モデル別デフォルト設定
    MODEL_DEFAULTS = {
        GeminiModel.FLASH_2_5: {
            "supports_thinking": False,
            "default_temperature": 1.0,
        },
        GeminiModel.FLASH_3_PREVIEW: {
            "supports_thinking": True,
            "default_temperature": 1.0,
            "default_thinking_level": ThinkingLevel.MEDIUM,
        },
    }

    def __init__(
        self,
        api_key: str,
        model: GeminiModel = GeminiModel.FLASH_3_PREVIEW,
        language: str = "ja",
        thinking_level: ThinkingLevel | None = None,
        temperature: float | None = None,
    ):
        self.api_key = api_key
        self.model = model
        self.language = language

        # Gemini 3 Flash Preview用のthinking_level設定
        model_config = self.MODEL_DEFAULTS[model]
        if model_config["supports_thinking"]:
            self.thinking_level = thinking_level or model_config.get(
                "default_thinking_level", ThinkingLevel.MEDIUM
            )
        else:
            self.thinking_level = None

        self.temperature = temperature or model_config["default_temperature"]

    def transcribe(
        self,
        audio_path: Path,
        *,
        chunk_sec: float = 300.0,  # 5分単位で分割
    ) -> List[GeminiSegment]:
        """
        音声ファイルを文字起こしし、セグメント単位で返す。

        長尺ファイルは chunk_sec 単位に分割して処理する
        （Geminiのコンテキスト制限対策）
        """
        # 実装詳細は省略
        pass

    def _build_generation_config(self) -> dict:
        """モデルに応じたgenerationConfigを構築"""
        config = {
            "temperature": self.temperature,
            "response_mime_type": "application/json",
        }

        # Gemini 3 Flash Preview用のthinking_level設定
        if self.thinking_level is not None:
            config["thinking_level"] = self.thinking_level.value

        return config

    def _build_prompt(self) -> str:
        return f"""
あなたは音声文字起こしの専門家です。
以下の音声を正確に文字起こししてください。

言語: {self.language}

出力形式（JSON）:
{{
  "segments": [
    {{"text": "発話内容", "start": 0.0, "end": 2.5}},
    ...
  ]
}}

注意事項:
- 聞き取れない部分は [不明瞭] と記載
- 固有名詞は文脈から推測して記載
- フィラー（えー、あのー）は省略可
"""

    @classmethod
    def create_for_use_case(
        cls,
        api_key: str,
        use_case: Literal["standard", "complex", "cost_efficient"],
        language: str = "ja",
    ) -> "GeminiTranscriber":
        """ユースケースに応じた最適な設定でインスタンスを生成"""

        if use_case == "standard":
            # 通常の字幕作成: バランス型
            return cls(
                api_key=api_key,
                model=GeminiModel.FLASH_2_5,
                language=language,
            )
        elif use_case == "complex":
            # 専門用語・複雑な音声: 高精度
            return cls(
                api_key=api_key,
                model=GeminiModel.FLASH_3_PREVIEW,
                language=language,
                thinking_level=ThinkingLevel.HIGH,
            )
        elif use_case == "cost_efficient":
            # コスト重視: 最速設定
            return cls(
                api_key=api_key,
                model=GeminiModel.FLASH_3_PREVIEW,
                language=language,
                thinking_level=ThinkingLevel.MINIMAL,
            )
        else:
            raise ValueError(f"Unknown use_case: {use_case}")
```

#### 4.2.4 TextAligner クラス設計

```python
# src/transcribe/hybrid/aligner.py

from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import List, Tuple

@dataclass
class AlignmentResult:
    whisper_idx: int
    gemini_idx: int
    similarity: float
    action: str  # "keep_whisper" | "use_gemini" | "merge"

class TextAligner:
    """Whisper出力とGemini出力のアライメント"""

    def __init__(
        self,
        similarity_threshold: float = 0.8,
        window_size: int = 50,  # 比較ウィンドウサイズ（単語数）
    ):
        self.similarity_threshold = similarity_threshold
        self.window_size = window_size

    def align(
        self,
        whisper_words: List["WordTimestamp"],
        gemini_segments: List["GeminiSegment"],
    ) -> List[AlignmentResult]:
        """
        2つの文字起こし結果をアライメントする。

        アルゴリズム:
        1. 時間範囲で大まかにマッチング
        2. テキスト類似度で細かくアライメント
        3. 低類似度区間を検出
        """
        # 実装詳細は省略
        pass

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """2つのテキストの類似度を計算（0.0-1.0）"""
        return SequenceMatcher(None, text1, text2).ratio()
```

#### 4.2.5 WordMerger クラス設計

```python
# src/transcribe/hybrid/merger.py

from dataclasses import dataclass
from typing import List

@dataclass
class MergedWord:
    word: str
    start: float
    end: float
    confidence: float
    source: str  # "whisper" | "gemini" | "merged"

class WordMerger:
    """アライメント結果に基づいてWordTimestampをマージ"""

    def merge(
        self,
        whisper_words: List["WordTimestamp"],
        gemini_segments: List["GeminiSegment"],
        alignments: List["AlignmentResult"],
    ) -> List[MergedWord]:
        """
        アライメント結果に基づいて最終的なWordTimestamp列を生成。

        戦略:
        - similarity >= threshold: Whisperの単語とタイムスタンプを使用
        - similarity < threshold: Geminiのテキストを採用、タイムスタンプは
          Whisperから推定または比例配分
        """
        # 実装詳細は省略
        pass
```

---

## 5. 既存ワークフローへの統合

### 5.1 統合オプション

#### オプション1: 新規ワークフローとして追加（推奨）

```
workflow1: 標準（Pass1-4フル）
workflow2: 標準（分割並列）
workflow3: 標準（分割並列）← 現在workflow2のコピー
workflow4: Whisper+Geminiハイブリッド ← 新規追加
```

**メリット**: 既存ワークフローに影響なし、選択式で使い分け可能

#### オプション2: workflow2/3 の拡張オプションとして追加

```python
# GUIのオプションとして
enable_gemini_correction: bool = False  # チェックボックス
```

**メリット**: 既存UIへの追加が容易

### 5.2 GUI変更案

```
┌───────────────────────────────────────────────────────────┐
│  詳細設定                                                  │
├───────────────────────────────────────────────────────────┤
│  Whisperランナー: [openai ▼]                              │
│  ワークフロー:    [workflow4: ハイブリッド ▼]              │
│                                                            │
│  ┌─ Gemini音声補正設定 ─────────────────────────────────┐ │
│  │                                                        │ │
│  │  ☑ Gemini音声補正を有効化                              │ │
│  │                                                        │ │
│  │  音声認識モデル:                                       │ │
│  │  ○ Gemini 2.5 Flash（コスト効率・安定）               │ │
│  │  ● Gemini 3 Flash Preview（高精度・最新）             │ │
│  │                                                        │ │
│  │  ┌─ Gemini 3 Flash オプション ─────────────────────┐  │ │
│  │  │  thinking_level: [medium ▼]                      │  │ │
│  │  │    minimal: 最速（シンプルな音声向け）           │  │ │
│  │  │    low: 軽い推論                                 │  │ │
│  │  │    medium: バランス型（推奨）                    │  │ │
│  │  │    high: 最高精度（専門用語・複雑な音声向け）    │  │ │
│  │  └────────────────────────────────────────────────┘  │ │
│  │                                                        │ │
│  │  補正閾値: [0.8] （類似度、低いほど積極的に補正）      │ │
│  │                                                        │ │
│  └────────────────────────────────────────────────────────┘ │
│                                                            │
│  Pass1 モデル: [gemini-2.5-pro ▼]                         │
│  Pass2 モデル: [gemini-2.5-pro ▼]                         │
│  ...                                                       │
└───────────────────────────────────────────────────────────┘
```

### 5.3 CLI オプション追加案

```bash
# Gemini 2.5 Flashを使用（デフォルト）
python -m src.cli.main run audio.mp3 \
    --workflow workflow4 \
    --gemini-transcribe \
    --gemini-model gemini-2.5-flash

# Gemini 3 Flash Previewを使用（高精度モード）
python -m src.cli.main run audio.mp3 \
    --workflow workflow4 \
    --gemini-transcribe \
    --gemini-model gemini-3-flash-preview \
    --gemini-thinking-level high

# コスト最適化モード
python -m src.cli.main run audio.mp3 \
    --workflow workflow4 \
    --gemini-transcribe \
    --gemini-model gemini-3-flash-preview \
    --gemini-thinking-level minimal
```

---

## 6. コスト見積もり

### 6.1 Gemini 2.5 Flash APIコスト

| 音声長 | トークン数 | 音声入力コスト | 出力コスト（推定） | 合計 |
|--------|-----------|---------------|------------------|------|
| 5分 | 9,600 | $0.0048 | $0.014 | ~$0.019 |
| 30分 | 57,600 | $0.029 | $0.086 | ~$0.12 |
| 1時間 | 115,200 | $0.058 | $0.17 | ~$0.23 |
| 2時間 | 230,400 | $0.115 | $0.35 | ~$0.46 |

※ gemini-2.5-flash: 音声入力 $0.50/1M, 出力 $1.50/1M で計算

### 6.2 Gemini 3 Flash Preview APIコスト

| 音声長 | トークン数 | 音声入力コスト | 出力コスト（推定） | 合計 |
|--------|-----------|---------------|------------------|------|
| 5分 | 9,600 | $0.0096 | $0.029 | ~$0.039 |
| 30分 | 57,600 | $0.058 | $0.17 | ~$0.23 |
| 1時間 | 115,200 | $0.115 | $0.35 | ~$0.46 |
| 2時間 | 230,400 | $0.230 | $0.69 | ~$0.92 |

※ gemini-3-flash-preview: 音声入力 $1.00/1M, 出力 $3.00/1M で計算

### 6.3 モデル別コスト比較（30分動画）

| モデル | 音声文字起こしコスト | 既存Pass1-4コスト | 合計 | コスト増分 |
|--------|---------------------|------------------|------|----------|
| **既存workflow1**（Whisperのみ） | $0（ローカル） | ~$0.15 | ~$0.15 | - |
| **Gemini 2.5 Flash** | ~$0.12 | ~$0.15 | ~$0.27 | +80% |
| **Gemini 3 Flash Preview** | ~$0.23 | ~$0.15 | ~$0.38 | +153% |
| **Gemini 3 Flash (thinking=high)** | ~$0.23 | ~$0.23 | ~$0.46 | +207% |

### 6.4 コンテキストキャッシュによる最適化

Gemini 3 Flashはコンテキストキャッシュで90%コスト削減が可能：

| シナリオ | 通常コスト | キャッシュ適用後 |
|---------|-----------|----------------|
| 同一音声を複数回処理 | $0.23/回 | ~$0.023/回（2回目以降） |
| バッチ処理（10本） | $2.30 | ~$0.46（初回+キャッシュ） |

### 6.5 コスト・精度トレードオフの結論

| 優先事項 | 推奨構成 | 30分動画コスト |
|---------|---------|---------------|
| **コスト最小** | Whisperのみ + workflow1 | ~$0.15 |
| **バランス型** | Whisper + Gemini 2.5 Flash補正 | ~$0.27 |
| **精度優先** | Whisper + Gemini 3 Flash Preview補正 | ~$0.38 |
| **最高精度** | Whisper + Gemini 3 Flash (thinking=high) | ~$0.46 |

**結論**:
- **Gemini 2.5 Flash**: コスト増+80%で精度向上。日常的な字幕作成に推奨
- **Gemini 3 Flash Preview**: コスト増+153%で最高精度。専門性の高いコンテンツに推奨

---

## 7. 実装ロードマップ

### Phase 1: 基盤実装（1-2週間相当の作業量）

1. `GeminiTranscriber` クラスの実装
2. Gemini APIとの音声ファイルアップロード・処理
3. 単体テストの作成

### Phase 2: アライメント実装（1-2週間相当の作業量）

1. `TextAligner` クラスの実装
2. 類似度計算アルゴリズムの最適化
3. エッジケース（完全欠落、重複など）の処理

### Phase 3: 統合・テスト（1週間相当の作業量）

1. `WordMerger` クラスの実装
2. `workflow4` の定義・レジストリ登録
3. GUI統合（ワークフロー選択への追加）
4. エンドツーエンドテスト

### Phase 4: 最適化・拡張（継続的）

1. パフォーマンス最適化（並列処理）
2. コスト最適化（選択的Gemini適用）
3. ユーザーフィードバックに基づく改善

---

## 8. リスクと対策

| リスク | 影響度 | 対策 |
|--------|--------|------|
| Gemini APIの応答遅延 | 中 | タイムアウト設定、リトライ機構 |
| アライメント精度の低下 | 高 | 類似度閾値の調整、手動確認オプション |
| APIコスト超過 | 中 | 使用量監視、上限設定 |
| Gemini APIの仕様変更 | 低 | プロバイダー抽象化、バージョン固定 |

---

## 9. 結論

Whisper + Gemini のハイブリッド文字起こし方式は、長尺動画における認識精度の課題を解決する有効なアプローチである。

**推奨アクション**:

1. **提案A（シーケンシャル補正方式）** を採用
2. **workflow4** として新規ワークフローを追加
3. 既存ワークフロー（workflow1-3）には影響を与えない形で実装
4. GUIに「Gemini音声補正」オプションを追加
5. **Gemini 2.5 Flash**と**Gemini 3 Flash Preview**の両モデルを選択可能にする

**モデル選択の推奨**:

| 状況 | 推奨モデル | 設定 |
|------|-----------|------|
| **通常運用** | Gemini 2.5 Flash | - |
| **精度重視・複雑な音声** | Gemini 3 Flash Preview | thinking_level=high |
| **コスト重視だが最新モデル希望** | Gemini 3 Flash Preview | thinking_level=minimal |
| **プロダクション安定性重視** | Gemini 2.5 Flash | GA版のため |

**次のステップ**:

1. まずGemini 3 Flash Previewで実験・検証を実施
2. 精度とコストのバランスを評価
3. 本番運用ではGemini 2.5 Flash（GA）をデフォルトとし、必要に応じて3 Flash Previewを選択可能に

---

## 付録A: 代替案の比較表

| 項目 | 提案A: シーケンシャル | 提案B: ギャップ補完 | 提案C: LLMプライマリ |
|------|---------------------|--------------------|--------------------|
| 実装難易度 | 中 | 中 | 高 |
| 精度向上期待度 | 高 | 中 | 最高 |
| コスト効率 | 中 | 高 | 低 |
| 既存資産活用 | 高 | 高 | 低 |
| フォールバック | 容易 | 容易 | 複雑 |
| **総合評価** | **推奨** | 次点 | 将来検討 |

---

## 付録B: 関連ファイル

- `src/llm/workflows/registry.py` - ワークフローレジストリ
- `src/llm/workflows/definition.py` - WorkflowDefinition
- `src/llm/two_pass.py` - TwoPassFormatter
- `src/pipeline/poc.py` - パイプライン実行
- `src/transcribe/base.py` - トランスクリプションランナー基底クラス

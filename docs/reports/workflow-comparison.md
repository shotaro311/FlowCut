# FlowCut ワークフロー比較レポート

※ 2026-01-27 時点でワークフローは `workflow2` のみに統一され、`workflow1`/`workflow3` は削除済みです。本資料は削除前の比較として残しています。

**作成日**: 2025-12-27
**バージョン**: feat/workflow-registry

## 概要

FlowCutには3つのワークフローが実装されており、それぞれ異なる特性と用途を持っています。本レポートでは、各ワークフローの処理内容、特徴、使い分けについて詳細に解説します。

---

## ワークフロー一覧

| ワークフロー | 表示名 | 主な特徴 | 推奨用途 |
|------------|--------|---------|---------|
| workflow1 | 標準 | シングルスレッド処理 | 短時間動画（〜5分） |
| workflow2 | 標準（分割並列） | チャンク分割並列処理 | 長時間動画（5分以上） |
| workflow3 | Whisper+Geminiハイブリッド | Gemini音声認識で補正 | 認識精度重視・長尺動画 |

---

## 共通処理フロー

全ワークフローは以下の基本フローを共有しています：

```
Phase 1: Whisper文字起こし
  ↓
Phase 1.5: ハイブリッド処理（workflow3のみ）
  ↓
Phase 2: LLM処理
  - Pass1: テキスト校正
  - Pass2: 行分割
  - Pass3: 検証と修正
  - Pass4: セグメント生成
  ↓
Phase 3: SRT出力
```

---

## Workflow1: 標準

### 基本情報

- **slug**: `workflow1`
- **表示名**: workflow1: 標準
- **説明**: 従来の行分割 + 修正（Pass3で範囲変更あり）

### 処理の特徴

#### 1. Whisper文字起こし
- ユーザー選択のWhisperエンジン（MLX、OpenAI、ローカル等）を使用
- 単語レベルのタイムスタンプを取得
- 動画の場合は音声を自動抽出（WAV 16kHz）

#### 2. LLM処理（4パス）

**Pass1: テキスト校正**
- 目的: 誤字脱字の修正、固有名詞の統一
- 許可操作:
  - `replace`: 誤変換を正しい表記に置換
  - `delete`: ノイズ（フィラー・重複）を削除
- 禁止操作:
  - `insert`: 音声にない単語の追加
  - 並び替え、要約、意訳
- Glossary参照: ユーザー定義の用語集を最優先

**Pass2: 行分割**
- 1行の最大文字数: 17文字（設定可能: 12〜20）
- 自然な分割ルール:
  1. 行頭に助詞・補助表現を置かない
  2. 接続表現・接続詞で分割しない
  3. 1〜4文字の極端に短い行を作らない（5文字以上必須）
  4. 活用語尾・引用表現を保持
- 思考ワークフロー:
  1. チャンク分解
  2. 行の構築と決定
  3. クリーニング（行末句読点削除）

**Pass3: 検証と修正**
- 検出された問題を修正:
  - 1-4文字の極端に短い行
  - 引用表現の分割
  - 語の途中での切断
  - 時間幅の超過（10秒以上）
- 範囲変更可能（`allow_pass3_range_change=True`）
- 最小限の修正を実施

**Pass4: セグメント生成**
- 行のインデックス範囲から実際のテキストを抽出
- タイムスタンプを付与
- SRT形式に変換

#### 3. 処理方式
- **シングルスレッド**: 1つのLLMリクエストで全体を処理
- **適性**: 短時間動画（〜5分程度）

### 利点と制限

**利点**:
- シンプルで理解しやすい
- 全体の文脈を保持
- 安定した出力品質

**制限**:
- 長時間動画では処理時間が増加
- LLMのコンテキスト長制限の影響を受けやすい

---

## Workflow2: 標準（分割並列）

### 基本情報

- **slug**: `workflow2`
- **表示名**: workflow2: 標準（分割並列）
- **説明**: workflow1と同等のプロンプト/処理（Pass3で範囲変更あり）。長尺は約5分で分割し並列処理。

### 処理の特徴

#### workflow1との違い

**同じ部分**:
- Pass1〜Pass4のプロンプトは完全に同一
- LLM処理ロジックも同一
- 品質・精度はworkflow1と同等

**異なる部分**:
- **チャンク分割**: 約5分（300秒）単位で動画を分割
- **並列処理**: 各チャンクを並列にLLM処理
- **環境変数対応**: `wf_env_number=2` で環境変数による設定上書き可能

#### チャンク分割処理

```python
# 疑似コード
chunk_duration = 300秒  # 約5分
chunks = split_words_into_time_chunks(words, chunk_duration)

# 各チャンクを並列処理
for chunk in chunks:
    pass1_result = llm.process(chunk, pass1_prompt)
    pass2_result = llm.process(chunk, pass2_prompt)
    pass3_result = llm.process(chunk, pass3_prompt)
    pass4_result = llm.process(chunk, pass4_prompt)

# 結果を統合
final_segments = merge_all_chunks(results)
```

### 利点と制限

**利点**:
- 長時間動画でも高速処理
- 各チャンクが独立してエラー耐性向上
- 並列実行でスループット向上

**制限**:
- チャンク境界で文脈が途切れる可能性
- 並列処理のオーバーヘッド
- workflow1より若干複雑

### 推奨使用シーン

- 5分以上の長時間動画
- 処理速度を優先したい場合
- 安定したネットワーク環境

---

## Workflow3: Whisper+Geminiハイブリッド

### 基本情報

- **slug**: `workflow3`
- **表示名**: workflow3: Whisper+Geminiハイブリッド
- **説明**: Whisperの文字起こしをGemini 3 Flash音声認識で補正。長尺動画の認識漏れ対策に推奨。

### 処理の特徴

#### ハイブリッド処理（Phase 1.5）

workflow3の最大の特徴は、WhisperとGeminiの2つの音声認識エンジンを組み合わせることです。

**処理フロー**:

```
1. Whisper文字起こし
   ↓
2. Gemini音声認識（同じ音声ファイル）
   ↓
3. アライメント（2つの結果を時間軸で対応付け）
   ↓
4. テキスト類似度計算
   ↓
5. マージ（低類似度区間はGeminiを採用）
   ↓
6. LLM処理（Pass1〜4）
```

#### ハイブリッド処理の詳細

**1. Gemini音声認識**
- モデル: `Gemini 3 Flash Preview`
- Thinking Level: `medium`（minimal, low, medium, high）
- 対応フォーマット: WAV, MP3, AIFF, AAC, OGG, FLAC, M4A
- セグメント単位のタイムスタンプ付き

**2. アライメント処理**

各Geminiセグメントについて：
1. 時間範囲に対応するWhisper単語を特定（±1秒の許容誤差）
2. テキスト類似度を計算（SequenceMatcher）
3. アクションを決定:
   - 類似度 ≥ 0.8: Whisperを維持（`keep_whisper`）
   - 類似度 < 0.8: Geminiを採用（`use_gemini`）
   - Whisper単語なし: Geminiを挿入（`use_gemini`）

**3. マージ処理**

- `keep_whisper`: Whisperの単語とタイムスタンプをそのまま使用
- `use_gemini`:
  - Geminiのテキストを日本語単語に分割
  - タイムスタンプを時間範囲内で比例配分
  - Confidence: 0.8（Whisperより若干低め）

#### Whisper欠落区間の補完

**問題**: MLX Whisperでは、稀に20〜30秒程度の区間で単語が完全に欠落することがある

**解決**: workflow3のハイブリッド処理

```
例: 648.94秒〜673.62秒（約25秒）のギャップ

Whisper:
  648.94秒: "こ"
  673.62秒: "これは"  ← 25秒のギャップ！

Gemini:
  652.00-666.90秒: "全部党中央でグリップするというのが..."

アライメント結果:
  - Whisper単語なし → action="use_gemini"

マージ結果:
  652.00秒: "全部党中央でで"
  652.97秒: "グリップするとと"
  654.07秒: "いうののがが"
  ...（19個の単語が生成される）
```

#### LLM処理の調整

**Pass2でのエラーハンドリング**:
- ハイブリッド処理で単語数が変化するため、範囲チェックでエラーが発生する可能性
- workflow3では、範囲エラーを**警告に変更**してPass3で修正可能に
- workflow2と同様の動作

**Pass3での修正**:
- 範囲の不整合を自動修正
- 単語数の変化に対応

### ハイブリッド処理のパラメータ

| パラメータ | デフォルト値 | 説明 |
|----------|------------|------|
| `hybrid_enabled` | `True` | ハイブリッド処理の有効/無効 |
| `hybrid_thinking_level` | `"medium"` | Geminiの思考レベル |
| `hybrid_similarity_threshold` | `0.8` | この閾値未満でGeminiを採用 |
| `time_tolerance_sec` | `1.0` | 時間範囲マッチングの許容誤差 |

### 必須設定

**環境変数**:
```bash
export GOOGLE_API_KEY="your-api-key"
```

- API KEY未設定の場合、ハイブリッド処理はスキップされWhisperのみの結果を使用
- GUIに「GOOGLE_API_KEY未設定（Whisperのみ）」と表示

### ログ出力

ハイブリッド処理のログは `output/{動画名}_{timestamp}/logs/hybrid_logs/` に保存:

**1. Gemini文字起こし結果**
```json
{
  "type": "gemini_transcription",
  "gemini": {
    "model": "gemini-3-flash-preview",
    "thinking_level": "medium",
    "text": "...",
    "segments": [...]
  },
  "whisper_comparison": {
    "text": "...",
    "word_count": 4126
  }
}
```

**2. ハイブリッド統合結果**
```json
{
  "type": "hybrid_merged",
  "summary": {
    "whisper_word_count": 4126,
    "gemini_segment_count": 76,
    "merged_word_count": 2111,
    "blocks_using_gemini": 74,
    "blocks_using_whisper": 3,
    "average_similarity": 0.2213,
    "similarity_threshold": 0.8
  },
  "alignment_blocks": [...],
  "merged_words": [...]
}
```

### 利点と制限

**利点**:
- **認識精度の向上**: WhisperとGeminiの長所を組み合わせ
- **欠落区間の補完**: Whisperが認識できなかった区間をGeminiで補完
- **長尺動画対応**: 認識漏れが発生しやすい長時間動画に最適
- **詳細なログ**: どの区間でGeminiを採用したか確認可能

**制限**:
- **処理時間**: Gemini APIの呼び出しで時間が増加（約1〜2分）
- **API費用**: Gemini APIの利用料金が発生
- **API KEY必須**: GOOGLE_API_KEY環境変数の設定が必要
- **単語数変化**: マージ後の単語数が変わるため、デバッグが複雑

### 推奨使用シーン

- 長時間動画（10分以上）
- 認識精度を最優先する場合
- Whisperで認識漏れが発生している場合
- 複数話者・専門用語が多い動画

---

## ワークフロー比較表

### 処理ステップの比較

| ステップ | workflow1 | workflow2 | workflow3 |
|---------|-----------|-----------|-----------|
| Whisper文字起こし | ✓ | ✓ | ✓ |
| ハイブリッド処理 | ✗ | ✗ | ✓（Gemini） |
| LLM Pass1（校正） | ✓ | ✓ | ✓ |
| LLM Pass2（行分割） | ✓ | ✓ | ✓ |
| LLM Pass3（検証） | ✓ | ✓ | ✓ |
| LLM Pass4（セグメント生成） | ✓ | ✓ | ✓ |
| チャンク並列処理 | ✗ | ✓ | ✗ |

### 設定パラメータの比較

| パラメータ | workflow1 | workflow2 | workflow3 |
|-----------|-----------|-----------|-----------|
| `wf_env_number` | `None` | `2` | `3` |
| `optimized_pass4` | `False` | `False` | `False` |
| `allow_pass3_range_change` | `True` | `True` | `True` |
| `pass3_enabled` | `True` | `True` | `True` |
| `hybrid_enabled` | `False` | `False` | `True` |
| `hybrid_thinking_level` | - | - | `"medium"` |
| `hybrid_similarity_threshold` | - | - | `0.8` |

### パフォーマンス比較（目安）

10分動画の場合：

| ワークフロー | Whisper | ハイブリッド | LLM処理 | 合計 |
|------------|---------|-------------|---------|------|
| workflow1 | 2分 | - | 3分 | 5分 |
| workflow2 | 2分 | - | 2分（並列） | 4分 |
| workflow3 | 2分 | 1.5分 | 3分 | 6.5分 |

※ MLX Whisperを使用、ネットワーク速度により変動

### 品質比較

| 指標 | workflow1 | workflow2 | workflow3 |
|-----|-----------|-----------|-----------|
| 認識精度 | Whisperに依存 | Whisperに依存 | 高（Geminiで補正） |
| 欠落対応 | なし | なし | Geminiで補完 |
| 文脈保持 | 良好 | やや低い（チャンク分割） | 良好 |
| 安定性 | 高 | 中 | 中（API依存） |

---

## 使い分けガイド

### フローチャート

```
動画時間は？
  ├─ 5分未満
  │   └─ 認識精度を重視？
  │       ├─ はい → workflow3
  │       └─ いいえ → workflow1
  │
  └─ 5分以上
      └─ 認識漏れが気になる？
          ├─ はい → workflow3
          └─ いいえ → workflow2
```

### シーン別推奨

**短時間・高品質重視**
- **推奨**: workflow3
- 理由: Geminiによる補正で最高品質

**短時間・シンプル**
- **推奨**: workflow1
- 理由: 最もシンプルで安定

**長時間・速度重視**
- **推奨**: workflow2
- 理由: 並列処理で高速

**長時間・品質重視**
- **推奨**: workflow3
- 理由: 認識漏れを補完

**専門用語・複数話者**
- **推奨**: workflow3
- 理由: Geminiの高精度認識

---

## トラブルシューティング

### workflow3で「GOOGLE_API_KEY未設定」と表示される

**原因**: 環境変数が設定されていない

**解決**:
```bash
export GOOGLE_API_KEY="your-api-key"
```

### workflow3で「行分割結果の範囲（from/to）が不正です」エラー

**原因**: ハイブリッド処理で単語数が変化

**解決**: 最新版にアップデート（修正済み: commit `64ba8c6f`）

### workflow2でチャンク境界が不自然

**原因**: 5分単位の分割で文脈が途切れる

**解決**: workflow1またはworkflow3を使用

### Gemini APIエラー

**原因**: APIレート制限、無効なAPI KEY、音声フォーマット非対応

**解決**:
1. API KEYを確認
2. 音声ファイルがWAV/MP3等の対応フォーマットか確認
3. APIクォータを確認

---

## 今後の拡張予定

### workflow4（検討中）
- 2コールモード（Pass2-4を統合）
- さらに高速な処理

### ハイブリッド処理の改善
- 複数のGeminiモデル対応
- カスタム類似度閾値設定
- リアルタイム進捗表示

---

## まとめ

| ワークフロー | 一言で言うと | おすすめ度 |
|------------|------------|-----------|
| workflow1 | シンプルで安定 | ★★★☆☆ |
| workflow2 | 長時間動画向け | ★★★★☆ |
| workflow3 | 最高品質 | ★★★★★ |

**初めて使う方**: workflow1から始めて、必要に応じてworkflow2/3を試す
**品質重視**: workflow3一択
**速度重視**: workflow2を推奨
**バランス重視**: 5分以下ならworkflow1、5分以上ならworkflow2

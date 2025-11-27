要件定義書
**Target Device:** Mac M3 (16GB Memory)
**User:** 友人（非エンジニア）

## 1. システム概要
音声ファイルをドラッグ＆ドロップするだけで、**「1行17文字以内」**かつ**「文脈として自然な区切り」**で構成されたSRT（字幕ファイル）を出力する自動化ツール。
従来ツールのような機械的な分割を排除し、プロの編集者が手動で切ったような「意味のまとまり」を再現することを最重要視する。

## 2. 技術スタック

### 音声認識（2モデルで比較検証）
プロトタイプでは以下2つのモデルを実装。**デフォルトは mlx-whisper large-v3（MLX版）** とする（Plan方針）。

*   **モデル1: mlx-whisper large-v3（汎用×MLX最適化｜デフォルト）**
    *   リポジトリ: `mlx-community/whisper-large-v3-mlx`
    *   Apple Silicon（M3）でMetal GPU加速
    *   word-levelタイムスタンプ対応
    *   **期待値**: 最高精度（専門用語・複雑な内容）

*   **モデル2: OpenAI Whisper large-v3（公式実装）**
    *   リポジトリ: `openai/whisper`
    *   MPS（Metal Performance Shaders）対応
    *   word-levelタイムスタンプ対応
    *   **期待値**: 安定性・互換性重視

### 文章整形（LLM API）
以下のプロバイダーから選択可能。**Plan推奨デフォルトは Google (gemini-3-pro-preview = Gemini 3.0 Pro)**、ただし **Pass3 のみ gemini-2.5-flash** でコスト最適化。  
環境変数 `LLM_PASS1_MODEL` / `LLM_PASS2_MODEL` / `LLM_PASS3_MODEL` で各パスのモデルを自由に上書き可能（例: `gpt-5.1`, `claude-sonnet-4-20250514`）。プロバイダー指定は `--llm` で行い、モデル名は文字列そのまま渡せる。
※ CLIでは `--llm` を明示指定しない限り整形とSRT生成は実行されず、文字起こしJSONのみ保存される。  
※ **三段階LLMワークフロー（Three-Pass）** を採用し、全文をLLMに渡して意味的改行および最終検証を実施。
  - **Pass 1**: テキストクリーニング（削除・置換のみ）
  - **Pass 2**: 17文字行分割（自然な改行位置を決定）
  - **Pass 3**: 品質検証（Python検出器 + LLM修正、問題なし時はスキップ）
  - **ブロック分割は廃止**: 旧BlockSplitterは撤去し、全文を一括で処理する。長尺対応は再開機構（`--resume`）とFuzzyアライメントで担保。

*   **Google** (gemini-3-pro-preview / gemini-2.5-flash) ←推奨
    - **Pass 1/2 デフォルト**: gemini-3-pro-preview（高精度）
    - **Pass 3 デフォルト**: gemini-2.5-flash（コスト削減）
*   **OpenAI** (gpt-5.1, gpt-5-mini など)
*   **Anthropic** (claude-sonnet-4-20250514, claude-haiku-4-5 など) ※MVP完了後に比較テスト予定

**設計方針:**
*   各プロバイダーのAPIキーとモデル名を`.env`ファイルで管理
*   CLIオプション `--llm {openai|google|anthropic}` で実行時に切り替え（未指定なら整形スキップ）
*   非エンジニアでもモデル更新可能（コード変更不要）
*   原文の単語を極力変更せず、アライメント精度を向上

**.env設定例:**
```env
# OpenAI
OPENAI_API_KEY=sk-proj-xxxxxxxxxxxxx
OPENAI_MODEL=gpt-5-mini
# Google Gemini
GOOGLE_API_KEY=AIzaSyxxxxxxxxxxxxxxxxx
GOOGLE_MODEL=gemini-3-pro-preview
# Two-pass モデル上書き（省略可）
LLM_PASS1_MODEL=gemini-3-pro-preview
LLM_PASS2_MODEL=gemini-3-pro-preview
LLM_PASS3_MODEL=gemini-2.5-flash
LLM_PASS4_MODEL=gemini-2.5-flash  # 省略時は LLM_PASS3_MODEL を再利用
# Anthropic Claude
ANTHROPIC_API_KEY=sk-ant-xxxxxxxxxxxxx
ANTHROPIC_MODEL=claude-sonnet-4-20250514
```

### 言語・環境
*   **Python 3.10-3.12**（3.13は非対応）
*   **開発環境:** Python仮想環境（venv）
*   **Docker不使用の理由:**
    *   MLXはApple SiliconのMetal GPUに直接アクセスする必要がある
    *   DockerはMetal GPUサポートなし（CPU動作になり性能が大幅低下）
    *   ネイティブ環境が最も高速で開発効率も高い

### プラットフォーム別のローカルWhisperモデル前提

*   **Mac（友人に配布する `.app` の想定環境）**
    *   文字起こしのデフォルトは **MLX Whisper Large-v3**。  
    *   開発時は `pip install mlx-whisper` で依存を導入し、配布時の `.app` にはこのランタイムを同梱する想定。  
    *   友人側のMacでは、`.app` を展開して開くだけで MLX Whisper が利用できる（別途 `pip install` は不要）。
*   **Windows（将来対応を想定：ローカル実行）**
    *   文字起こしの基盤は **OpenAI Whisper Large-v3 のローカル実行版** を前提とする。  
    *   開発・実行環境では `openai-whisper` など公式Whisper実装を事前インストール、もしくは `.exe` パッケージに同梱する方針。  
    *   Windows向けの配布形態（`.exe` やインストーラ）は別PLANで詳細設計する（現時点ではTODO）。

### 依存関係管理
*   **フェーズ1（推奨）:** `pip + venv + requirements.txt`
    *   シンプルで初心者向け
    *   Python標準ツール（追加インストール不要）
    *   このプロジェクト規模では十分
*   **将来的な選択肢:** uv または Poetry
    *   プロジェクトが大規模化した際に検討
    *   バージョン競合の自動解決が必要になったら移行
*   **同一環境に2モデルを共存:**
    *   mlx-whisper（MLX Whisper Large-v3）
    *   openai-whisper（公式 Whisper large-v3 ローカル/API 実装）
*   競合が発生した場合のみ環境分離を検討

### GUI（フェーズ4で実装）
*   **フェーズ1:** GUIなし（Pythonスクリプトのみ）
*   **フェーズ4:** 簡易GUI（Tkinter または Flet）を想定（Planに合わせて後ろ倒し）
*   **将来的な選択肢:** Tauri + React（モダンなUI）

## 3. 処理フロー（内部ロジック）

### Step 1: 精密文字起こし (Local)
*   mlx-whisper（デフォルト）または kotoba / OpenAI Whisper を使用し、音声からテキストデータを作成。
*   **必須:** `word_timestamps=True`を指定して、単語ごとの開始・終了時刻を取得。
    *   例: `{'word': '設定を', 'start': 10.5, 'end': 11.2}, ...`
*   **出力データ:**
    *   全文テキスト
    *   単語レベルのタイムスタンプリスト（JSON形式で保持）

### Step 2: 文脈理解と整形 (Cloud API)
*   Whisper全文を選択したLLM APIへ送信し、改行・整形を行う。
*   **採用方針：二段階＋検証ワークフロー**（TwoPassFormatter + Pass3/Pass4）: `docs/specs/llm_two_pass_workflow.md` 参照  
    - **パス1 (Text Cleaning):** 置換・削除のみ（挿入禁止）で誤字・フィラー除去。単語インデックス指定のJSONを返す。  
    - **パス2 (Line Splitting):** 17文字分割のみを決める。行範囲をインデックスで返す。  
    - **時間計算:** ローカルで行い、Whisperのwordタイムスタンプを維持する（尺伸び防止）。  
    - **タイムスタンプ補正 (Clamping):** LLMのインデックス指定ミスによる時間の巻き戻りを検知し、強制的に時系列順に補正する。  
    - **最小行長/Pass3:** 行は原則5〜17文字。問題がなくても Pass3 で最終確認し、短行は統合する。  
    - **Pass4（長さ違反行のみ再LLM）:** Pass3後に5文字未満/17文字超の行だけを再度LLMにかけ直す。出力が空/不正の場合は元行を維持し、Pass4 の段階で出力された `lines` をそのまま採用する（Pass4 後にローカルでの強制再分割は行わない）。  
    - **末尾カバレッジ保証（プロバイダ差異の吸収）:** 一部プロバイダ（特に OpenAI）では、Pass2/Pass3 の `lines` が先頭側に偏り、末尾の単語に対応する行が生成されないケースがある。この場合は TwoPassFormatter 内のフォールバック（`_ensure_trailing_coverage`）により、未カバーの単語から簡易な行を自動生成し、**常に文字起こし全文がSRTに反映される**ようにする。
*   **プロンプトの役割（two-pass）:**
    1.  パス1: 置換/削除のみを operations 配列(JSON)で返す。挿入禁止・順序を変えない。
    2.  パス2: 17文字以内の自然な行分割を `{"lines":[{"from":0,"to":10,"text":"..."}]}` 形式で返す。
    3.  **出力形式:** `[WORD: ]` タグは使用せず、行範囲（from/to）とテキストのみを返す。
    4.  （オプションON時のみ）語尾の微調整・リライト。

#### API出力例（パス2）
```json
{"lines":[{"from":0,"to":5,"text":"設定を開いてくださいね"}]}
```

#### エラーハンドリング
*   **リトライ戦略:** API失敗時は指数バックオフ（1s → 3s → 5s）で最大3回リトライ
*   **失敗時の動作:**
    *   処理済みブロックを `temp/progress_{timestamp}.json` に保存
    *   エラー終了時に具体的な再開手順を表示
    *   例: `python -m src.cli.main run input.wav --resume temp/progress_20250120_103000.json`
*   **ログ記録:**
    *   `logs/processing.log` に処理状況を記録
    *   LLM生応答は `logs/llm_raw/` に **1 run（音声×モデル）につき1ファイル** として集約保存（Pass1〜Pass4の生JSONをパスごとセクションにまとめる）
    *   LLM使用量・時間と概算コストは `logs/metrics/{音声ファイル名}_{日付}_{run_id}_metrics.json` に保存し、以下をJSONで持つ:
        *   全体の経過時間（人間が読みやすい形式の `total_elapsed_time`。例: `8m 22.35s`）
        *   transcribe / two-pass 各工程の所要時間（`stage_timings_time`。例: `"transcribe_sec": "1m 28.12s"`）
        *   入力音声の長さ（wordタイムスタンプの先頭〜末尾を元にした `audio_duration_time`）
        *   Pass1〜Pass3 のトークン数・処理時間、および実際に使用した `provider` / `model`
        *   Pass4 のトークン数・処理時間（複数回呼び出し分を合計）、および `provider` / `model`
        *   プロバイダー/モデルごとの 1M トークン単価（`config/llm_pricing.json`）を用いて算出した概算コスト（Pass単位の `cost_input_usd` / `cost_output_usd` / `cost_total_usd` と、run全体の `run_total_cost_usd`）

### Step 3: タイムスタンプ再計算 (Local)

#### 3-1. 行範囲（from/to）からSRTを生成
Pass2 の `lines` で示されたインデックス範囲を word-level タイムスタンプへ直接マップする。

1. **JSONの取り出し:** `{"lines":[{"from":0,"to":5,"text":"..."}]}` をパース。
2. **17文字検証:** 行長の制約は主に Pass3/Pass4 の LLM で担保し、SRT生成段階ではローカルでの再分割は行わない（LLM が返した `text` をそのまま使用）。  
3. **タイムスタンプ算出:**
   - 開始: `words[from].start`
   - 終了: `words[to].end`
   - 巻き戻り防止: 前行の終了より小さい場合は前行終了にクランプ。

#### 3-2. タイムスタンプ補正 (Clamping)
パス2で指定されたインデックスが原因で時間が逆行する場合、開始時刻を前行終了に合わせて補正する。

1.  **巻き戻り検知:** `current_start < last_end` の場合
2.  **クランプ処理:** `current_start = last_end` に強制補正
3.  **整合性維持:** `current_end < current_start` となった場合は `current_end = current_start + 0.1` で微調整

### Step 4: SRT出力
*   決定したタイムコードとテキストをSRT形式でファイルに書き出す。
*   セグメント間にタイムコードの空白（ギャップ）がある場合は、前の行の終了時刻を次の行の開始まで延長し、字幕表示が途切れないようにする。
*   **SRTフォーマット:**
    ```
    1
    00:00:10,500 --> 00:00:11,200
    設定を開いて
    2
    00:00:11,200 --> 00:00:12,000
    くださいね
    ```
## 4. UI/UX 要件
### フェーズ1: コマンドライン版（優先実装）
#### 基本実行（Typer CLI）
```bash
python -m src.cli.main run <音声ファイル> [オプション]
```

#### 主なオプション（実装に合わせて更新）
- `--models kotoba,mlx` : 使用するランナーをカンマ区切り指定。未指定なら **MLX large-v3のみ** 実行。
- `--llm {google|openai|anthropic}` : LLM整形プロバイダー。**未指定なら整形・SRT出力を行わず、文字起こしJSONのみ保存**（Plan推奨は google）。
- `--rewrite / --no-rewrite` : 語尾リライトを有効化（デフォルトNoneでプロバイダー既定に従う）。
- （二段階LLMはデフォルトで常時有効。切替オプションなし）
- `--language ja` / `--chunk-size 30` などは各ランナーへ伝播。
- `--resume temp/progress_xxx.json` : 途中から再開。
- `--simulate/--no-simulate` : ランナーのシミュレーション切替（デフォルトON）。
* `--subtitle-dir ./my_output` : SRT字幕を書き出すディレクトリを指定（未指定時は `output/`）。

#### 出力パス
- 音声×モデル×実行時刻ごとに `temp/poc_samples/{run_id}.json` を保存（内部的に **最大5件まで** を保持し、古いJSONから自動削除してディスク肥大化を防ぐ）。
- LLM整形を実行した場合のみ `{subtitle_dir}/{run_id}.srt` を自動命名で保存（`subtitle_dir` のデフォルトは `output/`、`--subtitle-dir` で変更可能）。

#### 実行例
```bash
# MLXのみで文字起こし（整形なし）
python -m src.cli.main run samples/sample_audio.m4a

# MLXとOpenAI Whisperを実行し、Geminiで整形＋SRT出力
python -m src.cli.main run samples/sample_audio.m4a \
  --models mlx,openai \
  --llm google

# リライト有効＋Anthropic
python -m src.cli.main run samples/sample_audio.m4a --llm anthropic --rewrite
```

#### 進捗表示
*   コンソールに「音声解析中...」「AI思考中 (ブロック 5/20)...」「ファイル生成中」を表示
*   tqdmなどでプログレスバー表示

#### 出力
*   LLM整形を実行した場合のみ `output/{音声名}_{モデル}_{日時}.srt` を自動保存（--outputオプションなし）
*   文字起こしJSONは `temp/poc_samples/{run_id}.json` に保存されるが、最大5件までを保持し、古いファイルから自動削除される

### フェーズ4: GUIアプリ（将来実装｜Plan準拠）
友人にとっての使いやすさを考慮し、設定項目は最小限にする。

*   **メイン画面:**
    *   ファイル選択エリア（ドラッグ＆ドロップ対応）
    *   **実行ボタン**
    *   **進捗バー:** 「音声解析中...」「AI思考中...」「ファイル生成中」などのステータス表示。
*   **オプション設定（トグルスイッチ等）:**
    *   [ ] **モデルプリセット選択**（例: `default` / `low_cost` / `high_quality`）  
        - 内部的には `config/llm_profiles.json` に定義した LLMプロファイルを選択し、Pass1〜4 のモデルをまとめて変更する。  
        - 友人はプリセット名だけを意識すればよい想定。
    *   [ ] **詳細モード（上級者向け）**  
        - GUI上の折りたたみセクションで、Pass1〜Pass4 のモデル名を **プルダウン（Combobox）** から個別に選択できる（候補は `config/llm_profiles.json` などプロファイル定義から自動生成）。  
        - CLIの `--llm-profile` と `LLM_PASS*_MODEL` 相当の設定をGUIから調整できるイメージ。
    *   [ ] **語尾調整・リライトを行う**（デフォルトOFF：原文維持＋フィラー削除のみ）
    *   [ ] **高精度モード**（large-v3モデル使用。OFFの場合はmediumモデルで高速化）
*   **出力:**
    *   デフォルトでは `output/` ディレクトリに `filename_model_timestamp.srt` を保存（CLIと共通）。
    *   GUI からは「保存先フォルダを選択」ボタンで任意のディレクトリを指定できる。
    *   完了時に通知を表示し、GUI下部に「総トークン数」「概算APIコスト（USD、小数点第3位まで）」「総処理時間（X分Y秒）」を表示する。

#### 将来のWebアプリ版について（メモ）
- 本要件定義のMVPでは「ローカル実行できるGUI（Tkinter / Flet想定）」を優先し、ブラウザ上で動くWebアプリ版は**別フェーズ**で検討する。  
- そのため、音声→文字起こし→整形→SRT出力のコア処理は `src/pipeline/poc.py` などに集約し、CLI / GUI / 将来のWeb UIから **共通のパイプラインを呼び出す設計** を前提とする。  
- Webアプリ版を作る際は「既存ロジックをAPI化してフロントエンドから呼ぶ」構成とし、Tkinter実装を直接Webに移植しない。

## 5. プロンプト設計案（コアロジック）
OpenAI/Google/Anthropicの各LLMに送信する指示のプロトタイプです。
基本的なプロンプトは共通で、各プロバイダーのAPI仕様に合わせて送信します。

### 基本プロンプト（リライトなし）

```
【役割】
あなたはプロの動画編集者です。
【指示】
以下の音声認識テキストを、動画テロップ用に整形してください。
【制約条件】
1. **1行あたり全角17文字以内**に収めること。
2. 文脈を読み、**「意味のまとまり」や「息継ぎのタイミング」として自然な位置**で改行すること。
3. 「えー」「あー」「まあ」「その」などのフィラーは削除すること。
4. **原文の単語は極力変更しないこと**。表記ゆれのみ修正可（例：「出来る」→「できる」）。
5. 出力は JSON の `lines` 配列のみ（from/to/text）。説明文は不要。
【出力フォーマット例（パス2想定）】
{"lines":[{"from":0,"to":5,"text":"設定を開いてくださいね"}]}
【入力テキスト】
{transcribed_text}
```

### リライトありプロンプト（オプション）

```
【役割】
あなたはプロの動画編集者です。
【指示】
以下の音声認識テキストを、動画テロップ用に整形してください。
【制約条件】
1. **1行あたり全角17文字以内**に収めること。
2. 文脈を読み、**「意味のまとまり」や「息継ぎのタイミング」として自然な位置**で改行すること。
3. 「えー」「あー」「まあ」「その」などのフィラーは削除すること。
4. 語尾を整え、読みやすく調整すること（例：「〜っていう感じで」→「〜です」）。
5. 出力は JSON の `lines` 配列のみ（from/to/text）。説明文は不要。
【出力フォーマット例（パス2想定）】
{"lines":[{"from":0,"to":5,"text":"設定を開いてください"}]}
【入力テキスト】
{transcribed_text}
```

### プロンプト設計のポイント
*   two-pass固定：パス1は operations JSON、パス2は `lines` JSON（from/to/text）のみを返す
*   `[WORD: ]` タグは廃止。タイムスタンプは word インデックスで直接算出
*   フィラー削除はパス1で最小限に抑え、語順とインデックス整合性を維持
*   Pass3は問題が無くても必ず実行し、5〜17文字・自然な改行を最終確認（短行は統合）
*   Pass4は長さ違反行のみ再LLMし、成功すれば置換。失敗時は元行を残し、その結果をそのまま採用する（ローカル再分割は行わない）

---

## 6. 開発ロードマップ（Plan準拠）

### フェーズ1: コアロジック実装（完了間近）
**目標:** CLIで音声 → SRT を自動生成できるPoCを確立
- 2モデル（mlxデフォルト・openai）実装と比較
- 進捗JSON・SRT出力・エラーハンドリングの統合

### フェーズ2: LLM整形＋タイムスタンプアライメント精度向上（進行中）
- two-pass（operations + lines）出力の精度評価とプロンプト改善
- 実データで17文字制約と自然改行の精度を検証し、パス2プロンプトを調整
- LLMプロバイダー（推奨: Google）で整形を安定化

### フェーズ3: CLI UX / 再開機構強化
- `--resume` フローの長尺E2Eスモーク
- ログ整備と cleanup サブコマンド
- 実行メッセージ/ヘルプを初心者向けに簡潔化

### フェーズ4: GUIプロトタイプ
- Tkinter / Flet でドラッグ＆ドロップ + 進捗バー
- 最小限のオプション（モデル選択・リライト有無）に限定

### フェーズ5: パッケージ化（GUI後）
- PyInstaller / py2app 等で .app 化し、友人環境で配布確認

### （案）フェーズ6: Web UI（要検討・TODO）
- ブラウザから音声ファイルをアップロードし、バックエンドAPI（既存パイプラインのラッパー）経由でSRTを生成できるWebフロントエンド。  
- 具体的な技術スタック（例: FastAPI + React / Next.js）は未決定のため、ここでは **TODOレベルのメモ** として残しておく。  

---

## 7. 技術的リスクと対策

| リスク | 影響度 | 対策 |
|--------|--------|------|
| タイムスタンプアライメント精度が低い | 高 | パス2の from/to インデックス検証とクランプ処理で逆行を防止 |
| LLMが17文字制約を守らない | 中 | プロンプト改善や Pass3/Pass4 のプロンプト調整・テストで制約遵守率を高める（ローカル強制再分割には依存しない） |
| パス2 lines が空/不正 | 中 | 例外処理でSRT生成をスキップしログ出力、再試行を検討 |
| 特定のLLMプロバイダーが障害 | 中 | 3社から選択可能にし、障害時は別プロバイダーへ切り替え |
| LLM APIコストが高騰 | 低 | 各社の低コストモデルを.envでデフォルト設定（gpt-5-mini, gemini-2.5-flash 等） |
| M3 MacでのGPU加速が不安定 | 中 | MLXが落ちる場合は openai-whisper をCPU/MPSで実行する |
| 長時間音声でメモリ不足 | 中 | 全文処理前提。必要に応じて `--resume` で区切り実行し、将来必要なら軽量ブロック分割を再導入 |
| API失敗時の処理中断 | 中 | 進捗を`temp/progress_*.json`に保存、--resumeオプションで再開可能に |

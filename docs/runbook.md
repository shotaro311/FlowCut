# Flow Cut Runbook (開発者向け)

CLI（コマンドラインインターフェース）を使用してFlow Cutを実行するためのガイドです。

## セットアップ

### 1. 前提条件

- Python 3.10 以上 (3.11/3.12 推奨)
- 仮想環境の使用を強く推奨

### 2. インストール

```bash
# 仮想環境の作成と有効化
python -m venv .venv
source .venv/bin/activate

# 依存関係のインストール
pip install -r requirements-dev.txt
```

### 3. 環境変数設定

`.env.example` をコピーして `.env` を作成し、必要なAPIキーを設定してください。

```bash
cp .env.example .env
```

- `OPENAI_API_KEY`: 必須（Google/Anthropicなど他のプロバイダーを使う場合も推奨）
- `ANTHROPIC_API_KEY`: Claudeを使用する場合に必要
- `GOOGLE_API_KEY`: Geminiを使用する場合に必要

---

## コマンドリファレンス

すべてのコマンドはプロジェクトルートから実行します。

### 1. 利用可能なモデルの確認

音声認識に使用できるランナー（Whisperモデル）の一覧を表示します。

```bash
python -m src.cli.main models
```

### 2. 文字起こし・整形実行 (`run`)

音声ファイルまたは動画ファイルを処理してSRT字幕を生成します。
動画ファイル（.mp4, .mov, .mkv, .avi, .webm）の場合は、自動的に音声を抽出して処理します。

#### 基本セットアップ（推奨）

Google Geminiを使用して高速に処理する場合:

```bash
python -m src.cli.main run /path/to/audio.mp3 \
  --llm google \
  --llm-timeout 500
```

動画ファイルを処理する場合:

```bash
python -m src.cli.main run /path/to/video.mp4 \
  --llm google \
  --no-simulate
```

#### よく使うオプション

| オプション | 説明 | 例 |
|------------|------|-----|
| `--llm` | LLMプロバイダー指定 (google, openai, anthropic) | `--llm openai` |
| `--llm-profile` | `config/llm_profiles.json` のプロファイルを使用 | `--llm-profile high_accuracy` |
| `--start-delay` | 字幕全体の開始時間を遅らせる（秒）。冒頭の無音調整用 | `--start-delay 0.5` |
| `--workflow` | 使用するワークフロー (`workflow1`, `workflow2`) | `--workflow workflow2` |
| `--keep-audio` | 動画から抽出した音声ファイルを保存する | `--keep-audio` |
| `--simulate` | 音声認識をスキップし、ダミーデータでLLM整形のみテスト | `--simulate` (デフォルト有効) |
| `--no-simulate` | 実際に音声認識(Whisper)を実行する | `--no-simulate` |
| `--verbose` | 詳細ログを表示 | `--verbose` |

#### 実践的なコマンド例

**本番実行（Whisper + Geminiで整形 + 0.2秒遅延）:**

```bash
python -m src.cli.main run input.wav \
  --no-simulate \
  --llm google \
  --start-delay 0.2
```

**動画ファイル処理（抽出した音声を保存）:**

```bash
python -m src.cli.main run video.mp4 \
  --no-simulate \
  --llm google \
  --keep-audio
```

**開発・テスト（シミュレーション + OpenAI + 詳細ログ）:**

```bash
python -m src.cli.main run dummy.wav \
  --simulate \
  --llm openai \
  --verbose
```

**文字起こしのみ（LLM整形なし）:**

`--llm` オプションを省略すると、Whisperによる文字起こし結果（JSON）のみ保存されます。

```bash
python -m src.cli.main run input.wav --no-simulate
```

### 3. 一時ファイルの掃除 (`cleanup`)

`temp/` や `logs/` ディレクトリに溜まった古いファイルを削除します。

```bash
# 3日以上前のファイルを削除候補として表示（削除はしない）
python -m src.cli.main cleanup --days 3 --dry-run

# 実際に削除
python -m src.cli.main cleanup --days 3
```

---

## 運用・トラブルシューティング

- **出力先**: デフォルトでは `output/` ディレクトリにSRTファイルが生成されます。
- **進捗再開**: 処理が中断した場合、`--resume temp/progress/xxxx.json` で途中から再開できる場合があります。
- **APIエラー**: `llm_timeout` エラーが発生する場合は、`--llm-timeout 600` 等と長く設定してください。
- **GUI起動**: `python -m src.cli.main gui` でGUI版を起動できます。

### 4. ワークフローの切り替え

検証用の最適化ロジック（`src/llm/two_pass_optimized.py`）を使用したい場合は、`--workflow workflow2` を指定します。

```bash
.venv/bin/python -m src.cli.main run samples/sample_audio.m4a --llm google --workflow workflow2 --no-simulate
```

/Users/shotaro/code/client/FlowCut/samples/sample_audio.m4a

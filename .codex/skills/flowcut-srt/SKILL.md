---
name: flowcut-srt
description: "音声/動画ファイルからテロップ用SRT（.srt）を生成する（Whisper + Gemini CLI / FlowCut workflow2互換）。「SRT作って」「テロップ作って」「字幕作って」などの依頼で使用。"
---

# flowcut-srt（Whisper + Gemini CLI で SRT 生成）

## 目的
- 指定された音声/動画ファイルから、テロップ用の `.srt` を生成してファイル保存します（FlowCut の `workflow2` 相当）。

## 前提（必須）
- `gemini` コマンドが実行できる（認証済み）
- 動画入力の場合は `ffmpeg` が利用できる
- Whisper バックエンドが利用できる（デフォルト: `WHISPER_BACKEND=auto`）
  - mac: `mlx-whisper`（`python3 -c "import mlx_whisper"`）
  - Windows: `faster-whisper`（`python3 -c "import faster_whisper"`）
  - その他: `openai-whisper`（`python3 -c "import whisper"`）

## 生成物（固定）
- `output/<入力stem>_<timestamp>/<run_id>.srt`
- `output/<入力stem>_<timestamp>/logs/`
  - `poc_samples/<run_id>.json`（文字起こし結果）
  - `llm_raw/`（Passごとの生ログ）

## 実行手順（エージェント向け）
1. ユーザーの指示から「入力ファイルパス（音声/動画）」を1つ特定する。見つからなければ、パスを1つだけ質問する。
2. 事前チェック（不足があればユーザーに不足内容を伝えて止める）
   - `command -v gemini`
   - 動画の場合のみ `command -v ffmpeg`
   - Whisper（`WHISPER_BACKEND=auto` の場合）
     - mac: `python3 -c "import mlx_whisper"`
     - Windows: `python3 -c "import faster_whisper"`
     - その他: `python3 -c "import whisper"`
3. SRT生成を実行する（`.venv` がある場合はそれを優先）
   - `.venv/bin/python .codex/skills/flowcut-srt/scripts/make_srt.py "<入力パス>"`
   - `.venv` が無い場合: `python3 .codex/skills/flowcut-srt/scripts/make_srt.py "<入力パス>"`
4. 成功したら、スクリプトが出力する `done: <srt_path>` をそのままユーザーに返す。

## 注意点
- APIキーなどの機密値を出力しない（ログにも残さない）。
- 失敗時は `output/.../logs/llm_raw/` を確認すると原因が追いやすい。

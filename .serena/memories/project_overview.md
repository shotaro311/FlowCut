# FlowCut 概要

## 目的
- 音声/動画ファイルから日本語字幕（`.srt`）を生成するデスクトップアプリ。
- ローカルで Whisper による文字起こしを行い、必要に応じて LLM（Google/OpenAI/Anthropic）で字幕用に整形する。

## 技術スタック
- Python（CLI: Typer、GUI: Tkinter）
- LLM 呼び出し: requests ベース（プロバイダー別）
- テスト: pytest
- パッケージング: PyInstaller（macOS: `FlowCut.spec` / Windows: `FlowCut_win.spec`）

## ざっくり構成
- `src/cli/`: CLI エントリ（`python -m src.cli.main ...`）
- `src/gui/`: GUI 実装（`flowcut_gui_launcher.py` → `src.gui.app.run_gui`）
- `src/transcribe/`: 文字起こし（Whisper ランナー）
- `src/llm/`: LLM 整形（ワークフロー、two-pass など）
- `src/alignment/`: SRT セグメント生成/調整
- `src/utils/`: ログ、進捗、メトリクスなど
- `docs/`: 要件・手順・PLAN

## 主要ドキュメント
- `docs/requirement.md`
- `docs/runbook.md`
- `docs/plan/20251203_PLAN1.md`

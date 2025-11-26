# Flow Cut Runbook (開発者向け)

## セットアップ
- Python 3.12 推奨（3.10-3.12対応）
- 仮想環境作成＆依存導入
  ```bash
  python -m venv .venv
  . .venv/bin/activate
  pip install -r requirements-dev.txt
  ```
- 環境変数: `.env.example` をコピーし、少なくとも `OPENAI_API_KEY` を設定。  
  Whisper APIを使う場合は `OPENAI_WHISPER_MODEL`（例: whisper-1）も確認。

## よく使うコマンド
- モデル一覧: `python -m src.cli.main models`
- PoC実行（シミュレーション音声+LLM整形=two-pass固定）:
  ```bash
  python -m src.cli.main run samples/sample_audio.m4a \
    --llm google \
    --llm-timeout 500
  ```
  *実課金に注意。整形不要なら `--llm` を省略し、文字起こしJSONのみ保存します。*

- 中断再開: `python -m src.cli.main run --resume temp/progress/<run>.json`

- テンポラリ掃除: `python -m src.cli.main cleanup --days 3 --dry-run`  
  実行削除は `--dry-run` を外す。`--paths` で任意パスを指定可。

- テスト: `pytest`

## 主要オプション速見表
- `--llm` / `--rewrite` / `--llm-temperature` / `--llm-timeout`
- 出力・進捗: `--output-dir`, `--progress-dir`, `--subtitle-dir`（PocRunOptionsで統一）
- 一括掃除: `cleanup` サブコマンドまたは `scripts/cleanup_temp.py`

## 運用メモ
- Whisperランナー: kotoba / mlx / openai を選択可能（デフォルトは MLX）。フォールバックなしのtwo-pass専用フロー。
- アライン調整オプション（RapidFuzz）は撤去済み。SRTは two-pass の `lines` 出力をワードインデックス直結で生成。
- API安定性: LLM呼び出しは 1→3→5 秒のバックオフ付きリトライ。デフォルトタイムアウトは 500 秒（`LLM_REQUEST_TIMEOUT` / `--llm-timeout` で上書き可）。

## 次に触るときのチェックリスト
- `.env` のキーとモデル名が有効か（特に `OPENAI_WHISPER_MODEL`）。
- `temp/` の肥大化は `cleanup` サブコマンドで掃除。
- 長尺サンプルを受領したら、2モデル（mlx / openai）で実行して `reports/poc_whisper_metrics.csv` を更新。行分割は two-pass の出力を使用し、アライン調整は不要。

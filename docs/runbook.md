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
- PoC実行（シミュレーション音声+SRT生成、LLM付き）:
  ```bash
  python -m src.cli.main run samples/sample_audio.m4a \
    --llm openai --rewrite \
    --llm-timeout 60 \
    --align-thresholds 92,85,80 --align-gap 0.15 --align-fallback-padding 0.4
  ```
  *実課金に注意。シミュレーションで十分なら `--llm` を省略してください。*

- 中断再開: `python -m src.cli.main run --resume temp/progress/<run>.json`

- テンポラリ掃除: `python -m src.cli.main cleanup --days 3 --dry-run`  
  実行削除は `--dry-run` を外す。`--paths` で任意パスを指定可。

- テスト: `pytest`

## 運用メモ
- kotoba / mlx ランナーは現状 OpenAI Whisper へフォールバック実行。ネイティブ実装は今後対応。
- アライン調整: `--align-thresholds`（カンマ区切り）、`--align-gap`、`--align-fallback-padding` で調整可能。デフォルトは `90,85,80 / 0.1 / 0.3`。
- API安定性: LLM呼び出しは 1→3→5 秒のバックオフ付きリトライ、`--llm-timeout` でリクエストタイムアウトを上書き可能。

## 次に触るときのチェックリスト
- `.env` のキーとモデル名が有効か（特に `OPENAI_WHISPER_MODEL`）。
- `temp/` の肥大化は `cleanup` サブコマンドで掃除。
- 長尺サンプルを受領したら、3モデルで実行して `reports/poc_whisper_metrics.csv` を更新。Warningログを見てアライン閾値を調整。

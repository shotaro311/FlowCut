# Flow Cut Runbook (開発者向け)

## セットアップ
- Python 3.12 推奨（3.10-3.12対応）
- 仮想環境作成＆依存導入
  ```bash
  python -m venv .venv
  . .venv/bin/activate
  pip install -r requirements-dev.txt
  ```
- 環境変数: `.env.example` をコピーし、少なくとも `GOOGLE_API_KEY` を設定（OpenAI の LLM/API を使う場合のみ `OPENAI_API_KEY` も設定）。  
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
- Whisperランナー: mlx / openai を選択可能（デフォルトは MLX）。フォールバックなしのtwo-pass専用フロー。
- アライン調整オプション（RapidFuzz）は撤去済み。SRTは two-pass の `lines` 出力をワードインデックス直結で生成。
- API安定性: LLM呼び出しは 1→3→5 秒のバックオフ付きリトライ。デフォルトタイムアウトは 500 秒（`LLM_REQUEST_TIMEOUT` / `--llm-timeout` で上書き可）。

## 次に触るときのチェックリスト
- `.env` のキーとモデル名が有効か（特に `OPENAI_WHISPER_MODEL`）。
- `temp/` の肥大化は `cleanup` サブコマンドで掃除。
- 長尺サンプルを受領したら、2モデル（mlx / openai）で実行して `reports/poc_whisper_metrics.csv` を更新。行分割は two-pass の出力を使用し、アライン調整は不要。

## パッケージング（macOS / Windows 共通の考え方）

アプリの機能や画面を変更した場合は、**ソースコードの更新 → テスト → 各OS向けパッケージの再生成** という流れで配布物を更新する。

1. `main` ブランチを最新化し、ローカル環境でテスト・簡易動作確認を行う。
2. macOS / Windows それぞれの開発環境で PyInstaller を実行し、`.app` / `.exe` を含むパッケージを再生成する。
3. `dist/` 以下にできた成果物を zip 等で固めて、ユーザー（友人）に配布する。

### macOS 向け `.app` パッケージ化の手順（概要）

前提: 開発者用の macOS 環境（Apple Silicon 推奨）で、`mlx-whisper` と `ffmpeg` がインストール済み。

1. ブランチと依存を整える
   ```bash
   git switch main
   git pull --ff-only origin main
   python -m venv .venv
   . .venv/bin/activate
   pip install -r requirements-dev.txt
   ```
2. `.env` を `.env.example` から用意し、必要なAPIキー（特に **`GOOGLE_API_KEY`（必須） / `OPENAI_API_KEY`（OpenAI 利用時のみ）**）を設定する。
3. 開発環境で CLI / GUI の動作確認を行う（例: `python -m src.cli.main models`, `python -m src.cli.main gui`）。
4. PyInstaller で `.app` を生成する（レシピは `FlowCut.spec` を想定）。
   ```bash
   pyinstaller FlowCut.spec
   ```
5. `dist/FlowCut.app` が生成されるので、これを zip に固めて `FlowCut-mac.zip` のような名前で友人に渡す。

※ macOS 向けバンドル内容（`datas` / `binaries` / `hiddenimports`）の詳細は `FlowCut.spec` を参照。

### Windows 向け `.exe` パッケージ化の手順（概要）

前提: Windows 10/11 64bit の開発環境で Python 3.10〜3.12 と `pyinstaller` が利用可能になっていること。
（Whisper 用には `openai-whisper` を利用する想定）

1. ブランチと依存を整える
   ```bash
   git switch main
   git pull --ff-only origin main
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements-dev.txt
   ```
2. `.env` を `.env.example` から用意し、必要なAPIキー（特に **`GOOGLE_API_KEY`（必須） / `OPENAI_API_KEY`（OpenAI 利用時のみ）**）を設定する。
3. 開発環境で CLI / GUI の動作確認を行う。
   ```bash
   python -m src.cli.main models
   python -m src.cli.main gui
   ```
4. 事前に `assets/FlowCut.ico` を用意する（既存の `assets/FlowCut.iconset` / `assets/FlowCut.icns` からアイコン変換ツール等で生成し、リポジトリに配置する）。
5. Windows 用の PyInstaller レシピ（`FlowCut_win.spec`）を使って one-folder 形式の出力を作成する。
   ```bash
   pyinstaller FlowCut_win.spec
   ```
6. `dist/FlowCut/FlowCut.exe` が生成されたら、Whisper が参照する `assets` データを dist 側にコピーする（暫定手順）。  
   ```powershell
   New-Item -ItemType Directory -Force -Path "dist\FlowCut\_internal\whisper\assets"
   Copy-Item ".venv\Lib\site-packages\whisper\assets\*" "dist\FlowCut\_internal\whisper\assets\" -Recurse -Force
   ```
7. `dist/FlowCut/` フォルダごと zip に固めて `FlowCut-win.zip` とし、友人には「解凍 → `FlowCut.exe` ダブルクリック」で使ってもらう。

※ Windows 版で同梱する ffmpeg や Whisper ランタイムの詳細構成は `docs/plan/20251203_PLAN1.md`（Windows版 FlowCut GUI パッケージ化 PLAN）を参照。

# よく使うコマンド

## セットアップ
- venv + 依存導入:
  - `python3 -m venv .venv`
  - `. .venv/bin/activate`
  - `python -m pip install -U pip`
  - `python -m pip install -r requirements-dev.txt`
- `.env` 作成:
  - `cp .env.example .env`

## 実行（ソースから）
- GUI 起動: `python -m src.cli.main gui`
- CLI 実行: `python -m src.cli.main run <audio_or_video> --no-simulate`

## テスト
- `python -m pytest -q`

## パッケージング（PyInstaller）
- macOS: `python -m PyInstaller FlowCut.spec --clean --noconfirm`
- Windows: `python -m PyInstaller FlowCut_win.spec --clean --noconfirm`

## 便利
- 変更確認: `git status` / `git diff`
- 文字列検索: `rg "pattern"`

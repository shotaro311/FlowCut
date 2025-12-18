# Mac Application Packaging Guide

FlowCut の Mac アプリケーション (`.app`) パッケージング手順書です。
PyInstaller を使用して、Python 環境を含んだ単一のアプリケーションバンドルを作成します。

## 前提条件

1. **Python 環境**
   - macOS では `python` コマンドが無い環境があるため、基本は `python3` を使用してください。
   - 既存の venv を使う場合は、`./.venv-gui/bin/python`（または `./.venv/bin/python`）が存在することを確認してください。

2. **PyInstaller**
   - venv に PyInstaller がインストールされていること。
   - `python3 -m pip install -U pip pyinstaller`

3. **依存ライブラリ（FlowCut.spec を使う場合）**
   - `FlowCut.spec` は `mlx_whisper` / `mlx` を同梱対象にしているため、事前に venv へ導入してください。
   - `python3 -m pip install mlx-whisper mlx`

4. **FFmpeg**
   - アプリケーションに同梱するため、`ffmpeg` バイナリが必要です。
   - `FlowCut.spec` では `/opt/homebrew/bin/ffmpeg` を参照しています。Homebrew でインストール済みであることを確認してください。

## ビルド手順

プロジェクトのルートディレクトリで以下のコマンドを実行します。

```bash
# 既存のビルドキャッシュをクリアしてビルド（PATH に依存しない実行方法）
./.venv-gui/bin/python -m PyInstaller FlowCut.spec --clean --noconfirm

# もしくは（venv を有効化している場合）
# pyinstaller FlowCut.spec --clean --noconfirm
```

成功すると、最後に `INFO: Build complete!` と表示されます。

## 生成物

ビルドが完了すると、`dist/` ディレクトリにアプリケーションが生成されます。

- **パス**: `dist/FlowCut.app`

この `.app` ファイルは、Python ランタイムと依存ライブラリ（FFmpeg含む）を内包しているため、Python がインストールされていない他の Mac でも（アーキテクチャが同じなら）動作する可能性があります。

## トラブルシューティング

### "command not found: python" / "command not found: pyinstaller"
`python` / `pyinstaller` が PATH に無い状態です。以下のいずれかで回避できます。

- venv の Python をフルパスで呼ぶ: `./.venv-gui/bin/python -m PyInstaller ...`
- venv を有効化してから実行: `. .venv-gui/bin/activate`

### "tkinter" 関連の警告・エラー
ビルドログに `WARNING: tkinter installation is broken.` と出る場合、または起動時に `ModuleNotFoundError: No module named 'tkinter'` でクラッシュする場合は、Python 環境に `python-tk` が不足しています。
以下のコマンドでインストールしてください（Python 3.11 の場合）。

```bash
brew install python-tk@3.11
```

その後、再度ビルドを行ってください。

### 起動時のセキュリティ警告
署名（Code Signing）を行っていないため、他の Mac 別環境で開こうとすると macOS のセキュリティ機能（Gatekeeper）により「開発元が未確認のため開けません」と表示される場合があります。
その場合は、Finder でアプリを**右クリック（Ctrl+クリック）して「開く」**を選択し、警告ダイアログで「開く」を押してください。

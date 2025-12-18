# Mac Application Packaging Guide

FlowCut の Mac アプリケーション (`.app`) パッケージング手順書です。
PyInstaller を使用して、Python 環境を含んだ単一のアプリケーションバンドルを作成します。

## 前提条件

1. **Python 環境**
   - 開発で使用している仮想環境 (`.venv`) に `pyinstaller` がインストールされていること。
   - `pip install pyinstaller`

2. **FFmpeg**
   - アプリケーションに同梱するため、`ffmpeg` バイナリが必要です。
   - `FlowCut.spec` では `/opt/homebrew/bin/ffmpeg` を参照しています。Homebrew でインストール済みであることを確認してください。

## ビルド手順

プロジェクトのルートディレクトリで以下のコマンドを実行します。

```bash
# 既存のビルドキャッシュをクリアしてビルド
  cd /Users/shotaro/code/client/FlowCut
  ./.venv-gui/bin/python -m pip install -U pip pyinstaller
  ./.venv-gui/bin/python -m PyInstaller FlowCut.spec --clean --noconfirm
```

成功すると、最後に `INFO: Build complete!` と表示されます。

## 生成物

ビルドが完了すると、`dist/` ディレクトリにアプリケーションが生成されます。

- **パス**: `dist/FlowCut.app`

この `.app` ファイルは、Python ランタイムと依存ライブラリ（FFmpeg含む）を内包しているため、Python がインストールされていない他の Mac でも（アーキテクチャが同じなら）動作する可能性があります。

## トラブルシューティング

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

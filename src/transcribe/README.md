# src/transcribe

音声認識まわりのランナー群・共通インターフェースを配置します。Phase 1 の運用では、macOS は `mlx` / `whisper-local`、Windows は `faster` / `whisper-local` を利用し、word-level タイムスタンプ付きのJSONを出力します。

- `base.py`: Runnerインターフェース、結果データモデル、レジストリ
- `mlx_runner.py` / `faster_whisper_runner.py` / `whisper_local_runner.py`: 実際のモデル統合レイヤー
- `__init__.py`: 外部公開APIと自動登録

PoC中はシミュレーションモードでのダミー結果生成にも対応させ、ベンチマーク時に本実装へ差し替えます。

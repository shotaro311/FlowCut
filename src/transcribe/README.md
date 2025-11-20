# src/transcribe

音声認識まわりのランナー群・共通インターフェースを配置します。Phase 1 のPoCでは kotoba-mlx / mlx-large-v3 / openai-whisper の3モデルを切り替えながら比較し、word-levelタイムスタンプ付きのJSONを出力します。

- `base.py`: Runnerインターフェース、結果データモデル、レジストリ
- `kotoba_runner.py` など: 実際のモデル統合レイヤー
- `__init__.py`: 外部公開APIと自動登録

PoC中はシミュレーションモードでのダミー結果生成にも対応させ、ベンチマーク時に本実装へ差し替えます。

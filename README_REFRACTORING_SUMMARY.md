# リファクタリングまとめ（2025-11-21 00:45 JST）

- LLMフォーマッター: バックオフ付きリトライ（1s→3s→5s）を実装し、`max_retries` 設定を実際に活用。
- LLM呼び出し: temperature / timeout を CLI → パイプライン → プロバイダーまで伝播。
- Whisperランナー: OpenAI本実装を共通関数化し、kotoba/mlx は当面フォールバックで動作させる方式へ統合。
- ハウスキーピング: `python -m src.cli.main cleanup` / `scripts/cleanup_temp.py` で temp/log の古いファイルを一括削除。
- テスト: 40→42件まで拡充（リトライ・フォールバック・クリーンアップ）。

今後の優先リファクタリング候補
1. kotoba/mlx ネイティブ実装を追加し、フォールバック依存を解消。
2. LLM整形→アライン→SRT を1コマンドで実データ検証するE2Eスモークを追加。
3. `align_kwargs` を CLI から受け取れるよう辞書引数を設計し、閾値調整を容易化。

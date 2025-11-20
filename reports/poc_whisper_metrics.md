# Whisper モデル比較レポート（テンプレート）

- **日付**: 2025-11-20
- **担当**: Shotaro
- **対象フェーズ**: Phase 1 - 音声認識PoC

## 1. テスト条件
- サンプル音声:
  - [ ] news_anchor_ja_5m.wav
  - [ ] dialog_podcast_mix_12m.wav
  - [ ] その他: 
- 共通設定: 1200文字/30秒ブロック, 44.1kHz WAV, シミュレーション=OFF
- 実行環境: Apple M3 Max / 64GB RAM / macOS 15.0.1

## 2. サマリー指標
| モデル | WER (%) | RTF | GPUメモリ(MB) | 備考 |
|--------|---------|-----|---------------|------|
| kotoba |  |  |  |  |
| mlx    |  |  |  |  |
| openai |  |  |  |  |

## 3. 詳細ログ
- `temp/poc_samples/` に出力された raw JSON へのリンク
- 実行コマンド例: `python scripts/poc_transcribe.py --audio ...`

## 4. 所感 / 次アクション
- [ ] モデル差分の仮説
- [ ] アライメントへの影響
- [ ] コスト試算（API/推論時間）

---
※ 数値が揃い次第、docs/plan のリスク・フェーズ進捗を更新してください。

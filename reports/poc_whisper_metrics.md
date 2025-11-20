# Whisper モデル比較レポート（テンプレート）

- **日付**: 2025-11-20
- **担当**: Shotaro
- **対象フェーズ**: Phase 1 - 音声認識PoC

## 1. テスト条件
- サンプル音声:
  - [ ] news_anchor_ja_5m.wav
  - [ ] dialog_podcast_mix_12m.wav
  - [x] その他: `samples/sample_audio.m4a`（約3秒、PoCシミュレーション）
- 共通設定: 1200文字/30秒ブロック, 44.1kHz, シミュレーション=ON（runner simulate=True）
- 実行環境: Apple M3 Max / 64GB RAM / macOS 15.0.1

## 2. サマリー指標
| モデル | WER (%) | RTF | GPUメモリ(MB) | 備考 |
|--------|---------|-----|---------------|------|
| kotoba | N/A (simulate) | N/A | 0 (simulate) | `temp/poc_samples/sample_audio_kotoba_20251120T173552.json` |
| mlx    | N/A (simulate) | N/A | 0 (simulate) | `temp/poc_samples/sample_audio_mlx_20251120T173552.json` |
| openai | N/A (simulate) | N/A | 0 (simulate) | `temp/poc_samples/sample_audio_openai_20251120T173552.json` |

## 3. 詳細ログ
- `temp/poc_samples/sample_audio_{kotoba,mlx,openai}_20251120T173552.json`
- `temp/progress/sample_audio_{kotoba,mlx,openai}_20251120T173552.json`
- 実行コマンド: `python scripts/poc_transcribe.py --audio samples/sample_audio.m4a --models kotoba,mlx,openai --simulate`

## 4. 所感 / 次アクション
- [ ] モデル差分の仮説（実サンプル待ち）
- [ ] アライメントへの影響
- [ ] コスト試算（API/推論時間）
- メモ: シミュレーション出力でCSVの体裁を確認済み。本番音声取得後にWER/RTF等を採取予定。

---
※ 数値が揃い次第、docs/plan のリスク・フェーズ進捗を更新してください。

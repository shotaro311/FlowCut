# Whisper モデル比較レポート（テンプレート）

- **日付**: 2025-11-20
- **担当**: Shotaro
- **対象フェーズ**: Phase 1 - 音声認識PoC

## 1. テスト条件
- サンプル音声:
  - [ ] news_anchor_ja_5m.wav
  - [ ] dialog_podcast_mix_12m.wav
  - [x] その他: `samples/sample_audio.m4a`（約5分、OpenAIランナー実走 + LLM整形/SRT）
- 共通設定: 1200文字/30秒ブロック, 44.1kHz, シミュレーション=ON→OpenAIのみOFF（LLM_REQUEST_TIMEOUT=120s）
- 実行環境: Apple M3 Max / 64GB RAM / macOS 15.0.1

## 2. サマリー指標
| モデル | WER (%) | RTF | GPUメモリ(MB) | 備考 |
|--------|---------|-----|---------------|------|
| kotoba | N/A (simulate) | N/A | 0 (simulate) | `temp/poc_samples/sample_audio_kotoba_20251120T173552.json` |
| mlx    | N/A (simulate) | N/A | 0 (simulate) | `temp/poc_samples/sample_audio_mlx_20251120T173552.json` |
| openai | TBD (実測, LLM整形あり) | TBD | Cloud | `temp/poc_samples/sample_audio_openai_20251121T141838.json` / `output/sample_audio_openai_20251121T141838.srt` |

## 3. 詳細ログ
- `temp/poc_samples/sample_audio_{kotoba,mlx,openai}_20251120T173552.json`
- `temp/progress/sample_audio_{kotoba,mlx,openai}_20251120T173552.json`
- 追加: `temp/poc_samples/sample_audio_openai_20251121T141838.json`, `output/sample_audio_openai_20251121T141838.srt`, `temp/progress/sample_audio_openai_20251121T141838.json`
- 実行コマンド: `LLM_REQUEST_TIMEOUT=120 python -m src.cli.main run samples/sample_audio.m4a --models openai --no-simulate --llm openai --llm-temperature 0.7 --align-thresholds 90,85,80 --align-gap 0.1 --align-fallback-padding 0.3`

## 4. 所感 / 次アクション
- [ ] モデル差分の仮説（他2モデルも実走して比較）
- [x] アライメントへの影響（OpenAI実走でlong block検知、BlockSplitterのduration強制分割が課題）
- [ ] コスト試算（API/推論時間）
- メモ: OpenAIのみ実走済み。`reports/poc_whisper_metrics.csv` に長文1ブロックで30秒超過/1200文字超過の違反が記録されている。BlockSplitter改善で解消予定。WER/RTFは手元の参照テキストがないため未算出。

---
※ 数値が揃い次第、docs/plan のリスク・フェーズ進捗を更新してください。

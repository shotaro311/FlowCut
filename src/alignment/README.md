# alignment

`[WORD: ]` アンカー付き行と word-level タイムスタンプを突き合わせ、字幕行ごとの開始・終了時刻を決めるモジュールです。
- `timestamp.py`: RapidFuzz によるファジーマッチとフォールバックを実装し、警告は `logs/alignment_warnings.json` に記録。
- `srt.py`: `AlignedLine` から SRT テキストを生成。`align_to_srt()` でアラインとSRT出力を一気通貫に実行できます。
- LLM整形後の行を CLI / パイプラインから再利用できるように設計しています。

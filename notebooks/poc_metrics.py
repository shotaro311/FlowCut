"""Whisper PoC メトリクス集計スクリプト（ドラフト）

使い方:
    python notebooks/poc_metrics.py --input temp/poc_samples --output reports/poc_whisper_metrics.csv

現状はダミー実装で、将来的にWER・RTF・メモリを集計するロジックを追加します。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable


def iter_json_files(target_dir: Path) -> Iterable[Path]:
    for path in sorted(target_dir.glob('*.json')):
        if path.is_file():
            yield path


def main() -> None:
    parser = argparse.ArgumentParser(description='Whisper PoC メトリクス集計（WIP）')
    parser.add_argument('--input', type=Path, default=Path('temp/poc_samples'), help='JSON出力を含むディレクトリ')
    parser.add_argument('--output', type=Path, default=Path('reports/poc_whisper_metrics.csv'), help='集計結果を書き出すCSVファイル')
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows: list[str] = ['model,audio_file,text_length,num_words']

    for json_file in iter_json_files(args.input):
        data = json.loads(json_file.read_text())
        model = data.get('metadata', {}).get('model') or 'unknown'
        audio = data.get('metadata', {}).get('audio_file') or json_file.stem
        text_length = len(data.get('text', ''))
        num_words = len(data.get('words', []))
        rows.append(f"{model},{audio},{text_length},{num_words}")

    args.output.write_text('\n'.join(rows))
    print(f'[INFO] CSVを書き出しました: {args.output}')


if __name__ == '__main__':
    main()

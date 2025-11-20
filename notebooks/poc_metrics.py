"""Whisper PoC メトリクス集計スクリプト。

ブロックJSON (`blocks`) を用いて 1200文字/30秒ルールや重複文の比率をモニタリングする。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List
import unicodedata


DEFAULT_CHAR_LIMIT = 1200.0
DEFAULT_DURATION_LIMIT = 30.0


def _display_width(text: str) -> float:
    width = 0.0
    for ch in text:
        if ch == '\n':
            continue
        width += 1.0 if unicodedata.east_asian_width(ch) in {'W', 'F'} else 0.5
    return width


def iter_json_files(target_dir: Path) -> Iterable[Path]:
    for path in sorted(target_dir.glob('*.json')):
        if path.is_file():
            yield path


def summarize_blocks(blocks: List[dict], *, char_limit: float, duration_limit: float) -> dict:
    if not blocks:
        return {
            'num_blocks': 0,
            'max_block_chars': 0.0,
            'max_block_duration': 0.0,
            'char_violation_blocks': 0,
            'duration_violation_blocks': 0,
            'overlap_sentence_ratio': 0.0,
        }

    max_chars = 0.0
    max_duration = 0.0
    char_violations = 0
    duration_violations = 0
    overlap_sentences = 0
    total_sentences = 0

    for block in blocks:
        text = (block.get('text') or '').strip()
        width = _display_width(text)
        max_chars = max(max_chars, width)
        if width > char_limit:
            char_violations += 1
        duration = block.get('duration') or 0.0
        max_duration = max(max_duration, duration)
        if duration > duration_limit:
            duration_violations += 1
        for sentence in block.get('sentences', []):
            total_sentences += 1
            if sentence.get('overlap'):
                overlap_sentences += 1

    overlap_ratio = overlap_sentences / total_sentences if total_sentences else 0.0
    return {
        'num_blocks': len(blocks),
        'max_block_chars': round(max_chars, 2),
        'max_block_duration': round(max_duration, 2),
        'char_violation_blocks': char_violations,
        'duration_violation_blocks': duration_violations,
        'overlap_sentence_ratio': round(overlap_ratio, 3),
    }


def summarize_file(json_file: Path, *, char_limit: float, duration_limit: float) -> dict:
    data = json.loads(json_file.read_text())
    metadata = data.get('metadata', {})
    base = {
        'model': metadata.get('model') or 'unknown',
        'audio_file': metadata.get('audio_file') or json_file.stem,
        'text_length': len(data.get('text', '')),
        'num_words': len(data.get('words', [])),
    }
    block_stats = summarize_blocks(data.get('blocks', []), char_limit=char_limit, duration_limit=duration_limit)
    return {**base, **block_stats}


def format_csv_row(record: dict, *, headers: list[str]) -> str:
    cells: list[str] = []
    for key in headers:
        value = record.get(key, '')
        cells.append(str(value))
    return ','.join(cells)


def main() -> None:
    parser = argparse.ArgumentParser(description='Whisper PoC メトリクス集計（ブロック検証対応）')
    parser.add_argument('--input', type=Path, default=Path('temp/poc_samples'), help='JSON出力を含むディレクトリ')
    parser.add_argument('--output', type=Path, default=Path('reports/poc_whisper_metrics.csv'), help='集計結果を書き出すCSVファイル')
    parser.add_argument('--char-limit', type=float, default=DEFAULT_CHAR_LIMIT, help='ブロック当たりの文字数上限')
    parser.add_argument('--duration-limit', type=float, default=DEFAULT_DURATION_LIMIT, help='ブロック当たりの秒数上限')
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    headers = [
        'model',
        'audio_file',
        'text_length',
        'num_words',
        'num_blocks',
        'max_block_chars',
        'max_block_duration',
        'char_violation_blocks',
        'duration_violation_blocks',
        'overlap_sentence_ratio',
    ]

    rows: list[str] = [','.join(headers)]
    for json_file in iter_json_files(args.input):
        record = summarize_file(
            json_file,
            char_limit=args.char_limit,
            duration_limit=args.duration_limit,
        )
        rows.append(format_csv_row(record, headers=headers))

    args.output.write_text('\n'.join(rows))
    print(f'[INFO] CSVを書き出しました: {args.output}')


if __name__ == '__main__':
    main()

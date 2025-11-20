#!/usr/bin/env python3
"""Whisperモデル比較用PoCスクリプト。

サンプル:
    python scripts/poc_transcribe.py \
        --audio samples/dummy.wav \
        --models kotoba,mlx \
        --output-dir temp/poc_samples
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable, List

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / 'src'
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from transcribe import (  # noqa: E402
    TranscriptionConfig,
    TranscriptionResult,
    available_runners,
    describe_runners,
    get_runner,
)

logger = logging.getLogger('poc_transcribe')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Whisperモデル比較PoC')
    parser.add_argument('--audio', nargs='+', type=Path, help='入力音声ファイルへのパス', required=False)
    parser.add_argument('--models', type=str, default=None, help='カンマ区切りでランナーを指定 (例: kotoba,mlx)')
    parser.add_argument('--language', type=str, default=None, help='明示的な言語コード（例: ja, en）')
    parser.add_argument('--chunk-size', type=int, default=None, help='モデルごとのチャンクサイズを上書き')
    parser.add_argument('--output-dir', type=Path, default=Path('temp/poc_samples'), help='結果を保存するディレクトリ')
    parser.add_argument('--list-models', action='store_true', help='利用可能なランナー一覧を表示して終了')
    parser.add_argument('--simulate', action=argparse.BooleanOptionalAction, default=True, help='シミュレーションモードをON/OFF')
    parser.add_argument('--verbose', action='store_true', help='DEBUGログを出力')
    args = parser.parse_args()

    if args.list_models:
        show_models()
        sys.exit(0)

    if not args.audio:
        parser.error('--audio は必須です')
    return args


def show_models() -> None:
    print('利用可能なランナー:')
    for data in describe_runners():
        print(f"- {data['slug']:7} | {data['display_name']} | model={data['default_model']}")


def resolve_models(raw: str | None) -> List[str]:
    if not raw:
        return available_runners()
    requested = [token.strip() for token in raw.split(',') if token.strip()]
    unknown = [slug for slug in requested if slug not in available_runners()]
    if unknown:
        raise SystemExit(f'未登録のランナーが指定されました: {unknown}. 候補: {available_runners()}')
    return requested


def ensure_audio_files(paths: Iterable[Path]) -> List[Path]:
    resolved: List[Path] = []
    for path in paths:
        if not path.exists():
            raise SystemExit(f'音声ファイルが見つかりません: {path}')
        resolved.append(path)
    return resolved


def save_result(result: TranscriptionResult, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(result.to_json())
    logger.info('結果を保存しました: %s', dest)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format='[%(levelname)s] %(message)s')

    audio_files = ensure_audio_files(args.audio)
    models = resolve_models(args.models)
    output_dir = args.output_dir
    timestamp = datetime.now().strftime('%Y%m%dT%H%M%S')

    for slug in models:
        runner = get_runner(slug)
        config = TranscriptionConfig(
            language=args.language,
            chunk_size=args.chunk_size,
            simulate=args.simulate,
            extra={'requested_at': timestamp},
        )
        logger.info('=== %s (%s) ===', slug, runner.display_name)
        runner.prepare(config)
        for audio_path in audio_files:
            result = runner.transcribe(audio_path, config)
            out_name = f"{audio_path.stem}_{slug}_{timestamp}.json"
            save_result(result, output_dir / out_name)


if __name__ == '__main__':
    main()

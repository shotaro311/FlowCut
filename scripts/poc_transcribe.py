#!/usr/bin/env python3
"""Whisperモデル比較用PoCスクリプト（モジュール化された実装を呼び出す）。"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / 'src'
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline.poc import (  # noqa: E402
    PocRunOptions,
    ensure_audio_files,
    execute_poc_run,
    list_models,
    resolve_models,
)

logger = logging.getLogger('poc_transcribe')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Whisperモデル比較PoC')
    parser.add_argument('--audio', nargs='+', type=Path, help='入力音声ファイルへのパス', required=False)
    parser.add_argument('--models', type=str, default=None, help='カンマ区切りでランナーを指定 (例: kotoba,mlx)')
    parser.add_argument('--language', type=str, default=None, help='明示的な言語コード（例: ja, en）')
    parser.add_argument('--chunk-size', type=int, default=None, help='モデルごとのチャンクサイズを上書き')
    parser.add_argument('--output-dir', type=Path, default=Path('temp/poc_samples'), help='結果を保存するディレクトリ')
    parser.add_argument('--progress-dir', type=Path, default=Path('temp/progress'), help='進捗ファイルの書き出し先')
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
    for data in list_models():
        print(f"- {data['slug']:7} | {data['display_name']} | model={data['default_model']}")


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format='[%(levelname)s] %(message)s')

    audio_files = ensure_audio_files(args.audio)
    try:
        models = resolve_models(args.models)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    options = PocRunOptions(
        language=args.language,
        chunk_size=args.chunk_size,
        output_dir=args.output_dir,
        progress_dir=args.progress_dir,
        simulate=args.simulate,
        verbose=args.verbose,
    )
    execute_poc_run(audio_files, models, options)


if __name__ == '__main__':
    main()

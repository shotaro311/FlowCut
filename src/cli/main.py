"""TyperベースのFlow Cut CLIエントリーポイント。"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import typer

from src.pipeline import (
    PocRunOptions,
    ensure_audio_files,
    execute_poc_run,
    list_models,
    resolve_models,
)
from src.utils.progress import load_progress

app = typer.Typer(help="Flow Cut コマンドラインインターフェース")


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO, format='[%(levelname)s] %(message)s')


@app.command("models")
def list_available_models() -> None:
    """登録済みの音声認識ランナーを一覧表示する。"""
    for data in list_models():
        typer.echo(f"- {data['slug']:7} | {data['display_name']} | model={data['default_model']}")


@app.command()
def run(
    audio: List[Path] = typer.Argument(None, help='入力音声ファイルへのパス（複数可）。--resume 指定時は省略可'),
    models: Optional[str] = typer.Option(None, help='カンマ区切りのランナー一覧 (例: kotoba,mlx)。未指定なら全ランナー'),
    language: Optional[str] = typer.Option(None, help='言語コード（例: ja, en）。未指定なら自動判定'),
    chunk_size: Optional[int] = typer.Option(None, help='モデルごとのチャンクサイズ上書き'),
    output_dir: Path = typer.Option(Path('temp/poc_samples'), help='結果を書き出すディレクトリ'),
    progress_dir: Path = typer.Option(Path('temp/progress'), help='進捗ファイルの出力ディレクトリ'),
    resume: Optional[Path] = typer.Option(None, help='再開する progress JSON のパス'),
    simulate: bool = typer.Option(True, '--simulate/--no-simulate', help='シミュレーションモードを切り替える'),
    verbose: bool = typer.Option(False, '--verbose', help='詳細ログを有効化'),
) -> None:
    """PoC向けの文字起こしパイプラインを実行する。"""
    _configure_logging(verbose)

    if resume:
        record = load_progress(resume)
        typer.echo(f"[resume] {resume} を読み込み audio={record.audio_file} model={record.model}")
        audio_files = [Path(record.audio_file)]
        model_slugs = [record.model]
    else:
        if not audio:
            raise typer.BadParameter('audio を指定するか --resume を利用してください')
        audio_files = ensure_audio_files(audio)
        try:
            model_slugs = resolve_models(models)
        except ValueError as exc:
            raise typer.BadParameter(str(exc)) from exc

    options = PocRunOptions(
        language=language,
        chunk_size=chunk_size,
        output_dir=output_dir,
        progress_dir=progress_dir,
        simulate=simulate,
        verbose=verbose,
    )
    execute_poc_run(audio_files, model_slugs, options)


if __name__ == '__main__':  # pragma: no cover
    app()

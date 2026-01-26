"""TyperベースのFlow Cut CLIエントリーポイント。"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import typer
from src.pipeline import (
    PocRunOptions,
    ResumeCompletedError,
    ensure_audio_files,
    execute_poc_run,
    list_models,
    prepare_resume_run,
    resolve_models,
)
from src.llm import available_providers as list_llm_providers
from src.llm.profiles import get_profile

app = typer.Typer(help="Flow Cut コマンドラインインターフェース")


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO, format='[%(levelname)s] %(message)s')


@app.command("models")
def list_available_models() -> None:
    """登録済みの音声認識ランナーを一覧表示する。"""
    for data in list_models():
        typer.echo(f"- {data['slug']:7} | {data['display_name']} | model={data['default_model']}")


@app.command("gui")
def launch_gui() -> None:
    """Tkinter ベースの最小GUIを起動する。"""
    try:
        # Tkinter / GUI 関連の import は、このコマンド実行時にのみ行う
        from src.gui.app import run_gui
    except ImportError as exc:  # _tkinter が無いなど
        message = (
            "Tkinter が利用できない Python 環境です（_tkinter モジュールが見つかりません）。\n"
            "macOS では、python.org 版の Python 3 系をインストールしてから、\n"
            "その Python で仮想環境を作り直して実行してください。"
        )
        typer.echo(message, err=True)
        raise typer.Exit(code=1) from exc

    run_gui()


def _normalize_llm_provider(raw: Optional[str]) -> Optional[str]:
    if raw is None:
        return None
    slug = raw.strip().lower()
    if not slug:
        return None
    providers = list_llm_providers()
    if slug not in providers:
        raise typer.BadParameter(f"未登録のLLMプロバイダーです: {slug}. 候補: {providers}")
    return slug


def _normalize_workflow(raw: Optional[str]) -> str:
    """
    ワークフロー名を正規化するヘルパー。
    - None / 空文字は workflow2 にマップ
    - 未知の値はエラーにする
    """
    from src.llm.workflows.registry import list_workflows

    if raw is None:
        return "workflow2"
    slug = raw.strip().lower()
    if not slug:
        return "workflow2"
    allowed = [wf.slug for wf in list_workflows()]
    if slug not in allowed:
        raise typer.BadParameter(f"未対応のワークフローです: {slug}. 候補: {allowed}")
    return slug


@app.command()
def run(
    audio: List[Path] = typer.Argument(None, help='入力音声/動画ファイルへのパス（複数可）。動画の場合は音声を自動抽出。--resume 指定時は省略可'),
    models: Optional[str] = typer.Option(None, help='カンマ区切りのランナー一覧 (例: mlx,whisper-local / faster,whisper-local)。未指定ならOS別のデフォルトランナー'),
    language: Optional[str] = typer.Option(None, help='言語コード（例: ja, en）。未指定なら自動判定'),
    chunk_size: Optional[int] = typer.Option(None, help='モデルごとのチャンクサイズ上書き'),
    output_dir: Path = typer.Option(Path('temp/poc_samples'), help='結果を書き出すディレクトリ'),
    subtitle_dir: Path = typer.Option(Path('output'), help='SRT字幕を書き出すディレクトリ'),
    progress_dir: Path = typer.Option(Path('temp/progress'), help='進捗ファイルの出力ディレクトリ'),
    resume: Optional[Path] = typer.Option(None, help='再開する progress JSON のパス'),
    llm: Optional[str] = typer.Option(None, '--llm', help='使用するLLMプロバイダー（例: openai, google, anthropic）'),
    llm_profile: Optional[str] = typer.Option(None, '--llm-profile', help='使用するLLMプロファイル名（config/llm_profiles.json を参照）'),
    llm_temperature: Optional[float] = typer.Option(None, '--llm-temperature', help='LLM整形時のtemperature。未指定ならプロバイダー既定値'),
    llm_timeout: Optional[float] = typer.Option(None, '--llm-timeout', help='LLM APIリクエストのタイムアウト秒数'),
    rewrite: Optional[bool] = typer.Option(None, '--rewrite/--no-rewrite', help='LLM整形で語尾リライトを有効化する'),
    workflow: Optional[str] = typer.Option(None, '--workflow', help='使用するLLM整形ワークフロー（未指定なら workflow2）'),
    start_delay: float = typer.Option(0.0, '--start-delay', help='テロップ開始時間を遅らせる秒数（例: 0.2）。最初のテロップのstartと最後のendは維持される'),
    keep_audio: bool = typer.Option(False, '--keep-audio', help='動画から抽出した音声ファイルを保存する'),
    simulate: bool = typer.Option(True, '--simulate/--no-simulate', help='シミュレーションモードを切り替える'),
    verbose: bool = typer.Option(False, '--verbose', help='詳細ログを有効化'),
) -> None:
    """PoC向けの文字起こしパイプラインを実行する。"""
    _configure_logging(verbose)

    llm_provider = _normalize_llm_provider(llm)
    workflow_slug = _normalize_workflow(workflow)

    # プロファイルが指定されていれば、パスごとのモデル構成をそこから取得する
    pass1_model = pass2_model = pass3_model = pass4_model = None
    if llm_profile is not None:
        profile = get_profile(llm_profile)
        if profile is None:
            raise typer.BadParameter(f"指定されたLLMプロファイルが見つかりません: {llm_profile}")
        # プロバイダーが明示されていない場合のみ、プロファイル側の provider を採用
        if llm_provider is None:
            llm_provider = profile.provider
        pass1_model = profile.pass1_model
        pass2_model = profile.pass2_model
        pass3_model = profile.pass3_model
        pass4_model = profile.pass4_model

    if llm_provider is None:
        typer.echo("[info] --llm 未指定のため整形とSRT出力をスキップし、文字起こしJSONのみ保存します", err=True)

    base_options = PocRunOptions(
        language=language,
        chunk_size=chunk_size,
        output_dir=output_dir,
        progress_dir=progress_dir,
        subtitle_dir=subtitle_dir,
        simulate=simulate,
        verbose=verbose,
        llm_provider=llm_provider,
        llm_profile=llm_profile,
        workflow=workflow_slug,
        llm_pass1_model=pass1_model,
        llm_pass2_model=pass2_model,
        llm_pass3_model=pass3_model,
        llm_pass4_model=pass4_model,
        rewrite=rewrite,
        llm_temperature=llm_temperature,
        llm_timeout=llm_timeout,
        start_delay=start_delay,
        keep_extracted_audio=keep_audio,
    )
    if resume:
        try:
            record, audio_files, model_slugs, options = prepare_resume_run(
                resume,
                base_options=base_options,
            )
        except ResumeCompletedError as exc:
            typer.echo(str(exc))
            raise typer.Exit(code=0)
        except FileNotFoundError as exc:
            raise typer.BadParameter(str(exc)) from exc
        typer.echo(f"[resume] {resume} を読み込み audio={record.audio_file} model={record.model}")
    else:
        if not audio:
            raise typer.BadParameter('audio を指定するか --resume を利用してください')
        audio_files = ensure_audio_files(audio)
        try:
            model_slugs = resolve_models(models)
        except ValueError as exc:
            raise typer.BadParameter(str(exc)) from exc
        options = base_options

    execute_poc_run(audio_files, model_slugs, options)


@app.command("cleanup")
def cleanup_command(
    days: int = typer.Option(3, help='この日数より古いファイルを削除'),
    paths: List[Path] = typer.Argument(
        None,
        help='掃除対象パス（未指定時は temp/poc_samples, temp/progress, logs）',
    ),
    dry_run: bool = typer.Option(False, '--dry-run', help='削除せず候補のみ表示'),
) -> None:
    """temp / logs の古いファイルをまとめて削除する."""
    try:
        from src.utils.cleanup import cleanup_paths
    except ImportError as exc:  # pragma: no cover - defensive
        raise typer.Exit(code=1) from exc

    default_paths = [Path('temp/poc_samples'), Path('temp/progress'), Path('logs')]
    targets = paths or default_paths
    removed = cleanup_paths(targets, older_than_days=days, dry_run=dry_run)
    action = "would remove" if dry_run else "removed"
    typer.echo(f"{action}: {len(removed)} files")
    for p in removed:
        typer.echo(f"- {p}")


if __name__ == '__main__':  # pragma: no cover
    app()

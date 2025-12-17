"""動画ファイルから音声を抽出するユーティリティ。

ffmpegを使用して動画からWhisperに最適化されたモノラル16kHz WAV音声を抽出する。
"""
from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import Set

logger = logging.getLogger(__name__)

# 対応する動画フォーマット
VIDEO_EXTENSIONS: Set[str] = {".mp4", ".mov", ".mkv", ".avi", ".webm"}

# 対応する音声フォーマット（動画ではないもの）
AUDIO_EXTENSIONS: Set[str] = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac"}


def is_video_file(path: Path) -> bool:
    """パスが対応動画フォーマットかどうかを判定する。

    Args:
        path: 判定するファイルパス

    Returns:
        動画ファイルの場合はTrue
    """
    return path.suffix.lower() in VIDEO_EXTENSIONS


def is_audio_file(path: Path) -> bool:
    """パスが音声フォーマットかどうかを判定する。

    Args:
        path: 判定するファイルパス

    Returns:
        音声ファイルの場合はTrue
    """
    return path.suffix.lower() in AUDIO_EXTENSIONS


def get_extracted_audio_path(video_path: Path, output_dir: Path | None = None) -> Path:
    """抽出先の音声ファイルパスを返す（実際に抽出はしない）。

    Args:
        video_path: 動画ファイルパス
        output_dir: 出力先ディレクトリ。Noneの場合は動画と同じディレクトリ

    Returns:
        抽出先音声ファイルパス（{動画名}_audio.wav）
    """
    target_dir = output_dir or video_path.parent
    audio_filename = f"{video_path.stem}_audio.wav"
    return target_dir / audio_filename


class AudioExtractionError(Exception):
    """音声抽出時のエラー。"""


def extract_audio_from_video(
    video_path: Path,
    output_dir: Path | None = None,
    *,
    overwrite: bool = False,
) -> Path:
    """動画からモノラル16kHz WAV音声を抽出する。

    Args:
        video_path: 入力動画ファイルパス
        output_dir: 出力先ディレクトリ。Noneの場合は動画と同じディレクトリ
        overwrite: 既存ファイルを上書きするかどうか

    Returns:
        抽出された音声ファイルパス

    Raises:
        AudioExtractionError: 抽出に失敗した場合
        FileNotFoundError: 動画ファイルが見つからない場合
    """
    if not video_path.exists():
        raise FileNotFoundError(f"動画ファイルが見つかりません: {video_path}")

    if not is_video_file(video_path):
        raise AudioExtractionError(
            f"対応していないファイル形式です: {video_path.suffix}. "
            f"対応形式: {', '.join(sorted(VIDEO_EXTENSIONS))}"
        )

    output_path = get_extracted_audio_path(video_path, output_dir)

    # 既存ファイルのチェック
    if output_path.exists() and not overwrite:
        logger.info("既存の抽出済み音声を使用します: %s", output_path)
        return output_path

    # 出力ディレクトリを作成
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ffmpegコマンドを構築
    # -vn: 動画ストリームを無視
    # -acodec pcm_s16le: 16bit PCM
    # -ar 16000: 16kHz サンプルレート
    # -ac 1: モノラル
    # -y: 上書き許可
    cmd = [
        "ffmpeg",
        "-i", str(video_path),
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        "-y",
        str(output_path),
    ]

    logger.info("音声抽出を開始: %s -> %s", video_path.name, output_path.name)

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
        )

        if result.returncode != 0:
            error_msg = result.stderr or result.stdout or "Unknown error"
            raise AudioExtractionError(
                f"ffmpegでの音声抽出に失敗しました: {error_msg}"
            )

        if not output_path.exists():
            raise AudioExtractionError(
                f"音声ファイルが生成されませんでした: {output_path}"
            )

        logger.info("音声抽出完了: %s (%.2f MB)", output_path.name, output_path.stat().st_size / (1024 * 1024))
        return output_path

    except FileNotFoundError:
        raise AudioExtractionError(
            "ffmpegが見つかりません。ffmpegがインストールされていることを確認してください。"
        )
    except subprocess.SubprocessError as exc:
        raise AudioExtractionError(f"ffmpeg実行中にエラーが発生しました: {exc}") from exc


def cleanup_extracted_audio(audio_path: Path) -> None:
    """抽出した音声ファイルを削除する。

    Args:
        audio_path: 削除する音声ファイルパス
    """
    try:
        if audio_path.exists() and audio_path.name.endswith("_audio.wav"):
            audio_path.unlink()
            logger.info("抽出した音声ファイルを削除しました: %s", audio_path)
    except Exception as exc:
        logger.warning("音声ファイルの削除に失敗しました: %s (%s)", audio_path, exc)


__all__ = [
    "VIDEO_EXTENSIONS",
    "AUDIO_EXTENSIONS",
    "AudioExtractionError",
    "is_video_file",
    "is_audio_file",
    "get_extracted_audio_path",
    "extract_audio_from_video",
    "cleanup_extracted_audio",
]

"""音声抽出ユーティリティのテスト。"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch, MagicMock
import subprocess

import pytest

from src.utils.audio_extractor import (
    VIDEO_EXTENSIONS,
    AUDIO_EXTENSIONS,
    AudioExtractionError,
    is_video_file,
    is_audio_file,
    get_extracted_audio_path,
    extract_audio_from_video,
    cleanup_extracted_audio,
)


class TestIsVideoFile:
    """is_video_file関数のテスト。"""

    @pytest.mark.parametrize("extension", VIDEO_EXTENSIONS)
    def test_video_extensions_return_true(self, extension: str) -> None:
        """対応動画フォーマットはTrueを返す。"""
        path = Path(f"test{extension}")
        assert is_video_file(path) is True

    @pytest.mark.parametrize("extension", [".MP4", ".MOV", ".MKV"])
    def test_uppercase_extensions_return_true(self, extension: str) -> None:
        """大文字の拡張子もTrueを返す。"""
        path = Path(f"test{extension}")
        assert is_video_file(path) is True

    @pytest.mark.parametrize("extension", AUDIO_EXTENSIONS)
    def test_audio_extensions_return_false(self, extension: str) -> None:
        """音声フォーマットはFalseを返す。"""
        path = Path(f"test{extension}")
        assert is_video_file(path) is False

    def test_unknown_extension_returns_false(self) -> None:
        """未知の拡張子はFalseを返す。"""
        path = Path("test.xyz")
        assert is_video_file(path) is False


class TestIsAudioFile:
    """is_audio_file関数のテスト。"""

    @pytest.mark.parametrize("extension", AUDIO_EXTENSIONS)
    def test_audio_extensions_return_true(self, extension: str) -> None:
        """対応音声フォーマットはTrueを返す。"""
        path = Path(f"test{extension}")
        assert is_audio_file(path) is True

    @pytest.mark.parametrize("extension", VIDEO_EXTENSIONS)
    def test_video_extensions_return_false(self, extension: str) -> None:
        """動画フォーマットはFalseを返す。"""
        path = Path(f"test{extension}")
        assert is_audio_file(path) is False


class TestGetExtractedAudioPath:
    """get_extracted_audio_path関数のテスト。"""

    def test_default_output_dir(self) -> None:
        """デフォルトでは動画と同じディレクトリに出力される。"""
        video_path = Path("/path/to/video.mp4")
        result = get_extracted_audio_path(video_path)
        assert result == Path("/path/to/video_audio.wav")

    def test_custom_output_dir(self, tmp_path: Path) -> None:
        """出力ディレクトリを指定できる。"""
        video_path = Path("/path/to/video.mp4")
        result = get_extracted_audio_path(video_path, tmp_path)
        assert result == tmp_path / "video_audio.wav"


class TestExtractAudioFromVideo:
    """extract_audio_from_video関数のテスト。"""

    def test_file_not_found_raises_error(self, tmp_path: Path) -> None:
        """存在しないファイルはエラー。"""
        video_path = tmp_path / "nonexistent.mp4"
        with pytest.raises(FileNotFoundError):
            extract_audio_from_video(video_path)

    def test_unsupported_format_raises_error(self, tmp_path: Path) -> None:
        """非対応フォーマットはエラー。"""
        video_path = tmp_path / "test.xyz"
        video_path.write_text("dummy")
        with pytest.raises(AudioExtractionError, match="対応していないファイル形式"):
            extract_audio_from_video(video_path)

    def test_existing_file_is_reused(self, tmp_path: Path) -> None:
        """既存の抽出済みファイルがあれば再利用する。"""
        video_path = tmp_path / "test.mp4"
        video_path.write_text("dummy video")
        
        audio_path = tmp_path / "test_audio.wav"
        audio_path.write_text("dummy audio")
        
        result = extract_audio_from_video(video_path, overwrite=False)
        assert result == audio_path

    @patch("src.utils.audio_extractor.subprocess.run")
    def test_ffmpeg_called_with_correct_args(self, mock_run: MagicMock, tmp_path: Path) -> None:
        """ffmpegが正しい引数で呼び出される。"""
        video_path = tmp_path / "test.mp4"
        video_path.write_text("dummy video")
        
        output_path = tmp_path / "test_audio.wav"
        
        # ffmpegが成功したことをシミュレート
        def side_effect(*args, **kwargs):
            output_path.write_text("extracted audio")
            return MagicMock(returncode=0)
        
        mock_run.side_effect = side_effect
        
        result = extract_audio_from_video(video_path, overwrite=True)
        
        assert result == output_path
        mock_run.assert_called_once()
        
        call_args = mock_run.call_args[0][0]
        assert call_args[0] == "ffmpeg"
        assert "-vn" in call_args
        assert "-ar" in call_args
        assert "16000" in call_args
        assert "-ac" in call_args
        assert "1" in call_args


class TestCleanupExtractedAudio:
    """cleanup_extracted_audio関数のテスト。"""

    def test_deletes_extracted_audio_file(self, tmp_path: Path) -> None:
        """抽出した音声ファイルを削除する。"""
        audio_path = tmp_path / "test_audio.wav"
        audio_path.write_text("dummy")
        
        cleanup_extracted_audio(audio_path)
        
        assert not audio_path.exists()

    def test_ignores_non_audio_wav_files(self, tmp_path: Path) -> None:
        """_audio.wavでないファイルは削除しない。"""
        other_path = tmp_path / "test.wav"
        other_path.write_text("dummy")
        
        cleanup_extracted_audio(other_path)
        
        assert other_path.exists()

    def test_handles_nonexistent_file(self, tmp_path: Path) -> None:
        """存在しないファイルでもエラーにならない。"""
        audio_path = tmp_path / "nonexistent_audio.wav"
        cleanup_extracted_audio(audio_path)  # エラーが発生しない

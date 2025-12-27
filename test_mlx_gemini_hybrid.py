#!/usr/bin/env python3
"""
MLX Whisper + Gemini ハイブリッド文字起こしテストスクリプト。

MLX Whisperで高速に文字起こしし、Geminiで補正、SRTファイルとして出力します。
"""
import os
import sys
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.transcribe.base import TranscriptionConfig, get_runner
from src.transcribe.hybrid import HybridProcessor
from src.alignment.srt import SubtitleSegment, segments_to_srt
from src.utils.audio_extractor import extract_audio_from_video, is_video_file


def create_segments_from_words(words, max_words_per_segment=10):
    """WordTimestampリストから字幕セグメントを生成する。

    Args:
        words: WordTimestampのリスト
        max_words_per_segment: 1セグメント当たりの最大単語数

    Returns:
        SubtitleSegmentのリスト
    """
    segments = []
    index = 1

    i = 0
    while i < len(words):
        segment_words = words[i:i + max_words_per_segment]
        if not segment_words:
            break

        text = "".join(w.word for w in segment_words)
        start = segment_words[0].start
        end = segment_words[-1].end

        segments.append(SubtitleSegment(
            index=index,
            start=start,
            end=end,
            text=text
        ))

        index += 1
        i += max_words_per_segment

    return segments


def main():
    video_path = Path("/Users/shotaro/code/client/FlowCut/docs/sample/videoplayback.mp4 のコピー.mp4")

    if not video_path.exists():
        print(f"エラー: 動画ファイルが見つかりません: {video_path}")
        return 1

    print(f"動画ファイル: {video_path}")
    print(f"ファイルサイズ: {video_path.stat().st_size / (1024*1024):.2f} MB")

    output_dir = Path("output/mlx_gemini_hybrid_test")
    output_dir.mkdir(parents=True, exist_ok=True)

    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    # Google API Keyを確認
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("エラー: GOOGLE_API_KEY環境変数が設定されていません")
        return 1

    # Step 1: 動画から音声を抽出
    print("\n[1/4] 動画から音声を抽出中...")
    try:
        if is_video_file(video_path):
            audio_path = extract_audio_from_video(
                video_path,
                output_dir=output_dir,
                overwrite=False
            )
            print(f"  - 音声抽出完了: {audio_path}")
        else:
            audio_path = video_path
            print(f"  - 音声ファイルを使用: {audio_path}")
    except Exception as e:
        print(f"エラー: 音声抽出に失敗しました: {e}")
        return 1

    # Step 2: MLX Whisperで文字起こし
    print("\n[2/4] MLX Whisperで文字起こし中...")
    runner = get_runner("mlx")
    config = TranscriptionConfig(language="ja")

    try:
        whisper_result = runner.transcribe(audio_path, config)
        print(f"  - Whisper完了: {len(whisper_result.words)} 単語")
        print(f"  - テキスト長: {len(whisper_result.text)} 文字")
    except Exception as e:
        print(f"エラー: Whisper文字起こしに失敗しました: {e}")
        return 1

    # Step 3: Geminiと統合
    print("\n[3/4] GeminiとWhisperの結果を統合中...")

    try:
        processor = HybridProcessor.create_for_use_case(
            use_case="standard",
            api_key=api_key,
            language="ja"
        )

        def progress_callback(message, percent):
            print(f"  - {message} ({percent}%)")

        hybrid_result = processor.process(
            audio_path=audio_path,
            whisper_result=whisper_result,
            progress_callback=progress_callback,
            log_dir=log_dir,
            run_id="mlx_gemini_test"
        )

        print(f"  - 統合完了: {len(hybrid_result.words)} 単語")
        print(f"  - テキスト長: {len(hybrid_result.text)} 文字")

        if "hybrid_processing" in hybrid_result.metadata:
            hp_meta = hybrid_result.metadata["hybrid_processing"]
            print(f"  - Geminiモデル: {hp_meta.get('gemini_model')}")
            print(f"  - Gemini採用ブロック数: {hp_meta.get('blocks_using_gemini')}")
            print(f"  - Whisper採用ブロック数: {hp_meta.get('blocks_using_whisper')}")

    except Exception as e:
        print(f"警告: ハイブリッド処理に失敗しました: {e}")
        print("Whisperのみの結果を使用します")
        hybrid_result = whisper_result

    # Step 4: SRTファイルを生成
    print("\n[4/4] SRTファイルを生成中...")

    segments = create_segments_from_words(hybrid_result.words, max_words_per_segment=10)
    print(f"  - セグメント数: {len(segments)}")

    srt_content = segments_to_srt(segments)

    srt_path = output_dir / "mlx_gemini_output.srt"
    srt_path.write_text(srt_content, encoding="utf-8")
    print(f"  - SRTファイル保存: {srt_path}")

    text_path = output_dir / "mlx_gemini_output.txt"
    text_path.write_text(hybrid_result.text, encoding="utf-8")
    print(f"  - テキストファイル保存: {text_path}")

    print(f"\n完了！出力ディレクトリ: {output_dir}")
    print(f"ログディレクトリ: {log_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

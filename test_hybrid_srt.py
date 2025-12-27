#!/usr/bin/env python3
"""
GeminiとWhisperのハイブリッド文字起こしテストスクリプト。

指定された動画ファイルをWhisperとGeminiで文字起こしし、
統合結果をSRTファイルとして出力します。
"""
import os
import sys
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.transcribe.base import TranscriptionConfig, get_runner
from src.transcribe.hybrid import HybridProcessor
from src.alignment.srt import SubtitleSegment, segments_to_srt


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
        # 最大単語数分の単語を取得
        segment_words = words[i:i + max_words_per_segment]
        if not segment_words:
            break

        # セグメントのテキスト、開始時刻、終了時刻を計算
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
    # 動画ファイルのパス
    video_path = Path("/Users/shotaro/code/client/FlowCut/docs/sample/videoplayback.mp4 のコピー.mp4")

    if not video_path.exists():
        print(f"エラー: 動画ファイルが見つかりません: {video_path}")
        return 1

    print(f"動画ファイル: {video_path}")
    print(f"ファイルサイズ: {video_path.stat().st_size / (1024*1024):.2f} MB")

    # 出力ディレクトリを作成
    output_dir = Path("output/hybrid_test")
    output_dir.mkdir(parents=True, exist_ok=True)

    # ログディレクトリを作成
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    # Google API Keyを環境変数から取得
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("警告: GOOGLE_API_KEY環境変数が設定されていません")
        print("Geminiとの統合処理はスキップされ、Whisperのみの結果が使用されます")

    # Step 1: Whisperで文字起こし
    print("\n[1/3] Whisperで文字起こし中...")
    runner = get_runner("whisper-local")
    config = TranscriptionConfig(language="ja")

    try:
        whisper_result = runner.transcribe(video_path, config)
        print(f"  - Whisper完了: {len(whisper_result.words)} 単語")
        print(f"  - テキスト長: {len(whisper_result.text)} 文字")
    except Exception as e:
        print(f"エラー: Whisper文字起こしに失敗しました: {e}")
        return 1

    # Step 2: Geminiと統合
    print("\n[2/3] GeminiとWhisperの結果を統合中...")

    try:
        # HybridProcessorを作成（標準設定）
        processor = HybridProcessor.create_for_use_case(
            use_case="standard",
            api_key=api_key,
            language="ja"
        )

        def progress_callback(message, percent):
            print(f"  - {message} ({percent}%)")

        # ハイブリッド処理を実行
        hybrid_result = processor.process(
            audio_path=video_path,
            whisper_result=whisper_result,
            progress_callback=progress_callback,
            log_dir=log_dir,
            run_id="test"
        )

        print(f"  - 統合完了: {len(hybrid_result.words)} 単語")
        print(f"  - テキスト長: {len(hybrid_result.text)} 文字")

        # メタデータを表示
        if "hybrid_processing" in hybrid_result.metadata:
            hp_meta = hybrid_result.metadata["hybrid_processing"]
            print(f"  - Geminiモデル: {hp_meta.get('gemini_model')}")
            print(f"  - Gemini採用ブロック数: {hp_meta.get('blocks_using_gemini')}")
            print(f"  - Whisper採用ブロック数: {hp_meta.get('blocks_using_whisper')}")

    except Exception as e:
        print(f"警告: ハイブリッド処理に失敗しました: {e}")
        print("Whisperのみの結果を使用します")
        hybrid_result = whisper_result

    # Step 3: SRTファイルを生成
    print("\n[3/3] SRTファイルを生成中...")

    # 単語からセグメントを作成（1セグメントあたり10単語）
    segments = create_segments_from_words(hybrid_result.words, max_words_per_segment=10)
    print(f"  - セグメント数: {len(segments)}")

    # SRTテキストを生成
    srt_content = segments_to_srt(segments)

    # SRTファイルを保存
    srt_path = output_dir / "hybrid_output.srt"
    srt_path.write_text(srt_content, encoding="utf-8")
    print(f"  - SRTファイル保存: {srt_path}")

    # 統合テキストも保存
    text_path = output_dir / "hybrid_output.txt"
    text_path.write_text(hybrid_result.text, encoding="utf-8")
    print(f"  - テキストファイル保存: {text_path}")

    print(f"\n完了！出力ディレクトリ: {output_dir}")
    print(f"ログディレクトリ: {log_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

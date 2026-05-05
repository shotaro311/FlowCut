#!/usr/bin/env python3
from __future__ import annotations

import argparse
import bisect
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
import importlib
import json
import logging
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import uuid
from typing import Any, Callable, Dict, Iterable, List, Literal, Sequence

logger = logging.getLogger(__name__)

try:  # pragma: no cover - 環境に応じてロード
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass


class FormatterError(RuntimeError):
    pass


class TranscriptionError(RuntimeError):
    pass


def generate_sequential_path(base_path: Path) -> Path:
    if not base_path.exists():
        return base_path

    directory = base_path.parent
    stem = base_path.stem
    suffix = base_path.suffix

    if not suffix:
        stem = base_path.name
        suffix = ""

    index = 1
    while True:
        candidate = directory / f"{stem} ({index}){suffix}"
        if not candidate.exists():
            return candidate
        index += 1


def _format_timestamp(seconds: float) -> str:
    total_ms = int(round(max(seconds, 0.0) * 1000))
    ms = total_ms % 1000
    total_sec = total_ms // 1000
    s = total_sec % 60
    total_min = total_sec // 60
    m = total_min % 60
    h = total_min // 60
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


@dataclass(slots=True)
class SubtitleSegment:
    index: int
    start: float
    end: float
    text: str

    def to_srt_block(self) -> str:
        return f"{self.index}\n{_format_timestamp(self.start)} --> {_format_timestamp(self.end)}\n{self.text}\n"


def segments_to_srt(segments: Iterable[SubtitleSegment]) -> str:
    blocks = [seg.to_srt_block().strip() for seg in segments]
    return "\n\n".join(blocks) + ("\n" if blocks else "")


@dataclass(slots=True)
class WordTimestamp:
    word: str
    start: float
    end: float
    confidence: float | None = None

    def to_dict(self) -> dict:
        return {
            "word": self.word,
            "start": self.start,
            "end": self.end,
            "confidence": self.confidence,
        }


@dataclass(slots=True)
class TranscriptionResult:
    text: str
    words: List[WordTimestamp]
    metadata: Dict[str, Any]

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "words": [w.to_dict() for w in self.words],
            "metadata": self.metadata,
        }


VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm"}
AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac"}


def is_video_file(path: Path) -> bool:
    return path.suffix.lower() in VIDEO_EXTENSIONS


class AudioExtractionError(Exception):
    pass


def get_extracted_audio_path(video_path: Path, output_dir: Path | None = None) -> Path:
    target_dir = output_dir or video_path.parent
    return target_dir / f"{video_path.stem}_audio.wav"


def extract_audio_from_video(
    video_path: Path,
    output_dir: Path | None = None,
    *,
    overwrite: bool = False,
) -> Path:
    if not video_path.exists():
        raise FileNotFoundError(f"動画ファイルが見つかりません: {video_path}")

    if not is_video_file(video_path):
        raise AudioExtractionError(
            f"対応していないファイル形式です: {video_path.suffix}. "
            f"対応形式: {', '.join(sorted(VIDEO_EXTENSIONS))}"
        )

    output_path = get_extracted_audio_path(video_path, output_dir)
    if output_path.exists() and not overwrite:
        logger.info("既存の抽出済み音声を使用します: %s", output_path)
        return output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-i",
        str(video_path),
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "16000",
        "-ac",
        "1",
        "-y",
        str(output_path),
    ]

    logger.info("音声抽出を開始: %s -> %s", video_path.name, output_path.name)
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    except FileNotFoundError as exc:
        raise AudioExtractionError("ffmpegが見つかりません。ffmpegがインストールされていることを確認してください。") from exc
    except subprocess.SubprocessError as exc:
        raise AudioExtractionError(f"ffmpeg実行中にエラーが発生しました: {exc}") from exc

    if result.returncode != 0:
        error_msg = result.stderr or result.stdout or "Unknown error"
        raise AudioExtractionError(f"ffmpegでの音声抽出に失敗しました: {error_msg}")
    if not output_path.exists():
        raise AudioExtractionError(f"音声ファイルが生成されませんでした: {output_path}")
    return output_path


def cleanup_extracted_audio(audio_path: Path) -> None:
    try:
        if audio_path.exists() and audio_path.name.endswith("_audio.wav"):
            audio_path.unlink()
    except Exception as exc:
        logger.warning("音声ファイルの削除に失敗しました: %s (%s)", audio_path, exc)


_OPENAI_WHISPER_MODEL_CACHE: Dict[str, Any] = {}
_FASTER_WHISPER_MODEL_CACHE: Dict[tuple[str, str, str], Any] = {}


def _get_whisper_model(model_name: str):
    try:
        import whisper  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise TranscriptionError("Pythonで whisper を import できません（openai-whisper をインストールしてください）") from exc
    if model_name not in _OPENAI_WHISPER_MODEL_CACHE:
        _OPENAI_WHISPER_MODEL_CACHE[model_name] = whisper.load_model(model_name)
    return _OPENAI_WHISPER_MODEL_CACHE[model_name]


def _load_mlx_whisper():
    try:
        return importlib.import_module("mlx_whisper")
    except ImportError as exc:
        raise TranscriptionError(f"mlx-whisper のロードに失敗しました: {exc!r}") from exc


def _resolve_mlx_model_id(model: str) -> str:
    raw = str(model or "").strip()
    if not raw:
        return "mlx-community/whisper-large-v3-mlx"
    if "/" in raw:
        return raw
    # e.g. large-v3 -> mlx-community/whisper-large-v3-mlx
    return f"mlx-community/whisper-{raw}-mlx"


def _parse_mlx_words(words: Sequence[Dict[str, Any]]) -> List[WordTimestamp]:
    parsed: List[WordTimestamp] = []
    for w in words or []:
        text = str(w.get("word", "")).strip()
        if not text:
            continue
        try:
            start = float(w.get("start"))
            end = float(w.get("end"))
        except (TypeError, ValueError):
            continue
        confidence = w.get("probability") or w.get("confidence")
        try:
            conf_val = float(confidence) if confidence is not None else None
        except (TypeError, ValueError):
            conf_val = None
        parsed.append(WordTimestamp(word=text, start=start, end=end, confidence=conf_val))
    return parsed


def transcribe_mlx_whisper_local(
    audio_path: Path,
    *,
    language: str | None,
    model: str,
) -> TranscriptionResult:
    mlx_whisper = _load_mlx_whisper()
    if not audio_path.exists():
        raise FileNotFoundError(f"音声ファイルが見つかりません: {audio_path}")

    model_id = _resolve_mlx_model_id(model)
    logger.info("[mlx-whisper] model=%s language=%s", model_id, language or "auto")
    try:
        output = mlx_whisper.transcribe(
            str(audio_path),
            path_or_hf_repo=model_id,
            word_timestamps=True,
            language=language,
            verbose=False,
            temperature=0.0,
            condition_on_previous_text=False,
        )
    except Exception as exc:  # pragma: no cover
        raise TranscriptionError(f"mlx-whisper 実行に失敗しました: {exc}") from exc

    text = str(output.get("text", ""))
    segments = output.get("segments") or []
    words: List[WordTimestamp] = []
    for seg in segments:
        words.extend(_parse_mlx_words(seg.get("words", [])))
    if not words and output.get("words"):
        words = _parse_mlx_words(output.get("words") or [])

    metadata: Dict[str, Any] = {
        "runner": "mlx",
        "model": model_id,
        "local": True,
        "audio_file": str(audio_path),
        "language": output.get("language") or language,
    }
    return TranscriptionResult(text=text, words=words, metadata=metadata)


def _load_faster_whisper():
    try:
        return importlib.import_module("faster_whisper")
    except ImportError as exc:
        raise TranscriptionError(f"faster-whisper のロードに失敗しました: {exc!r}") from exc


def _get_faster_whisper_model(*, model_size: str, device: str, compute_type: str):
    key = (model_size, device, compute_type)
    if key in _FASTER_WHISPER_MODEL_CACHE:
        return _FASTER_WHISPER_MODEL_CACHE[key]
    faster_whisper = _load_faster_whisper()
    WhisperModel = getattr(faster_whisper, "WhisperModel", None)
    if WhisperModel is None:  # pragma: no cover - defensive
        raise TranscriptionError("faster_whisper.WhisperModel が見つかりません")
    _FASTER_WHISPER_MODEL_CACHE[key] = WhisperModel(model_size, device=device, compute_type=compute_type)
    return _FASTER_WHISPER_MODEL_CACHE[key]


def transcribe_faster_whisper_local(
    audio_path: Path,
    *,
    language: str | None,
    model: str,
    device: str | None = None,
    compute_type: str | None = None,
) -> TranscriptionResult:
    if not audio_path.exists():
        raise FileNotFoundError(f"音声ファイルが見つかりません: {audio_path}")

    model_size = str(model or "").strip() or "large-v3"
    resolved_device = str(device or os.getenv("FASTER_WHISPER_DEVICE", "cpu")).strip() or "cpu"
    resolved_compute = str(compute_type or os.getenv("FASTER_WHISPER_COMPUTE_TYPE", "int8")).strip() or "int8"

    logger.info("[faster-whisper] model=%s device=%s compute_type=%s language=%s", model_size, resolved_device, resolved_compute, language or "auto")
    model_obj = _get_faster_whisper_model(model_size=model_size, device=resolved_device, compute_type=resolved_compute)

    try:
        segments, info = model_obj.transcribe(
            str(audio_path),
            word_timestamps=True,
            language=language,
        )
    except Exception as exc:  # pragma: no cover
        raise TranscriptionError(f"faster-whisper 実行に失敗しました: {exc}") from exc

    text_parts: List[str] = []
    words: List[WordTimestamp] = []
    for seg in segments:
        seg_text = getattr(seg, "text", None)
        if isinstance(seg_text, str) and seg_text:
            text_parts.append(seg_text)
        for w in getattr(seg, "words", None) or []:
            word_text = str(getattr(w, "word", "")).strip()
            if not word_text:
                continue
            try:
                start = float(getattr(w, "start"))
                end = float(getattr(w, "end"))
            except (TypeError, ValueError):
                continue
            prob = getattr(w, "probability", None)
            try:
                conf_val = float(prob) if prob is not None else None
            except (TypeError, ValueError):
                conf_val = None
            words.append(WordTimestamp(word=word_text, start=start, end=end, confidence=conf_val))

    text = "".join(text_parts).strip() or "".join((w.word for w in words))
    metadata: Dict[str, Any] = {
        "runner": "faster",
        "model": model_size,
        "device": resolved_device,
        "compute_type": resolved_compute,
        "local": True,
        "audio_file": str(audio_path),
        "language": getattr(info, "language", None) if info is not None else language,
    }
    return TranscriptionResult(text=text, words=words, metadata=metadata)


def _parse_words_from_segments(segments: List[Dict[str, Any]]) -> List[WordTimestamp]:
    words: List[WordTimestamp] = []
    for seg in segments or []:
        for w in seg.get("words") or []:
            text = str(w.get("word", "")).strip()
            if not text:
                continue
            try:
                start = float(w.get("start"))
                end = float(w.get("end"))
            except (TypeError, ValueError):
                continue
            confidence = w.get("confidence")
            try:
                conf_val = float(confidence) if confidence is not None else None
            except (TypeError, ValueError):
                conf_val = None
            words.append(WordTimestamp(word=text, start=start, end=end, confidence=conf_val))
    return words


def transcribe_openai_whisper_local(
    audio_path: Path,
    *,
    language: str | None,
    model_name: str = "large-v3",
) -> TranscriptionResult:
    if not audio_path.exists():
        raise FileNotFoundError(f"音声ファイルが見つかりません: {audio_path}")

    model = _get_whisper_model(model_name)
    logger.info("[openai-whisper-local] model=%s language=%s", model_name, language or "auto")
    try:
        result: Dict[str, Any] = model.transcribe(
            str(audio_path),
            language=language,
            word_timestamps=True,
        )
    except Exception as exc:  # pragma: no cover
        raise TranscriptionError(f"openai-whisper ローカル実行に失敗しました: {exc}") from exc

    text = str(result.get("text", ""))
    segments = result.get("segments") or []
    words = _parse_words_from_segments(segments)
    metadata: Dict[str, Any] = {
        "runner": "openai",
        "model": model_name,
        "local": True,
        "audio_file": str(audio_path),
        "language": result.get("language") or language,
    }
    return TranscriptionResult(text=text, words=words, metadata=metadata)


DEFAULT_GLOSSARY_TERMS: List[str] = [
    "菅義偉",
    "岸田文雄",
    "安倍晋三",
    "小池百合子",
    "立花孝志",
    "石破茂",
    "松野博一",
    "神谷宗幣",
    "小泉進次郎",
    "榛葉賀津也",
    "木原誠二",
    "高市早苗",
    "河合ゆうすけ",
    "大津力",
    "門田隆将",
    "北野裕子",
    "北村晴男",
    "公明党",
]


def normalize_glossary_terms(terms: Iterable[str] | None) -> List[str]:
    if terms is None:
        return []
    seen: set[str] = set()
    normalized: List[str] = []
    for raw in terms:
        term = str(raw).strip()
        if not term:
            continue
        if term in seen:
            continue
        seen.add(term)
        normalized.append(term)
    return normalized


MAX_LINE_DURATION_SEC = 10.0
MAX_GAP_DURATION_SEC = MAX_LINE_DURATION_SEC


def build_indexed_words(words: Sequence[WordTimestamp]) -> str:
    return "\n".join(f"{i}: {w.word}" for i, w in enumerate(words))


def build_pass4_prompt(line: "LineRange", words: Sequence[WordTimestamp], max_chars: int) -> str:
    def _format_sec(value: object) -> str:
        try:
            return f"{float(value):.2f}"
        except (TypeError, ValueError):
            return "?"

    indexed = "\n".join(
        f"[{i}] {w.word} (time: {_format_sec(getattr(w, 'start', None))}-{_format_sec(getattr(w, 'end', None))}s)"
        for i, w in enumerate(words[line.start_idx : line.end_idx + 1], start=line.start_idx)
    )
    start = getattr(words[line.start_idx], "start", None) if 0 <= line.start_idx < len(words) else None
    end = getattr(words[line.end_idx], "end", None) if 0 <= line.end_idx < len(words) else None
    duration = None
    if start is not None and end is not None:
        try:
            duration = float(end) - float(start)
        except (TypeError, ValueError):
            duration = None
    line_time = f"{_format_sec(start)}-{_format_sec(end)} (duration={_format_sec(duration)}s)"
    return (
        "# Role\n"
        "あなたはテロップ最終チェックの追加ステップ担当です。与えられた行に対してのみ、条件を満たす複数行に必要最小限で分割してください。\n\n"
        "# Constraints\n"
        f"- 必ず1行あたり全角5〜{int(max_chars)}文字に収めること\n"
        f"- 1行の時間幅（end-start）が{MAX_LINE_DURATION_SEC:.1f}秒を超える場合は必ず分割\n"
        "- 語順を変えない、語を追加/削除しない\n"
        "- 要約・翻訳・意訳をしない\n"
        "- 行末の句読点（、。）は削除。文中の句読点は残してよい\n"
        "- 改行の優先度: (1)「。?!」直後 → (2)「、」直後 → (3) 接続助詞・係助詞など自然な切れ目。\n\n"
        "# Input\n"
        f"対象行のインデックス範囲: from={line.start_idx}, to={line.end_idx}\n"
        f"対象の行テキスト:\n{line.text}\n\n"
        f"対象行の時間情報（参考）:\n{line_time}\n\n"
        f"単語リスト（[インデックス] 単語 (time: 開始-終了)）:\n{indexed}\n\n"
        "# Output\n"
        "以下のJSONだけを返してください（説明・コードフェンス禁止）。\n"
        "**重要**: from/toは単語リストのインデックス番号（上記の[角括弧内の数字]）を指定してください。時間（秒）ではありません。\n\n"
        f"例（対象行が from={line.start_idx}, to={line.end_idx} の場合）:\n"
        "{\n"
        '  "lines": [\n'
        f'    {{"from": {line.start_idx}, "to": {line.start_idx + 5}, "text": "...."}},\n'
        f'    {{"from": {line.start_idx + 6}, "to": {line.end_idx}, "text": "...."}}\n'
        "  ]\n"
        "}\n"
    )


def build_pass1_prompt(raw_text: str, words: Sequence[WordTimestamp], glossary_terms: Sequence[str]) -> str:
    indexed = build_indexed_words(words)
    glossary_text = "\n".join(glossary_terms or [])
    return (
        "# Role\n"
        "あなたはプロの字幕エディターです。\n"
        "以下の単語列（index付き）を、語順を変えずに**最小限**で校正してください。\n\n"
        "# 目的（この順で優先）\n"
        "1. 誤字・脱字を修正\n"
        "2. 固有名詞（人名・地名・組織名）を、Glossary と照らし合わせて正しい表記に揃える\n"
        "3. 政治関連用語（政党名・法案名・政策名など）は、一般に使われる公式表記に統一（例: Wikipedia等）\n"
        "   - ただし確信がない場合は変更しない（誤修正を避ける）\n\n"
        "# 許可される操作（JSON operations）\n"
        "- replace: 誤変換/誤字を正しい表記に置換（必要なら複数単語を1つにまとめて置換してよい）\n"
        "- delete: 明らかなノイズ（フィラー・重複）を削除\n\n"
        "# 禁止（厳守）\n"
        "- insert（音声に無い単語を追加しない）\n"
        "- 並び替え、要約、意訳\n\n"
        "# Glossary（最優先）\n"
        "Glossary にある表記が正解です。該当する場合は必ず Glossary 表記に揃えてください。\n"
        f"{glossary_text}\n\n"
        "# Input\n"
        f"元のテキスト:\n{raw_text}\n\n"
        f"単語リスト（index:word）:\n{indexed}\n\n"
        "# Output\n"
        "以下のJSONのみを返してください（説明文・コードフェンス禁止）:\n"
        "{\n"
        '  "operations": [\n'
        '    {"type": "replace", "start_idx": 10, "end_idx": 11, "text": "菅義偉"},\n'
        '    {"type": "delete", "start_idx": 25, "end_idx": 25}\n'
        "  ]\n"
        "}\n"
        '操作が不要なら {"operations": []}\n'
    )


def build_pass2_prompt(words: Sequence[WordTimestamp], max_chars: float) -> str:
    indexed = build_indexed_words(words)
    return (
        "# Role\n"
        "あなたは熟練の動画テロップ編集者です。\n"
        "提供されたテキストを、視聴者が最も読みやすいリズムで読めるように、以下の【思考ワークフロー】に従って処理し、行のインデックス範囲を JSON で返してください。\n\n"
        "# Constraints (制約)\n"
        f"- 1行の最大文字数：全角{int(max_chars)}文字\n"
        "- 出力形式：JSON の lines 配列のみ（例を参照）\n"
        "- 単語の順序を変えない。結合もしない。\n\n"
        "# 自然な分割ルール（最優先）\n"
        "**以下のルールは文字数制約よりも優先度が高い：**\n"
        "1. **行頭に助詞・補助表現・小さい文字を置かない**: 「が」「は」「を」「に」「で」「と」「も」「から」「まで」「よ」「ね」「な」「わ」や、「んじゃない」「が必要」「と思って」、および「ぁぃぅぇぉゃゅょっァィゥェォャュョッ」「ん」「ン」などで行を *始めない*（前の行とひとまとまりにする）\n"
        "2. **接続表現・接続詞で分割しない**: 〜と思って、〜ものの、〜たら、〜ので、〜けど、〜けれど、んで、それで、そして 等で文を切らない\n"
        "3. **助詞だけ／短すぎる助詞行を作らない**: 「が」「に」「を」「んで」など助詞を含む行が1〜4文字程度しかない場合は必ず前後の行と統合し、5文字以上のまとまりにする\n"
        "4. **活用語尾の保持**: 〜てた、〜だった、〜たと 等の活用形は分割せずひとまとまりに\n"
        "5. **引用表現の保持**: 〜って言う、〜って思う、〜ってこと 等は分割しない\n\n"
        "# 分割の良い例・悪い例\n"
        "❌ 悪い例:\n"
        "  - 「考えた」→「ことがあったから」 （助詞「が」で分断）\n"
        "  - 「なった時」→「に」 （1文字のみの行）\n"
        "  - 「目指す」→「ものの」 （接続助詞で分断）\n"
        "  - 「何するんだ」→「って言うから」 （引用「って」で分断）\n"
        "  - 「思います」→「よ」 （終助詞「よ」が単独・1文字）\n"
        "  - 「怖かった」→「んで」 （接続詞「んで」が単独・2文字）\n\n"
        "✅ 良い例:\n"
        "  - 「考えたことが」→「あったから」 （助詞を含めてひとまとまり）\n"
        "  - 「なった時に」→「いやしないと」 （最小4文字以上）\n"
        "  - 「目指すものの」→「ダメな場合も」 （接続表現を保持）\n"
        "  - 「何するんだって言うから」→「家の手伝いを」 （引用表現を保持）\n"
        "  - 「思いますよ」→「いらっしゃって」 （終助詞を含めて4文字以上）\n"
        "  - 「怖かったんで」→「父がすっかり」 （接続詞を含めて4文字以上）\n\n"
        "# 禁止事項\n"
        "- 行末の句読点（、。）は必ず削除すること。文中の句読点は、読みやすさのために残してもよい。\n"
        "- 助詞・接続詞・活用語尾・終助詞での不自然な分割（上記ルール参照）\n"
        "- 1〜4文字のみの極端に短い行の生成（短い行は必ず統合して5文字以上にする）\n\n"
        "# 思考ワークフロー（必ずこの順序で検討）\n"
        "## Step 1: 意味のまとまり (Meaning)\n"
        "句（文節）のまとまりを最優先。助詞・接続表現・引用表現を切らない。\n\n"
        "## Step 2: 文字数調整 (Length)\n"
        f"1. 文字数チェック: {int(max_chars)}文字を超える場合のみ改行\n"
        f"2. バランス: 2行にする場合、できるだけ近い長さに調整\n"
        f"3. 文脈区切り: {int(max_chars)}文字以内でも、読点・強い切れ目（〜ます、〜です、〜だ等）で終わるなら改行を検討\n"
        "4. 最小行長チェック: 分割後の行が5文字未満にならないか確認（4文字以下は禁止）\n\n"
        "## Step 3: クリーニング (Cleaning)\n"
        "行末の句読点（、。）を削除。文中の句読点は残してよい。\n\n"
        "# Input\n"
        f"単語リスト（index:word）:\n{indexed}\n\n"
        "# Output\n"
        "以下のJSONだけを返してください（説明・コードフェンス禁止）。例:\n"
        "{\n"
        '  "lines": [\n'
        '    {"from": 0, "to": 10, "text": "私は大学の12月ぐらい"},\n'
        '    {"from": 11, "to": 25, "text": "政治家になろうと決めていて"}\n'
        "  ]\n"
        "}\n"
    )


@dataclass
class ValidationIssue:
    type: Literal["short_particle_line", "split_quotation", "missing_coverage"]
    line_idx: int
    severity: Literal["high", "medium"]
    description: str
    suggested_action: str


def detect_issues(lines: Sequence["LineRange"], words: Sequence[WordTimestamp]) -> List[ValidationIssue]:
    issues: List[ValidationIssue] = []

    if lines:
        ordered = sorted(lines, key=lambda l: (l.start_idx, l.end_idx))
        prev_end = ordered[0].end_idx
        if ordered[0].start_idx > 0:
            issues.append(
                ValidationIssue(
                    type="missing_coverage",
                    line_idx=0,
                    severity="high",
                    description=f"行範囲に欠落があります（0-{ordered[0].start_idx - 1}）",
                    suggested_action="欠けている範囲を追加し、連続した範囲になるよう補完",
                )
            )
        for idx, line in enumerate(ordered[1:], start=1):
            if line.start_idx > prev_end + 1:
                issues.append(
                    ValidationIssue(
                        type="missing_coverage",
                        line_idx=idx - 1,
                        severity="high",
                        description=f"行範囲に欠落があります（{prev_end + 1}-{line.start_idx - 1}）",
                        suggested_action="欠けている範囲を追加し、連続した範囲になるよう補完",
                    )
                )
            if line.end_idx > prev_end:
                prev_end = line.end_idx
        if words:
            max_idx = len(words) - 1
            if prev_end < max_idx:
                issues.append(
                    ValidationIssue(
                        type="missing_coverage",
                        line_idx=len(ordered) - 1,
                        severity="high",
                        description=f"行範囲に欠落があります（{prev_end + 1}-{max_idx}）",
                        suggested_action="欠けている範囲を追加し、連続した範囲になるよう補完",
                    )
                )

    particles = ["を", "に", "で", "が", "は", "も", "から", "まで", "へ", "と"]
    for i, line in enumerate(lines):
        if len(line.text) < 5 and line.text:
            ends_with_particle = line.text[-1] in particles
            issues.append(
                ValidationIssue(
                    type="short_particle_line",
                    line_idx=i,
                    severity="high" if ends_with_particle else "medium",
                    description=(
                        f"行{i+1}は{len(line.text)}文字で短すぎます"
                        + (f"（助詞「{line.text[-1]}」で終わる）" if ends_with_particle else "")
                    ),
                    suggested_action="前行または次行と統合",
                )
            )

    for i in range(len(lines) - 1):
        current = lines[i]
        next_line = lines[i + 1]
        if not current.text or not next_line.text:
            continue
        if current.text.endswith("って") and next_line.text.startswith(("言", "思")):
            issues.append(
                ValidationIssue(
                    type="split_quotation",
                    line_idx=i,
                    severity="medium",
                    description=f"行{i+1}-{i+2}で引用表現「〜って言う/思う」が分割されている",
                    suggested_action="引用表現を統合",
                )
            )

    return issues


def build_pass3_prompt(
    lines: Sequence["LineRange"],
    words: Sequence[WordTimestamp],
    issues: Sequence[ValidationIssue],
    glossary_terms: Sequence[str],
) -> str:
    def _format_sec(value: object) -> str:
        try:
            return f"{float(value):.2f}"
        except (TypeError, ValueError):
            return "?"

    indexed = "\n".join(
        f"{i}: {w.word} ({_format_sec(getattr(w, 'start', None))}-{_format_sec(getattr(w, 'end', None))})"
        for i, w in enumerate(words)
    )
    if issues:
        issue_text = "\n".join([f"- {issue.description} → {issue.suggested_action}" for issue in issues])
    else:
        issue_text = "問題は検出されませんでした。全行を確認し、以下のルールに従って最小限の修正を行ってください。"
    has_missing_coverage = any(issue.type == "missing_coverage" for issue in issues)
    missing_rule = ""
    if has_missing_coverage:
        missing_rule = "10. **欠落したインデックス範囲を必ず補完**: 欠けている範囲の単語を追加し、0から末尾まで連続にする\n"
    line_timings = []
    for l in lines:
        start = getattr(words[l.start_idx], "start", None) if 0 <= l.start_idx < len(words) else None
        end = getattr(words[l.end_idx], "end", None) if 0 <= l.end_idx < len(words) else None
        duration = None
        if start is not None and end is not None:
            try:
                duration = float(end) - float(start)
            except (TypeError, ValueError):
                duration = None
        line_timings.append(
            {
                "from": l.start_idx,
                "to": l.end_idx,
                "text": l.text,
                "start": _format_sec(start),
                "end": _format_sec(end),
                "duration": _format_sec(duration),
            }
        )
    current_lines = json.dumps(
        [{"from": l.start_idx, "to": l.end_idx, "text": l.text} for l in lines],
        ensure_ascii=False,
        indent=2,
    )
    current_lines_with_time = json.dumps(line_timings, ensure_ascii=False, indent=2)
    return (
        "# Role\n"
        "あなたはテロップの最終チェック担当の熟練編集者です。\n"
        "Pass 2で作成された字幕の行分割に問題がないか確認し、必要最小限の修正を行ってください。\n\n"
        "# 検出された問題\n"
        f"{issue_text}\n\n"
        "# 修正ルール\n"
        "1. **1-4文字の極端に短い行**: 前行または次行と統合し、結合後に全行がルールに沿っているか再確認\n"
        "2. **引用表現の分割「〜って言う/思う」**: 統合して1行に\n"
        "3. **修正後も制約を維持**: 17文字以内・5文字以上\n"
        "4. **最小限の修正**: 問題箇所のみ修正（全体を作り直さない）\n"
        "5. **要約・翻訳・意訳をしない**。語句の追加・削除もしない\n"
        "6. **語の途中で切れている箇所は必ず連結**（分断された語を統合）\n"
        "7. **改行の優先度**: (1)「。?!」直後 → (2)「、」直後 → (3) 接続助詞・係助詞など句が自然に切れる後ろ。名詞句/動詞句の途中は切らない。迷う場合は改行しない\n"
        "8. **元の語順と文脈を保つ**。句読点がない場合も上記7に沿って自然に整形する\n\n"
        f"9. **時間幅の上限**: 1行の時間幅（end-start）が{MAX_LINE_DURATION_SEC:.1f}秒を超える場合は必ず分割する\n\n"
        f"{missing_rule}"
        "- 必ず1件以上の行を `lines` 配列で返してください（空配列やnullは禁止）\n\n"
        "# Input\n"
        f"単語リスト（index:word）:\n{indexed}\n\n"
        f"現在の行分割:\n{current_lines}\n\n"
        f"現在の行分割（時間情報）:\n{current_lines_with_time}\n\n"
        "# Output\n"
        "以下のJSONのみを返してください。説明文・コードフェンス・前後のテキストを含めることは禁止です。\n"
        "{\n"
        '  "lines": [\n'
        '    {"from": 0, "to": 11, "text": "私は大学の時の12月ぐらいかな"},\n'
        '    {"from": 12, "to": 18, "text": "4年生の12月には"},\n'
        '    {"from": 19, "to": 28, "text": "政治家になろうという腹を決めていて"},\n'
        '    {"from": 29, "to": 33, "text": "1月ぐらいから"},\n'
        '    {"from": 34, "to": 45, "text": "司法試験予備校に申し込んで"}\n'
        "  ]\n"
        "}\n"
    )


def call_gemini_cli(prompt: str, *, model: str | None, timeout: float | None) -> str:
    cmd = ["gemini", "--output-format", "text"]
    if model:
        cmd.extend(["--model", model])
    try:
        result = subprocess.run(
            cmd,
            input=prompt,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
        )
    except FileNotFoundError as exc:
        raise FormatterError("gemini が見つかりません。Gemini CLI をインストールしてください。") from exc
    except subprocess.TimeoutExpired as exc:
        raise FormatterError(f"gemini がタイムアウトしました（{timeout}秒）") from exc

    stdout = result.stdout or ""
    stderr = result.stderr or ""
    raw = stdout
    if stderr.strip():
        raw = (stdout.rstrip("\n") + "\n" + stderr).strip("\n")
    if result.returncode != 0:
        raise FormatterError(f"gemini 実行に失敗しました（exit={result.returncode}）: {raw}")
    return raw.strip()


@dataclass(slots=True)
class EditOperation:
    type: str
    start_idx: int
    end_idx: int
    text: str | None = None


@dataclass(slots=True)
class LineRange:
    start_idx: int
    end_idx: int
    text: str


@dataclass(slots=True)
class TwoPassResult:
    segments: List[SubtitleSegment]

    @property
    def srt_text(self) -> str:
        return segments_to_srt(self.segments)


def _extract_json(text: str) -> Any:
    fenced = re.search(r"```(?:json)?\\s*(\\{.*\\}|\\[.*\\])\\s*```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1)
    brace = text.find("{")
    bracket = text.find("[")
    start = min([p for p in [brace, bracket] if p != -1], default=-1)
    if start == -1:
        raise ValueError("JSONの開始文字（{ または [）が見つかりません")
    if start > 0:
        text = text[start:]
    decoder = json.JSONDecoder()
    value, _ = decoder.raw_decode(text)
    return value


def _parse_operations(raw: Any) -> List[EditOperation]:
    ops: List[EditOperation] = []
    items = raw.get("operations", []) if isinstance(raw, dict) else []
    for item in items:
        try:
            t = item["type"]
            s = int(item["start_idx"])
            e = int(item["end_idx"])
            if t not in {"replace", "delete"}:
                continue
            ops.append(EditOperation(type=t, start_idx=s, end_idx=e, text=item.get("text")))
        except Exception:
            continue
    return ops


def _parse_lines(raw: Any) -> List[LineRange]:
    lines: List[LineRange] = []
    items = raw.get("lines", []) if isinstance(raw, dict) else []
    for item in items:
        try:
            s = int(item["from"])
            e = int(item["to"])
            txt = str(item.get("text", "")).strip()
            lines.append(LineRange(start_idx=s, end_idx=e, text=txt))
        except Exception:
            continue
    return lines


def _apply_operations(words: Sequence[WordTimestamp], ops: Sequence[EditOperation]) -> List[WordTimestamp]:
    result = list(words)
    for op in sorted(ops, key=lambda o: (o.start_idx, o.end_idx), reverse=True):
        if op.start_idx < 0 or op.end_idx >= len(result) or op.start_idx > op.end_idx:
            continue
        if op.type == "delete":
            del result[op.start_idx : op.end_idx + 1]
        elif op.type == "replace":
            new_word = WordTimestamp(
                word=op.text or "",
                start=result[op.start_idx].start,
                end=result[op.end_idx].end,
                confidence=result[op.start_idx].confidence,
            )
            result[op.start_idx : op.end_idx + 1] = [new_word]
    return result


def _safe_trim_json_response(text: str) -> Any:
    try:
        return _extract_json(text)
    except Exception as exc:
        raise FormatterError(f"LLM JSONのパースに失敗しました: {exc}") from exc


def _call_llm_with_parse(
    call_fn: Callable[..., str],
    *,
    pass_label: str,
    prompt: str,
    model_override: str | None,
    retries: int,
    soft_fail: bool,
    log_sink: "TwoPassFormatter | None",
) -> tuple[str | None, Any | None]:
    last_exc: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            raw = call_fn(prompt, model_override=model_override, pass_label=pass_label)
        except FormatterError as exc:
            last_exc = exc
            message = str(exc)
            retryable = any(code in message for code in [" 500", " 502", " 503", " 504", " 429"])
            if retryable and attempt < retries:
                logger.warning("%s error (retryable, attempt %d/%d): %s", pass_label, attempt, retries, exc)
                continue
            err_text = f"[API ERROR] pass={pass_label} model={model_override or '-'}: {exc}\n"
            if log_sink is not None:
                log_sink._append_raw_log(pass_label, err_text)
            if soft_fail:
                return None, None
            raise FormatterError(f"{pass_label} failed (model={model_override or '-'}): {exc}") from exc

        if log_sink is not None:
            log_sink._append_raw_log(pass_label, raw)

        try:
            parsed = _safe_trim_json_response(raw)
            return raw, parsed
        except FormatterError as exc:
            last_exc = exc
            logger.warning("%s parse failed (attempt %d/%d): %s", pass_label, attempt, retries, exc)

    if soft_fail:
        return None, None
    raise FormatterError(f"{pass_label} parse failed (model={model_override or '-'}): {last_exc}") from last_exc  # type: ignore[misc]


def _has_contiguous_word_coverage(lines: Sequence[LineRange], n_words: int) -> bool:
    if not lines or n_words <= 0:
        return False
    ordered = sorted(lines, key=lambda l: (l.start_idx, l.end_idx))
    if ordered[0].start_idx != 0:
        return False
    prev_end = -1
    for line in ordered:
        if line.start_idx < 0 or line.end_idx >= n_words or line.start_idx > line.end_idx:
            return False
        if line.start_idx != prev_end + 1:
            return False
        prev_end = line.end_idx
    return prev_end == n_words - 1


def _try_normalize_lines_for_coverage(
    lines: Sequence[LineRange],
    n_words: int,
) -> tuple[List[LineRange], str] | None:
    if not lines or n_words <= 0:
        return None
    ordered = sorted(lines, key=lambda l: (l.start_idx, l.end_idx))

    def _sorted(candidate: Sequence[LineRange]) -> List[LineRange]:
        return sorted(candidate, key=lambda l: (l.start_idx, l.end_idx))

    def _shift(candidate: Sequence[LineRange], *, start_delta: int, end_delta: int) -> List[LineRange]:
        return _sorted(
            [
                LineRange(start_idx=l.start_idx + start_delta, end_idx=l.end_idx + end_delta, text=l.text)
                for l in candidate
            ]
        )

    max_end = max(l.end_idx for l in ordered)
    if max_end == n_words and all(l.end_idx <= n_words for l in ordered):
        clamped = _sorted(
            [
                LineRange(
                    start_idx=l.start_idx,
                    end_idx=(n_words - 1) if l.end_idx == n_words else l.end_idx,
                    text=l.text,
                )
                for l in ordered
            ]
        )
        if _has_contiguous_word_coverage(clamped, n_words):
            return clamped, "clamp_end"

    end_exclusive = _shift(ordered, start_delta=0, end_delta=-1)
    if _has_contiguous_word_coverage(end_exclusive, n_words):
        return end_exclusive, "end_exclusive"

    one_based = _shift(ordered, start_delta=-1, end_delta=-1)
    if _has_contiguous_word_coverage(one_based, n_words):
        return one_based, "one_based"

    one_based_end_exclusive = _shift(ordered, start_delta=-1, end_delta=-2)
    if _has_contiguous_word_coverage(one_based_end_exclusive, n_words):
        return one_based_end_exclusive, "one_based_end_exclusive"

    return None


@dataclass(slots=True)
class WordTimeChunk:
    index: int
    start_idx: int
    end_idx: int
    start_sec: float
    end_sec: float


def split_words_into_time_chunks(
    words: Sequence[WordTimestamp],
    *,
    chunk_sec: float = 300.0,
    snap_window_sec: float = 15.0,
    min_gap_sec: float = 0.2,
) -> List[WordTimeChunk]:
    if not words:
        return []

    try:
        chunk_sec = float(chunk_sec)
    except (TypeError, ValueError):
        chunk_sec = 300.0
    if chunk_sec <= 0:
        chunk_sec = 300.0

    try:
        snap_window_sec = float(snap_window_sec)
    except (TypeError, ValueError):
        snap_window_sec = 15.0
    if snap_window_sec < 0:
        snap_window_sec = 0.0

    try:
        min_gap_sec = float(min_gap_sec)
    except (TypeError, ValueError):
        min_gap_sec = 0.2
    if min_gap_sec < 0:
        min_gap_sec = 0.0

    starts = [float(w.start or 0.0) for w in words]
    ends = [float((w.end if w.end is not None else w.start) or 0.0) for w in words]

    chunks: List[WordTimeChunk] = []
    idx = 0
    chunk_index = 1
    n = len(words)

    while idx < n:
        start_idx = idx
        start_sec = starts[start_idx]

        if start_idx == n - 1:
            chunks.append(
                WordTimeChunk(
                    index=chunk_index,
                    start_idx=start_idx,
                    end_idx=start_idx,
                    start_sec=start_sec,
                    end_sec=ends[start_idx],
                )
            )
            break

        target_sec = start_sec + chunk_sec
        if target_sec >= ends[-1]:
            chunks.append(
                WordTimeChunk(
                    index=chunk_index,
                    start_idx=start_idx,
                    end_idx=n - 1,
                    start_sec=start_sec,
                    end_sec=ends[-1],
                )
            )
            break

        approx = bisect.bisect_left(starts, target_sec, lo=start_idx + 1)
        if approx >= n:
            approx = n - 1

        left = bisect.bisect_left(starts, target_sec - snap_window_sec, lo=start_idx + 1)
        right = bisect.bisect_right(starts, target_sec + snap_window_sec, lo=start_idx + 1)

        best_end_idx = None
        best_gap = -1.0
        for i in range(left, min(right, n - 1)):
            gap = starts[i] - ends[i - 1]
            if gap > best_gap:
                best_gap = gap
                best_end_idx = i - 1

        if best_end_idx is None or best_end_idx < start_idx:
            best_end_idx = max(start_idx, approx - 1)
        if best_gap < min_gap_sec:
            best_end_idx = max(start_idx, approx - 1)

        end_idx = min(max(best_end_idx, start_idx), n - 1)

        chunks.append(
            WordTimeChunk(
                index=chunk_index,
                start_idx=start_idx,
                end_idx=end_idx,
                start_sec=start_sec,
                end_sec=ends[end_idx],
            )
        )
        chunk_index += 1
        idx = end_idx + 1

    return chunks


def _normalize_segments_for_output(
    segments: List[SubtitleSegment],
    *,
    start_delay: float,
    fill_gaps: bool = True,
    max_gap_duration: float | None = None,
    gap_padding: float = 0.15,
) -> None:
    if not segments:
        return

    segments.sort(key=lambda s: (float(getattr(s, "start", 0.0)), float(getattr(s, "end", 0.0))))

    for i in range(1, len(segments)):
        prev_seg = segments[i - 1]
        curr_seg = segments[i]
        prev_end = float(getattr(prev_seg, "end", 0.0))
        curr_start = float(getattr(curr_seg, "start", 0.0))
        if prev_end > curr_start:
            prev_seg.end = curr_start

    for seg in segments:
        start = float(getattr(seg, "start", 0.0))
        end = float(getattr(seg, "end", start))
        if end < start:
            seg.end = start + 0.1

    if fill_gaps:
        _fill_segment_gaps(segments, max_gap_duration=max_gap_duration, gap_padding=gap_padding)

    try:
        start_delay = float(start_delay)
    except (TypeError, ValueError):
        start_delay = 0.0

    if start_delay > 0 and len(segments) > 1:
        original_last_end = float(getattr(segments[-1], "end", 0.0))

        for i in range(1, len(segments)):
            new_start = float(getattr(segments[i], "start", 0.0)) + start_delay
            if i == len(segments) - 1:
                max_start = max(original_last_end - 0.1, 0.0)
                if new_start > max_start:
                    new_start = max_start
            prev_end = float(getattr(segments[i - 1], "end", 0.0))
            if new_start < prev_end:
                new_start = prev_end
            segments[i].start = new_start

        if fill_gaps:
            _fill_segment_gaps(segments, max_gap_duration=max_gap_duration, gap_padding=gap_padding)

        segments[-1].end = original_last_end
        if segments[-1].end < segments[-1].start:
            segments[-1].start = max(segments[-1].end - 0.1, 0.0)

    for i, seg in enumerate(segments, start=1):
        seg.index = i


def _fill_segment_gaps(
    segments: List[SubtitleSegment],
    *,
    max_gap_duration: float | None = None,
    gap_padding: float = 0.0,
) -> None:
    if not segments:
        return

    for i in range(len(segments) - 1):
        current = segments[i]
        nxt = segments[i + 1]
        gap = float(getattr(nxt, "start", 0.0)) - float(getattr(current, "end", 0.0))
        if gap <= 0:
            continue
        if max_gap_duration is not None and gap > float(max_gap_duration):
            if gap_padding > 0:
                desired_end = float(getattr(current, "end", 0.0)) + float(gap_padding)
                current.end = min(desired_end, float(getattr(nxt, "start", 0.0)))
            continue
        current.end = float(getattr(nxt, "start", 0.0))


class TwoPassFormatter:
    def __init__(
        self,
        *,
        pass1_model: str | None,
        pass2_model: str | None,
        pass3_model: str | None,
        pass4_model: str | None,
        glossary_terms: Sequence[str] | None,
        run_id: str | None,
        source_name: str | None,
        raw_log_dir: Path,
        max_gap_duration: float | None,
        start_delay: float,
        timeout: float | None,
    ) -> None:
        self.pass1_model = pass1_model or os.getenv("LLM_PASS1_MODEL", "gemini-3-pro-preview")
        self.pass2_model = pass2_model or os.getenv("LLM_PASS2_MODEL", "gemini-3-pro-preview")
        self.pass3_model = pass3_model or os.getenv("LLM_PASS3_MODEL", "gemini-2.5-flash")
        self.pass4_model = pass4_model or os.getenv("LLM_PASS4_MODEL", self.pass3_model)

        self.glossary_terms = list(DEFAULT_GLOSSARY_TERMS) if glossary_terms is None else normalize_glossary_terms(glossary_terms)
        self.run_id = run_id
        self.source_name = source_name
        self.raw_log_dir = raw_log_dir
        self._log_buffer: Dict[str, str] = {}
        self._log_date_str = datetime.now(timezone.utc).strftime("%Y%m%d")
        self._log_written = False

        self.fill_gaps = True
        self.max_gap_duration = max_gap_duration
        self.gap_padding = 0.15
        self.start_delay = float(start_delay)
        self.timeout = timeout
        self._current_line_max_chars = 17

    def _append_raw_log(self, pass_label: str, raw: str) -> None:
        prefix = f"\n\n===== {pass_label} =====\n"
        existing = self._log_buffer.get(pass_label, "")
        self._log_buffer[pass_label] = existing + prefix + raw

    def _flush_logs(self) -> None:
        if self._log_written or not self._log_buffer:
            return
        try:
            self.raw_log_dir.mkdir(parents=True, exist_ok=True)
            base_name = (self.source_name or self.run_id or "llm_run").replace("/", "_")
            suffix = self.run_id or uuid.uuid4().hex[:8]
            fname = self.raw_log_dir / f"{base_name}_{self._log_date_str}_{suffix}.txt"
            ordered = []
            for label in sorted(self._log_buffer.keys()):
                ordered.append(self._log_buffer[label])
            fname.write_text("".join(ordered), encoding="utf-8")
            self._log_written = True
        except Exception as exc:
            logger.warning("Failed to save aggregated LLM raw log: %s", exc)

    def _call_llm(self, prompt_text: str, *, model_override: str | None, pass_label: str | None) -> str:
        return call_gemini_cli(prompt_text, model=model_override, timeout=self.timeout)

    def _line_duration_seconds(self, line: LineRange, words: Sequence[WordTimestamp]) -> float | None:
        if line.start_idx < 0 or line.end_idx >= len(words):
            return None
        start = getattr(words[line.start_idx], "start", None)
        end = getattr(words[line.end_idx], "end", None)
        try:
            return float(end) - float(start)
        except (TypeError, ValueError):
            return None

    def _needs_pass4(self, line: LineRange, words: Sequence[WordTimestamp]) -> bool:
        max_chars = getattr(self, "_current_line_max_chars", 17)
        if len(line.text) > max_chars or len(line.text) < 5:
            return True
        duration = self._line_duration_seconds(line, words)
        return duration is not None and duration > MAX_LINE_DURATION_SEC

    def _is_valid_pass4_replacement(
        self,
        original: LineRange,
        repl: Sequence[LineRange],
        words: Sequence[WordTimestamp],
        *,
        enforce_duration: bool,
    ) -> bool:
        if not repl:
            return False
        max_chars = getattr(self, "_current_line_max_chars", 17)
        prev_end = original.start_idx - 1
        durations: List[float | None] = []
        for line in repl:
            if line.start_idx < original.start_idx or line.end_idx > original.end_idx:
                return False
            if line.start_idx != prev_end + 1:
                return False
            if line.start_idx > line.end_idx:
                return False
            if len(line.text) < 5 or len(line.text) > max_chars:
                return False
            if enforce_duration:
                durations.append(self._line_duration_seconds(line, words))
            prev_end = line.end_idx
        if prev_end != original.end_idx:
            return False
        if enforce_duration and durations and all(d is not None for d in durations):
            if any(d > MAX_LINE_DURATION_SEC for d in durations):
                return False
        return True

    def _run_pass4_fix(
        self,
        line: LineRange,
        words: Sequence[WordTimestamp],
        *,
        enforce_duration: bool,
    ) -> List[LineRange]:
        prompt = build_pass4_prompt(line, words, getattr(self, "_current_line_max_chars", 17))
        raw, parsed = _call_llm_with_parse(
            self._call_llm,
            pass_label="pass4",
            prompt=prompt,
            model_override=self.pass4_model,
            retries=1,
            soft_fail=True,
            log_sink=self,
        )
        if parsed:
            repl = _parse_lines(parsed)
            if repl:
                repl = sorted(repl, key=lambda l: (l.start_idx, l.end_idx))
                if self._is_valid_pass4_replacement(line, repl, words, enforce_duration=enforce_duration):
                    return repl
        return [line]

    def _build_fallback_lines(self, words: Sequence[WordTimestamp]) -> List[LineRange]:
        if not words:
            return []

        max_chars = getattr(self, "_current_line_max_chars", 17)
        max_idx = len(words) - 1

        lines: List[LineRange] = []
        idx = 0
        while idx <= max_idx:
            line_start = idx
            text_parts: List[str] = []
            current_len = 0
            while idx <= max_idx:
                word = words[idx].word or ""
                next_len = current_len + len(word)
                if text_parts and next_len > max_chars and current_len >= 5:
                    break
                text_parts.append(word)
                current_len = next_len
                idx += 1
                if current_len >= 10 and (idx > max_idx or current_len >= max_chars):
                    break
            line_end = idx - 1
            if not text_parts:
                idx += 1
                continue
            lines.append(LineRange(start_idx=line_start, end_idx=line_end, text="".join(text_parts)))
        return lines

    def _ensure_trailing_coverage(self, lines: Sequence[LineRange], words: Sequence[WordTimestamp]) -> List[LineRange]:
        if not lines or not words:
            return list(lines)

        max_chars = getattr(self, "_current_line_max_chars", 17)
        max_idx = len(words) - 1
        last_line = max(lines, key=lambda l: l.end_idx)
        if last_line.end_idx >= max_idx:
            return list(lines)

        gap_start = max(last_line.end_idx + 1, 0)
        if gap_start > max_idx:
            return list(lines)

        fallback_lines: List[LineRange] = list(lines)
        idx = gap_start
        while idx <= max_idx:
            line_start = idx
            text_parts: List[str] = []
            current_len = 0
            while idx <= max_idx:
                word = words[idx].word or ""
                next_len = current_len + len(word)
                if text_parts and next_len > max_chars and current_len >= 5:
                    break
                text_parts.append(word)
                current_len = next_len
                idx += 1
                if current_len >= 10 and (idx > max_idx or current_len >= max_chars):
                    break
            line_end = idx - 1
            if not text_parts:
                idx += 1
                continue
            fallback_lines.append(LineRange(start_idx=line_start, end_idx=line_end, text="".join(text_parts)))
        return sorted(fallback_lines, key=lambda l: (l.start_idx, l.end_idx))

    def _ranges_to_segments(self, words: Sequence[WordTimestamp], lines: Sequence[LineRange]) -> List[SubtitleSegment]:
        segments: List[SubtitleSegment] = []
        ordered_lines = sorted(lines, key=lambda l: (l.start_idx, l.end_idx))
        for line in ordered_lines:
            if line.start_idx < 0 or line.end_idx >= len(words) or line.start_idx > line.end_idx:
                continue
            start = words[line.start_idx].start or 0.0
            end = words[line.end_idx].end or start
            if end < start:
                end = start + 0.1
            segments.append(SubtitleSegment(index=0, start=start, end=end, text=line.text))
        _normalize_segments_for_output(
            segments,
            start_delay=self.start_delay,
            fill_gaps=self.fill_gaps,
            max_gap_duration=self.max_gap_duration,
            gap_padding=self.gap_padding,
        )
        return segments

    def run(self, *, text: str, words: Sequence[WordTimestamp], max_chars: float) -> TwoPassResult:
        if not words:
            raise FormatterError("wordタイムスタンプが空です")
        try:
            max_chars_int = int(max_chars)
        except (TypeError, ValueError):
            max_chars_int = 17
        if max_chars_int <= 0:
            max_chars_int = 17
        self._current_line_max_chars = max_chars_int
        max_chars = float(max_chars_int)

        try:
            raw1, parsed1 = _call_llm_with_parse(
                self._call_llm,
                pass_label="pass1",
                prompt=build_pass1_prompt(text, words, self.glossary_terms),
                model_override=self.pass1_model,
                retries=2,
                soft_fail=False,
                log_sink=self,
            )
            ops = _parse_operations(parsed1)
            updated_words = _apply_operations(words, ops) if ops else list(words)
            if not updated_words:
                return TwoPassResult(segments=[])

            raw2, parsed2 = _call_llm_with_parse(
                self._call_llm,
                pass_label="pass2",
                prompt=build_pass2_prompt(updated_words, max_chars),
                model_override=self.pass2_model,
                retries=2,
                soft_fail=False,
                log_sink=self,
            )
            pass2_lines = _parse_lines(parsed2)
            pass2_lines = sorted(pass2_lines, key=lambda l: (l.start_idx, l.end_idx))
            if not pass2_lines or not _has_contiguous_word_coverage(pass2_lines, len(updated_words)):
                normalized = _try_normalize_lines_for_coverage(pass2_lines, len(updated_words))
                if normalized is None:
                    logger.warning("pass2: invalid line ranges; fallback to naive segmentation")
                    pass2_lines = self._build_fallback_lines(updated_words)
                else:
                    pass2_lines, reason = normalized
                    logger.warning("pass2: normalized line ranges (%s)", reason)
            lines = pass2_lines

            issues = detect_issues(lines, updated_words)
            raw3, parsed3 = _call_llm_with_parse(
                self._call_llm,
                pass_label="pass3",
                prompt=build_pass3_prompt(lines, updated_words, issues, self.glossary_terms),
                model_override=self.pass3_model,
                retries=2,
                soft_fail=True,
                log_sink=self,
            )
            if parsed3:
                pass3_lines = _parse_lines(parsed3)
                pass3_lines = sorted(pass3_lines, key=lambda l: (l.start_idx, l.end_idx))
                if pass3_lines and not _has_contiguous_word_coverage(pass3_lines, len(updated_words)):
                    normalized = _try_normalize_lines_for_coverage(pass3_lines, len(updated_words))
                    if normalized is not None:
                        pass3_lines, reason = normalized
                        logger.warning("pass3: normalized line ranges (%s)", reason)
                if pass3_lines and _has_contiguous_word_coverage(pass3_lines, len(updated_words)):
                    lines = pass3_lines

            fixed_lines: List[LineRange] = []
            for line in lines:
                duration = self._line_duration_seconds(line, updated_words)
                duration_issue = duration is not None and duration > MAX_LINE_DURATION_SEC
                if self._needs_pass4(line, updated_words):
                    fixed_lines.extend(self._run_pass4_fix(line, updated_words, enforce_duration=duration_issue))
                else:
                    fixed_lines.append(line)
            lines = fixed_lines

            lines = self._ensure_trailing_coverage(lines, updated_words)
            segments = self._ranges_to_segments(updated_words, lines)
            return TwoPassResult(segments=segments)
        finally:
            self._flush_logs()


def run_workflow2_chunked_two_pass(
    *,
    formatter_builder: Callable[[str], TwoPassFormatter],
    text: str,
    words: Sequence[WordTimestamp],
    max_chars: float,
    start_delay: float,
    source_name: str,
) -> str:
    if not words:
        raise FormatterError("wordタイムスタンプが空です")

    chunks = split_words_into_time_chunks(words, chunk_sec=300.0, snap_window_sec=15.0, min_gap_sec=0.2)
    if len(chunks) <= 1:
        formatter = formatter_builder(source_name)
        result = formatter.run(text=text, words=words, max_chars=max_chars)
        return result.srt_text

    def _run_chunk(chunk_index: int, start_idx: int, end_idx: int) -> TwoPassResult:
        chunk_words = list(words[start_idx : end_idx + 1])
        chunk_text = "".join((w.word or "") for w in chunk_words)
        chunk_source = f"{source_name}_chunk{chunk_index:03d}"
        formatter = formatter_builder(chunk_source)
        formatter.start_delay = 0.0
        return formatter.run(text=chunk_text, words=chunk_words, max_chars=max_chars)

    max_workers = min(10, len(chunks))
    results: Dict[int, TwoPassResult] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_run_chunk, c.index, c.start_idx, c.end_idx): c.index for c in chunks}
        for future in as_completed(futures):
            chunk_idx = futures[future]
            results[chunk_idx] = future.result()

    merged_segments: List[SubtitleSegment] = []
    for i in sorted(results.keys()):
        merged_segments.extend(results[i].segments)

    _normalize_segments_for_output(
        merged_segments,
        start_delay=start_delay,
        fill_gaps=True,
        max_gap_duration=MAX_GAP_DURATION_SEC,
        gap_padding=0.15,
    )
    return TwoPassResult(segments=merged_segments).srt_text


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate SRT from audio/video (Whisper + Gemini CLI).")
    parser.add_argument("input", help="音声/動画ファイルパス")
    parser.add_argument("--output-dir", default="output", help="出力ルート（固定運用: output）")
    parser.add_argument("--language", default=None, help="Whisperの言語コード（例: ja）。未指定なら自動判定")
    parser.add_argument(
        "--whisper-backend",
        default=os.getenv("WHISPER_BACKEND", "auto"),
        choices=["auto", "mlx", "faster", "openai"],
        help="Whisperバックエンド（auto=OSで自動選択 / mac=mlx, win=faster）。",
    )
    parser.add_argument("--start-delay", type=float, default=float(os.getenv("FLOWCUT_START_DELAY", "0.2")))
    parser.add_argument("--line-max-chars", type=int, default=int(os.getenv("FLOWCUT_LINE_MAX_CHARS", "17")))
    parser.add_argument("--keep-extracted-audio", action="store_true", help="動画から抽出した音声を削除しない")
    parser.add_argument("--whisper-model", default=os.getenv("WHISPER_MODEL", "large-v3"))
    parser.add_argument("--llm-timeout", type=float, default=float(os.getenv("LLM_REQUEST_TIMEOUT", "500.0")))
    return parser.parse_args(argv)


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )


def _require_command(name: str) -> None:
    from shutil import which

    if which(name) is None:
        raise RuntimeError(f"必要なコマンドが見つかりません: {name}")


def _resolve_whisper_backend(raw: str) -> Literal["mlx", "faster", "openai"]:
    value = str(raw or "auto").strip().lower()
    if value == "auto":
        if sys.platform == "darwin":
            return "mlx"
        if os.name == "nt":
            return "faster"
        return "openai"
    if value in ("mlx", "faster", "openai"):
        return value
    return "openai"


def main(argv: Sequence[str]) -> int:
    _setup_logging()
    args = _parse_args(argv)

    input_path = Path(args.input).expanduser()
    if not input_path.exists():
        raise FileNotFoundError(f"入力ファイルが見つかりません: {input_path}")

    _require_command("gemini")
    if is_video_file(input_path):
        _require_command("ffmpeg")

    out_root = Path(args.output_dir)
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    run_base_dir = out_root / f"{input_path.stem}_{timestamp}"
    run_output_dir = generate_sequential_path(run_base_dir)
    logs_root = run_output_dir / "logs"
    json_output_dir = logs_root / "poc_samples"
    raw_llm_log_dir = logs_root / "llm_raw"

    json_output_dir.mkdir(parents=True, exist_ok=True)
    raw_llm_log_dir.mkdir(parents=True, exist_ok=True)

    extracted_audio_path: Path | None = None
    audio_path = input_path
    original_stem = input_path.stem
    if is_video_file(input_path):
        extracted_audio_path = extract_audio_from_video(input_path, output_dir=run_output_dir)
        audio_path = extracted_audio_path

    whisper_backend = _resolve_whisper_backend(str(args.whisper_backend))
    run_id = f"{original_stem}_{whisper_backend}_{timestamp}"

    t0 = time.perf_counter()
    if whisper_backend == "mlx":
        transcription = transcribe_mlx_whisper_local(
            audio_path,
            language=args.language,
            model=str(args.whisper_model),
        )
    elif whisper_backend == "faster":
        transcription = transcribe_faster_whisper_local(
            audio_path,
            language=args.language,
            model=str(args.whisper_model),
        )
    else:
        transcription = transcribe_openai_whisper_local(
            audio_path,
            language=args.language,
            model_name=str(args.whisper_model),
        )
    t1 = time.perf_counter()

    transcript_path = json_output_dir / f"{run_id}.json"
    transcript_path.write_text(json.dumps(transcription.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    def _build_formatter(source_name: str) -> TwoPassFormatter:
        return TwoPassFormatter(
            pass1_model=os.getenv("LLM_PASS1_MODEL"),
            pass2_model=os.getenv("LLM_PASS2_MODEL"),
            pass3_model=os.getenv("LLM_PASS3_MODEL"),
            pass4_model=os.getenv("LLM_PASS4_MODEL"),
            glossary_terms=None,
            run_id=run_id,
            source_name=source_name,
            raw_log_dir=raw_llm_log_dir,
            max_gap_duration=MAX_GAP_DURATION_SEC,
            start_delay=float(args.start_delay),
            timeout=float(args.llm_timeout),
        )

    srt_text = run_workflow2_chunked_two_pass(
        formatter_builder=_build_formatter,
        text=transcription.text,
        words=transcription.words,
        max_chars=float(args.line_max_chars),
        start_delay=float(args.start_delay),
        source_name=input_path.name,
    )
    t2 = time.perf_counter()

    srt_path = run_output_dir / f"{run_id}.srt"
    srt_path.write_text(srt_text, encoding="utf-8")

    meta = {
        "run_id": run_id,
        "input": str(input_path),
        "audio": str(audio_path),
        "timestamp": timestamp,
        "whisper": {
            "backend": whisper_backend,
            "model": str(args.whisper_model),
            "language": args.language,
            "elapsed_sec": round(t1 - t0, 3),
        },
        "llm": {
            "engine": "gemini-cli",
            "pass1_model": os.getenv("LLM_PASS1_MODEL", "gemini-3-pro-preview"),
            "pass2_model": os.getenv("LLM_PASS2_MODEL", "gemini-3-pro-preview"),
            "pass3_model": os.getenv("LLM_PASS3_MODEL", "gemini-2.5-flash"),
            "pass4_model": os.getenv("LLM_PASS4_MODEL", os.getenv("LLM_PASS3_MODEL", "gemini-2.5-flash")),
            "elapsed_sec": round(t2 - t1, 3),
        },
        "output": {
            "dir": str(run_output_dir),
            "srt": str(srt_path),
            "transcript": str(transcript_path),
        },
        "total_elapsed_sec": round(t2 - t0, 3),
    }
    meta["whisper"]["model"] = transcription.metadata.get("model") or meta["whisper"]["model"]
    meta["whisper"]["language"] = transcription.metadata.get("language") or meta["whisper"]["language"]
    if whisper_backend == "faster":
        meta["whisper"]["device"] = transcription.metadata.get("device")
        meta["whisper"]["compute_type"] = transcription.metadata.get("compute_type")
    (logs_root / "run_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    if extracted_audio_path is not None and not bool(args.keep_extracted_audio):
        cleanup_extracted_audio(extracted_audio_path)

    print(f"done: {srt_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv[1:]))
    except Exception as exc:
        logger.error("%s", exc)
        raise

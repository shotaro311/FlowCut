"""Gemini APIを使用した音声文字起こし（Gemini 3 Flash Preview対応）。"""
from __future__ import annotations

import base64
import json
import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Literal

import requests

from src.config import get_settings

logger = logging.getLogger(__name__)


class GeminiModel(Enum):
    """利用可能なGeminiモデル"""

    FLASH_2_5 = "gemini-2.5-flash-preview-native-audio-dialog"
    FLASH_3_PREVIEW = "gemini-3-flash-preview"


class ThinkingLevel(Enum):
    """Gemini 3 Flash Preview用のthinking_budget（トークン数）"""

    MINIMAL = 1024  # 最速、シンプルなタスク向け
    LOW = 4096  # 軽い推論
    MEDIUM = 8192  # バランス型
    HIGH = 24576  # 最高精度、複雑なタスク向け


@dataclass
class GeminiSegment:
    """Gemini文字起こしのセグメント"""

    text: str
    start_sec: float
    end_sec: float
    confidence: float = 1.0
    source_model: str = ""


class GeminiTranscriberError(RuntimeError):
    """Gemini文字起こしエラー"""


class GeminiTranscriber:
    """Gemini APIを使用した音声文字起こし（2.5 Flash / 3 Flash Preview対応）"""

    # モデル別デフォルト設定
    MODEL_DEFAULTS: Dict[GeminiModel, Dict[str, Any]] = {
        GeminiModel.FLASH_2_5: {
            "supports_thinking": False,
            "default_temperature": 1.0,
        },
        GeminiModel.FLASH_3_PREVIEW: {
            "supports_thinking": True,
            "default_temperature": 1.0,
            "default_thinking_level": ThinkingLevel.MEDIUM,
        },
    }

    # 対応音声フォーマット
    SUPPORTED_FORMATS = {
        ".wav": "audio/wav",
        ".mp3": "audio/mp3",
        ".aiff": "audio/aiff",
        ".aac": "audio/aac",
        ".ogg": "audio/ogg",
        ".flac": "audio/flac",
        ".m4a": "audio/mp4",
    }

    def __init__(
        self,
        api_key: str | None = None,
        model: GeminiModel = GeminiModel.FLASH_3_PREVIEW,
        language: str = "ja",
        thinking_level: ThinkingLevel | None = None,
        temperature: float | None = None,
        timeout: float = 300.0,
    ):
        settings = get_settings().llm
        self.api_key = api_key or settings.google_api_key
        if not self.api_key:
            raise GeminiTranscriberError("GOOGLE_API_KEY が未設定です")

        self.model = model
        self.language = language
        self.timeout = timeout
        self.api_base = settings.google_api_base

        # モデル別設定
        model_config = self.MODEL_DEFAULTS[model]
        if model_config["supports_thinking"]:
            self.thinking_level = thinking_level or model_config.get(
                "default_thinking_level", ThinkingLevel.MEDIUM
            )
        else:
            self.thinking_level = None

        self.temperature = (
            temperature if temperature is not None else model_config["default_temperature"]
        )

    def transcribe(
        self,
        audio_path: Path,
        *,
        chunk_sec: float = 300.0,
    ) -> List[GeminiSegment]:
        """
        音声ファイルを文字起こしし、セグメント単位で返す。

        Args:
            audio_path: 音声ファイルのパス
            chunk_sec: 長尺ファイルの分割単位（秒）※現在は未使用、将来の拡張用

        Returns:
            GeminiSegmentのリスト
        """
        if not audio_path.exists():
            raise GeminiTranscriberError(f"音声ファイルが見つかりません: {audio_path}")

        suffix = audio_path.suffix.lower()
        if suffix not in self.SUPPORTED_FORMATS:
            raise GeminiTranscriberError(
                f"未対応の音声フォーマット: {suffix}. "
                f"対応フォーマット: {list(self.SUPPORTED_FORMATS.keys())}"
            )

        mime_type = self.SUPPORTED_FORMATS[suffix]
        audio_data = self._encode_audio(audio_path)

        logger.info(
            "Gemini transcription start: model=%s, file=%s, size=%d bytes",
            self.model.value,
            audio_path.name,
            audio_path.stat().st_size,
        )

        try:
            response = self._call_api(audio_data, mime_type)
            segments = self._parse_response(response)
            logger.info(
                "Gemini transcription done: segments=%d",
                len(segments),
            )
            return segments
        except Exception as exc:
            logger.error("Gemini transcription failed: %s", exc)
            raise GeminiTranscriberError(f"文字起こしに失敗しました: {exc}") from exc

    def transcribe_text_only(self, audio_path: Path) -> str:
        """
        音声ファイルを文字起こしし、テキストのみを返す（タイムスタンプなし）。

        Args:
            audio_path: 音声ファイルのパス

        Returns:
            文字起こしテキスト
        """
        if not audio_path.exists():
            raise GeminiTranscriberError(f"音声ファイルが見つかりません: {audio_path}")

        suffix = audio_path.suffix.lower()
        if suffix not in self.SUPPORTED_FORMATS:
            raise GeminiTranscriberError(
                f"未対応の音声フォーマット: {suffix}. "
                f"対応フォーマット: {list(self.SUPPORTED_FORMATS.keys())}"
            )

        mime_type = self.SUPPORTED_FORMATS[suffix]
        audio_data = self._encode_audio(audio_path)

        logger.info(
            "Gemini text-only transcription start: model=%s, file=%s",
            self.model.value,
            audio_path.name,
        )

        try:
            response = self._call_api_text_only(audio_data, mime_type)
            text = self._extract_text(response)
            logger.info(
                "Gemini text-only transcription done: chars=%d",
                len(text),
            )
            return text
        except Exception as exc:
            logger.error("Gemini text-only transcription failed: %s", exc)
            raise GeminiTranscriberError(f"文字起こしに失敗しました: {exc}") from exc

    def _encode_audio(self, audio_path: Path) -> str:
        """音声ファイルをBase64エンコード"""
        with open(audio_path, "rb") as f:
            return base64.standard_b64encode(f.read()).decode("utf-8")

    def _build_generation_config(self, response_mime_type: str | None = None) -> Dict[str, Any]:
        """モデルに応じたgenerationConfigを構築"""
        config: Dict[str, Any] = {
            "temperature": self.temperature,
        }

        if response_mime_type:
            config["response_mime_type"] = response_mime_type

        # Gemini 3 Flash Preview用のthinking_level設定
        if self.thinking_level is not None:
            config["thinking_config"] = {
                "thinking_budget": self.thinking_level.value,
            }

        return config

    def _build_transcription_prompt(self) -> str:
        """文字起こし用プロンプト（セグメント付き）"""
        return f"""あなたは音声文字起こしの専門家です。
以下の音声を正確に文字起こししてください。

言語: {self.language}

出力形式（JSON）:
{{
  "segments": [
    {{"text": "発話内容", "start": 0.0, "end": 2.5}},
    {{"text": "次の発話", "start": 2.6, "end": 5.0}}
  ]
}}

注意事項:
- 発話の区切りごとにセグメントを分けてください
- start/end は秒単位の小数で記載してください
- 聞き取れない部分は [不明瞭] と記載してください
- 固有名詞は文脈から推測して記載してください
- フィラー（えー、あのー）は可能な限り省略してください
- 必ず有効なJSONのみを出力してください（説明文やコードフェンスは不要）
"""

    def _build_text_only_prompt(self) -> str:
        """文字起こし用プロンプト（テキストのみ）"""
        return f"""あなたは音声文字起こしの専門家です。
以下の音声を正確に文字起こししてください。

言語: {self.language}

注意事項:
- 聞き取れない部分は [不明瞭] と記載してください
- 固有名詞は文脈から推測して記載してください
- フィラー（えー、あのー）は可能な限り省略してください
- 文字起こしのテキストのみを出力してください（説明文は不要）
"""

    def _call_api(self, audio_data: str, mime_type: str) -> Dict[str, Any]:
        """Gemini APIを呼び出し（セグメント付き）"""
        endpoint = f"{self.api_base.rstrip('/')}/models/{self.model.value}:generateContent"

        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": self._build_transcription_prompt()},
                        {
                            "inline_data": {
                                "mime_type": mime_type,
                                "data": audio_data,
                            }
                        },
                    ]
                }
            ],
            "generationConfig": self._build_generation_config("application/json"),
        }

        response = requests.post(
            endpoint,
            params={"key": self.api_key},
            json=payload,
            timeout=self.timeout,
        )

        if response.status_code != 200:
            raise GeminiTranscriberError(
                f"API request failed: {response.status_code} - {response.text}"
            )

        return response.json()

    def _call_api_text_only(self, audio_data: str, mime_type: str) -> Dict[str, Any]:
        """Gemini APIを呼び出し（テキストのみ）"""
        endpoint = f"{self.api_base.rstrip('/')}/models/{self.model.value}:generateContent"

        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": self._build_text_only_prompt()},
                        {
                            "inline_data": {
                                "mime_type": mime_type,
                                "data": audio_data,
                            }
                        },
                    ]
                }
            ],
            "generationConfig": self._build_generation_config(),
        }

        response = requests.post(
            endpoint,
            params={"key": self.api_key},
            json=payload,
            timeout=self.timeout,
        )

        if response.status_code != 200:
            raise GeminiTranscriberError(
                f"API request failed: {response.status_code} - {response.text}"
            )

        return response.json()

    def _parse_response(self, response: Dict[str, Any]) -> List[GeminiSegment]:
        """API応答をパースしてGeminiSegmentリストを返す"""
        try:
            candidates = response.get("candidates", [])
            if not candidates:
                raise GeminiTranscriberError("API応答にcandidatesがありません")

            content = candidates[0].get("content", {})
            parts = content.get("parts", [])
            if not parts:
                raise GeminiTranscriberError("API応答にpartsがありません")

            text = parts[0].get("text", "").strip()
            if not text:
                raise GeminiTranscriberError("API応答が空です")

            # JSONをパース
            data = self._extract_json(text)
            segments_data = data.get("segments", [])

            segments: List[GeminiSegment] = []
            for seg in segments_data:
                segments.append(
                    GeminiSegment(
                        text=seg.get("text", ""),
                        start_sec=float(seg.get("start", 0.0)),
                        end_sec=float(seg.get("end", 0.0)),
                        confidence=1.0,
                        source_model=self.model.value,
                    )
                )

            return segments

        except (KeyError, IndexError, json.JSONDecodeError) as exc:
            raise GeminiTranscriberError(f"API応答のパースに失敗: {exc}") from exc

    def _extract_text(self, response: Dict[str, Any]) -> str:
        """API応答からテキストを抽出"""
        try:
            candidates = response.get("candidates", [])
            if not candidates:
                raise GeminiTranscriberError("API応答にcandidatesがありません")

            content = candidates[0].get("content", {})
            parts = content.get("parts", [])
            if not parts:
                raise GeminiTranscriberError("API応答にpartsがありません")

            return parts[0].get("text", "").strip()

        except (KeyError, IndexError) as exc:
            raise GeminiTranscriberError(f"API応答のパースに失敗: {exc}") from exc

    def _extract_json(self, text: str) -> Dict[str, Any]:
        """テキストからJSONを抽出"""
        import re

        # コードフェンスを除去
        fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
        if fenced:
            text = fenced.group(1)

        # 最初の { から最後の } までを抽出
        brace_start = text.find("{")
        brace_end = text.rfind("}")
        if brace_start != -1 and brace_end != -1:
            text = text[brace_start : brace_end + 1]

        return json.loads(text)

    @classmethod
    def create_for_use_case(
        cls,
        use_case: Literal["standard", "complex", "cost_efficient"],
        api_key: str | None = None,
        language: str = "ja",
    ) -> "GeminiTranscriber":
        """ユースケースに応じた最適な設定でインスタンスを生成"""

        if use_case == "standard":
            # 通常の字幕作成: バランス型
            return cls(
                api_key=api_key,
                model=GeminiModel.FLASH_3_PREVIEW,
                language=language,
                thinking_level=ThinkingLevel.MEDIUM,
            )
        elif use_case == "complex":
            # 専門用語・複雑な音声: 高精度
            return cls(
                api_key=api_key,
                model=GeminiModel.FLASH_3_PREVIEW,
                language=language,
                thinking_level=ThinkingLevel.HIGH,
            )
        elif use_case == "cost_efficient":
            # コスト重視: 最速設定
            return cls(
                api_key=api_key,
                model=GeminiModel.FLASH_3_PREVIEW,
                language=language,
                thinking_level=ThinkingLevel.MINIMAL,
            )
        else:
            raise ValueError(f"Unknown use_case: {use_case}")


__all__ = [
    "GeminiModel",
    "GeminiSegment",
    "GeminiTranscriber",
    "GeminiTranscriberError",
    "ThinkingLevel",
]

"""GUI設定の永続化を管理するモジュール。"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict

from src.utils.glossary import DEFAULT_GLOSSARY_TERMS, normalize_glossary_terms, parse_glossary_text

logger = logging.getLogger(__name__)


class GuiConfig:
    """GUI設定の読み込みと保存を管理するクラス。"""

    def __init__(self) -> None:
        # 設定ファイルのパス: ~/.flowcut/config.json
        self.config_dir = Path.home() / ".flowcut"
        self.config_file = self.config_dir / "config.json"
        self._config: Dict[str, Any] = {}
        self._load_config()

    def _load_config(self) -> None:
        """設定ファイルから設定を読み込む。"""
        try:
            if self.config_file.exists():
                with open(self.config_file, "r", encoding="utf-8") as f:
                    self._config = json.load(f)
                logger.debug("設定を読み込みました: %s", self.config_file)
            else:
                self._config = {}
                logger.debug("設定ファイルが存在しないため、デフォルト設定を使用します")
        except Exception as exc:
            logger.warning("設定ファイルの読み込みに失敗しました: %s", exc)
            self._config = {}

    def save_config(self) -> None:
        """現在の設定をファイルに保存する。"""
        try:
            # 設定ディレクトリが存在しない場合は作成
            self.config_dir.mkdir(parents=True, exist_ok=True)
            
            with open(self.config_file, "w", encoding="utf-8") as f:
                json.dump(self._config, f, ensure_ascii=False, indent=2)
            logger.debug("設定を保存しました: %s", self.config_file)
        except Exception as exc:
            logger.warning("設定ファイルの保存に失敗しました: %s", exc)

    def get_output_dir(self) -> Path | None:
        """保存先フォルダの設定を取得する。"""
        output_dir_str = self._config.get("output_dir")
        if output_dir_str:
            try:
                return Path(output_dir_str)
            except Exception:
                logger.warning("無効な保存先フォルダ設定: %s", output_dir_str)
        return None

    def set_output_dir(self, output_dir: Path) -> None:
        """保存先フォルダの設定を保存する。"""
        self._config["output_dir"] = str(output_dir)
        self.save_config()

    def get_llm_profile(self) -> str | None:
        """LLMプロファイルの設定を取得する。"""
        return self._config.get("llm_profile")

    def set_llm_profile(self, profile: str) -> None:
        """LLMプロファイルの設定を保存する。"""
        self._config["llm_profile"] = profile
        self.save_config()

    def get_llm_provider(self) -> str | None:
        """LLMプロバイダーの設定を取得する。"""
        provider = self._config.get("llm_provider")
        if isinstance(provider, str) and provider.strip():
            return provider.strip().lower()
        return None

    def set_llm_provider(self, provider: str | None) -> None:
        """LLMプロバイダーの設定を保存する。"""
        if provider is None or not provider.strip():
            self._config.pop("llm_provider", None)
        else:
            self._config["llm_provider"] = provider.strip().lower()
        self.save_config()

    def get_pass5_provider(self) -> str | None:
        """Pass5で使用するLLMプロバイダーの設定を取得する。"""
        provider = self._config.get("pass5_provider")
        if isinstance(provider, str) and provider.strip():
            return provider.strip().lower()
        return None

    def set_pass5_provider(self, provider: str | None) -> None:
        """Pass5で使用するLLMプロバイダーの設定を保存する。"""
        if provider is None or not provider.strip():
            self._config.pop("pass5_provider", None)
        else:
            self._config["pass5_provider"] = provider.strip().lower()
        self.save_config()

    def get_window_geometry(self) -> str | None:
        """ウィンドウ位置とサイズの設定を取得する。"""
        return self._config.get("window_geometry")

    def set_window_geometry(self, geometry: str) -> None:
        """ウィンドウ位置とサイズの設定を保存する。"""
        self._config["window_geometry"] = geometry
        self.save_config()

    def get_google_api_key(self) -> str | None:
        """Google APIキーの設定を取得する。"""
        api_key = self._config.get("google_api_key")
        if isinstance(api_key, str) and api_key.strip():
            return api_key
        return None

    def set_google_api_key(self, api_key: str) -> None:
        """Google APIキーの設定を保存する。"""
        self._config["google_api_key"] = api_key
        self.save_config()

    def get_openai_api_key(self) -> str | None:
        """OpenAI APIキーの設定を取得する。"""
        api_key = self._config.get("openai_api_key")
        if isinstance(api_key, str) and api_key.strip():
            return api_key
        return None

    def set_openai_api_key(self, api_key: str) -> None:
        """OpenAI APIキーの設定を保存する。"""
        self._config["openai_api_key"] = api_key
        self.save_config()

    def get_anthropic_api_key(self) -> str | None:
        """Anthropic APIキーの設定を取得する。"""
        api_key = self._config.get("anthropic_api_key")
        if isinstance(api_key, str) and api_key.strip():
            return api_key
        return None

    def set_anthropic_api_key(self, api_key: str) -> None:
        """Anthropic APIキーの設定を保存する。"""
        self._config["anthropic_api_key"] = api_key
        self.save_config()

    # --- 辞書（Glossary） ---

    def get_glossary_terms(self) -> list[str]:
        """辞書（Glossary）の用語リストを取得する。

        - 未設定の場合は DEFAULT_GLOSSARY_TERMS を返す
        - 明示的に空（[]）が保存されている場合は空を返す
        """
        raw = self._config.get("glossary_terms", None)
        if raw is None:
            return list(DEFAULT_GLOSSARY_TERMS)
        if isinstance(raw, str):
            return parse_glossary_text(raw)
        if isinstance(raw, list):
            return normalize_glossary_terms([str(item) for item in raw])
        logger.warning("無効な辞書（Glossary）設定: %s", type(raw))
        return list(DEFAULT_GLOSSARY_TERMS)

    def set_glossary_terms(self, terms: list[str]) -> None:
        """辞書（Glossary）の用語リストを保存する。"""
        self._config["glossary_terms"] = normalize_glossary_terms(terms)
        self.save_config()

    # --- Pass1-5モデル設定 ---

    def get_pass_models(self) -> dict[str, str]:
        """Pass1-5のモデル設定を取得する。"""
        models = self._config.get("pass_models", {})
        if not isinstance(models, dict):
            return {}
        return models

    def set_pass_model(self, pass_name: str, model: str) -> None:
        """特定Passのモデル設定を保存する。"""
        if "pass_models" not in self._config:
            self._config["pass_models"] = {}
        self._config["pass_models"][pass_name] = model
        self.save_config()

    def get_pass_model(self, pass_name: str, default: str = "") -> str:
        """特定Passのモデル設定を取得する。"""
        models = self.get_pass_models()
        return models.get(pass_name, default)

    # --- 開始遅延 ---

    def get_start_delay(self) -> float:
        """開始遅延の設定を取得する。"""
        delay = self._config.get("start_delay", 0.2)
        try:
            return float(delay)
        except (TypeError, ValueError):
            return 0.2

    def set_start_delay(self, delay: float) -> None:
        """開始遅延の設定を保存する。"""
        self._config["start_delay"] = delay
        self.save_config()

    # --- Pass5設定 ---

    def get_pass5_enabled(self) -> bool:
        """Pass5の有効/無効を取得する。"""
        return bool(self._config.get("pass5_enabled", False))

    def set_pass5_enabled(self, enabled: bool) -> None:
        """Pass5の有効/無効を保存する。"""
        self._config["pass5_enabled"] = enabled
        self.save_config()

    def get_pass5_max_chars(self) -> int:
        """Pass5の文字数閾値を取得する。"""
        chars = self._config.get("pass5_max_chars", 17)
        try:
            return int(chars)
        except (TypeError, ValueError):
            return 17

    def set_pass5_max_chars(self, chars: int) -> None:
        """Pass5の文字数閾値を保存する。"""
        self._config["pass5_max_chars"] = chars
        self.save_config()

    def get_line_max_chars(self) -> int:
        """字幕1行の最大文字数（12〜20）を取得する。"""
        chars = self._config.get("line_max_chars", 17)
        try:
            value = int(chars)
        except (TypeError, ValueError):
            value = 17
        if value < 12:
            value = 12
        if value > 20:
            value = 20
        return value

    def set_line_max_chars(self, chars: int) -> None:
        """字幕1行の最大文字数（12〜20）を保存する。"""
        value = int(chars)
        if value < 12:
            value = 12
        if value > 20:
            value = 20
        self._config["line_max_chars"] = value
        self.save_config()

    def get_pass5_model(self) -> str | None:
        """Pass5（長行改行）のモデル名を取得する。"""
        model = self.get_pass_model("pass5", "").strip()
        if model:
            return model
        legacy = self._config.get("pass5_model")
        if isinstance(legacy, str) and legacy.strip():
            return legacy.strip()
        return None

    def set_pass5_model(self, model: str | None) -> None:
        """Pass5（長行改行）のモデル名を保存する。"""
        if model is None:
            if isinstance(self._config.get("pass_models"), dict):
                self._config["pass_models"].pop("pass5", None)
            self._config.pop("pass5_model", None)
        else:
            self._config.setdefault("pass_models", {})
            if isinstance(self._config.get("pass_models"), dict):
                self._config["pass_models"]["pass5"] = model
            self._config.pop("pass5_model", None)
        self.save_config()

    # --- ワークフロー設定 ---

    def get_workflow(self) -> str:
        """選択中のワークフローを取得する。"""
        from src.llm.workflows.registry import get_workflow

        return get_workflow(self._config.get("workflow")).slug

    def set_workflow(self, workflow: str) -> None:
        """ワークフローの設定を保存する。"""
        from src.llm.workflows.registry import get_workflow, is_known_workflow

        if is_known_workflow(workflow):
            self._config["workflow"] = get_workflow(workflow).slug
            self.save_config()

    # --- 抽出音声保存設定 ---

    def get_keep_extracted_audio(self) -> bool:
        """抽出した音声を保存するかどうかを取得する。"""
        return bool(self._config.get("keep_extracted_audio", False))

    def set_keep_extracted_audio(self, keep: bool) -> None:
        """抽出した音声を保存するかどうかを保存する。"""
        self._config["keep_extracted_audio"] = keep
        self.save_config()

    # --- Whisperランナー設定 ---

    def get_whisper_runner(self) -> str:
        """Whisperランナーの設定を取得する。デフォルトは 'openai'。"""
        runner = self._config.get("whisper_runner")
        if isinstance(runner, str) and runner.strip():
            return runner.strip().lower()
        return "openai"  # デフォルトはopenai-whisper (local)

    def set_whisper_runner(self, runner: str) -> None:
        """Whisperランナーの設定を保存する。"""
        self._config["whisper_runner"] = runner.strip().lower()
        self.save_config()

    # --- ログ保存設定 ---

    def get_save_logs(self) -> bool:
        """ログを保存するかどうかを取得する。"""
        return bool(self._config.get("save_logs", True))

    def set_save_logs(self, save: bool) -> None:
        """ログを保存するかどうかを保存する。"""
        self._config["save_logs"] = save
        self.save_config()

    # --- 完了通知設定 ---

    def get_notify_on_complete(self) -> bool:
        """完了時に通知するかどうかを取得する。"""
        return bool(self._config.get("notify_on_complete", False))

    def set_notify_on_complete(self, notify: bool) -> None:
        """完了時に通知するかどうかを保存する。"""
        self._config["notify_on_complete"] = notify
        self.save_config()


# グローバルインスタンス
_config_instance: GuiConfig | None = None


def get_config() -> GuiConfig:
    """設定マネージャーのグローバルインスタンスを取得する。"""
    global _config_instance
    if _config_instance is None:
        _config_instance = GuiConfig()
    return _config_instance


__all__ = ["GuiConfig", "get_config"]

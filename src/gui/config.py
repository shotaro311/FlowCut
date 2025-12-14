"""GUI設定の永続化を管理するモジュール。"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict

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
        delay = self._config.get("start_delay", 0.0)
        try:
            return float(delay)
        except (TypeError, ValueError):
            return 0.0

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


# グローバルインスタンス
_config_instance: GuiConfig | None = None


def get_config() -> GuiConfig:
    """設定マネージャーのグローバルインスタンスを取得する。"""
    global _config_instance
    if _config_instance is None:
        _config_instance = GuiConfig()
    return _config_instance


__all__ = ["GuiConfig", "get_config"]

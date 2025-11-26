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


# グローバルインスタンス
_config_instance: GuiConfig | None = None


def get_config() -> GuiConfig:
    """設定マネージャーのグローバルインスタンスを取得する。"""
    global _config_instance
    if _config_instance is None:
        _config_instance = GuiConfig()
    return _config_instance


__all__ = ["GuiConfig", "get_config"]

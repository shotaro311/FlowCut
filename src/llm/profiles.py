"""LLMプロファイル（プロバイダーとパスごとのモデル構成）を管理するユーティリティ。"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional, Set, List
import json
import logging

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class LlmProfile:
    name: str
    provider: str
    pass1_model: str | None = None
    pass2_model: str | None = None
    pass3_model: str | None = None
    pass4_model: str | None = None


_PROFILE_CACHE: Dict[str, LlmProfile] | None = None


def _load_profiles() -> Dict[str, LlmProfile]:
    """config/llm_profiles.json を読み込み、名前→LlmProfile の辞書に変換する。"""
    global _PROFILE_CACHE
    if _PROFILE_CACHE is not None:
        return _PROFILE_CACHE

    path = Path("config/llm_profiles.json")
    if not path.exists():
        logger.warning("LLMプロファイル定義が見つかりません: %s", path)
        _PROFILE_CACHE = {}
        return _PROFILE_CACHE

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        profiles_raw = raw.get("profiles", {})
        if not isinstance(profiles_raw, dict):
            raise ValueError("llm_profiles.json のフォーマットが不正です（profiles が dict ではありません）")
        profiles: Dict[str, LlmProfile] = {}
        for name, cfg in profiles_raw.items():
            if not isinstance(cfg, dict) or "provider" not in cfg:
                continue
            profiles[name] = LlmProfile(
                name=name,
                provider=str(cfg["provider"]),
                pass1_model=str(cfg.get("pass1_model")) if cfg.get("pass1_model") else None,
                pass2_model=str(cfg.get("pass2_model")) if cfg.get("pass2_model") else None,
                pass3_model=str(cfg.get("pass3_model")) if cfg.get("pass3_model") else None,
                pass4_model=str(cfg.get("pass4_model")) if cfg.get("pass4_model") else None,
            )
        _PROFILE_CACHE = profiles
        return profiles
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("LLMプロファイルの読み込みに失敗しました: %s", exc)
        _PROFILE_CACHE = {}
        return _PROFILE_CACHE


def get_profile(name: str) -> Optional[LlmProfile]:
    """指定された名前の LLM プロファイルを返す。存在しない場合は None。"""
    profiles = _load_profiles()
    return profiles.get(name)


def list_profiles() -> Dict[str, LlmProfile]:
    """利用可能な LLM プロファイル一覧を返す（name → LlmProfile）。

    GUI などでプリセット一覧を表示する用途を想定。
    """
    return dict(_load_profiles())


def list_models_by_provider() -> Dict[str, Set[str]]:
    """プロバイダーごとの既知モデル名一覧を返す。

    優先度:
    1. config/llm_profiles.json の `models` セクション
    2. プロファイル定義からの自動集約（後方互換用）
    """
    path = Path("config/llm_profiles.json")
    by_provider: Dict[str, Set[str]] = {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        models_raw: Dict[str, List[str]] = raw.get("models", {})  # type: ignore[assignment]
        if isinstance(models_raw, dict):
            for provider, items in models_raw.items():
                if not isinstance(items, list):
                    continue
                bucket = by_provider.setdefault(str(provider), set())
                for m in items:
                    if m:
                        bucket.add(str(m))
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("llm_profiles.json の models セクション読み込みに失敗しました: %s", exc)

    if by_provider:
        return by_provider

    # フォールバック: プロファイルから自動集約
    profiles = _load_profiles()
    for profile in profiles.values():
        prov = profile.provider
        if not prov:
            continue
        bucket = by_provider.setdefault(prov, set())
        for m in (profile.pass1_model, profile.pass2_model, profile.pass3_model, profile.pass4_model):
            if m:
                bucket.add(m)
    return by_provider


__all__ = ["LlmProfile", "get_profile", "list_profiles", "list_models_by_provider"]

"""LLM料金テーブルを読み込み、トークン数から概算コストを計算するユーティリティ。"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional
import json
import logging

logger = logging.getLogger(__name__)

_PRICING_CACHE: Dict[str, Any] | None = None


def _load_pricing() -> Dict[str, Any]:
    """config/llm_pricing.json を読み込み、結果をキャッシュする。"""
    global _PRICING_CACHE
    if _PRICING_CACHE is not None:
        return _PRICING_CACHE
    config_path = Path("config/llm_pricing.json")
    if not config_path.exists():
        logger.warning("LLM 料金テーブルが見つかりません: %s", config_path)
        _PRICING_CACHE = {}
        return _PRICING_CACHE
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("llm_pricing.json のフォーマットが不正です（dict ではありません）")
        _PRICING_CACHE = data
        return data
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("LLM 料金テーブルの読み込みに失敗しました: %s", exc)
        _PRICING_CACHE = {}
        return _PRICING_CACHE


def _find_by_prefix(mapping: Dict[str, Any], model: str) -> Optional[Dict[str, Any]]:
    """モデル名が微妙に異なる場合に、部分一致で探すヘルパー。

    例: pricing 側が "gpt-5-mini"、実際のモデル名が "gpt-5-mini-2025-01-01" など。
    """
    for key, value in mapping.items():
        if key in model or model in key:
            return value
    return None


def estimate_cost(
    provider: Optional[str],
    model: Optional[str],
    prompt_tokens: int,
    completion_tokens: int,
) -> Optional[Dict[str, float]]:
    """プロバイダー・モデル・トークン数から概算金額を計算する。

    戻り値は USD 想定の金額:
    {
      "input_cost_usd": ...,
      "output_cost_usd": ...,
      "total_cost_usd": ...
    }

    対応する料金テーブルが存在しない場合は None を返す。
    """
    if not provider or not model:
        return None
    table = _load_pricing()
    prov_table = table.get(provider)
    if not isinstance(prov_table, dict):
        return None

    pricing = prov_table.get(model)
    if pricing is None:
        pricing = _find_by_prefix(prov_table, model)
    if not pricing:
        return None

    try:
        in_m = float(pricing["input_per_million"])
        out_m = float(pricing["output_per_million"])
    except (KeyError, TypeError, ValueError):
        return None

    input_cost = (max(prompt_tokens, 0) / 1_000_000.0) * in_m
    output_cost = (max(completion_tokens, 0) / 1_000_000.0) * out_m
    total_cost = input_cost + output_cost
    return {
        "input_cost_usd": input_cost,
        "output_cost_usd": output_cost,
        "total_cost_usd": total_cost,
    }


__all__ = ["estimate_cost"]


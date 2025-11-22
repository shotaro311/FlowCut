"""Common API client for LLM providers."""
from __future__ import annotations

import json
from typing import Any, Dict, Optional

import requests

from .formatter import FormatterError


def post_json_request(
    url: str,
    payload: Dict[str, Any],
    headers: Optional[Dict[str, str]] = None,
    params: Optional[Dict[str, str]] = None,
    timeout: Optional[float] = None,
    error_prefix: str = "API request failed",
) -> Dict[str, Any]:
    """
    共通のJSON POSTリクエスト処理。
    
    Args:
        url: リクエストURL
        payload: JSONペイロード
        headers: リクエストヘッダー
        params: クエリパラメータ
        timeout: タイムアウト（秒）
        error_prefix: エラーメッセージのプレフィックス

    Returns:
        レスポンスのJSONデータ

    Raises:
        FormatterError: リクエスト失敗時やJSONパースエラー時
    """
    try:
        response = requests.post(
            url,
            headers=headers,
            params=params,
            json=payload,
            timeout=timeout,
        )
    except requests.RequestException as exc:
        raise FormatterError(f"{error_prefix}: {exc}") from exc

    if response.status_code >= 400:
        raise FormatterError(f"{error_prefix} Error: {response.status_code} {response.text}")

    try:
        return response.json()
    except json.JSONDecodeError as exc:
        raise FormatterError(f"{error_prefix}: 応答をJSONとして解釈できません: {response.text}") from exc

"""Environment-driven application settings."""
from __future__ import annotations

from dataclasses import dataclass
import os
from functools import lru_cache

try:  # pragma: no cover - 環境に応じてロード
    from dotenv import load_dotenv

    load_dotenv()  # リポジトリ直下の .env を読み込む
except Exception:
    pass


@dataclass(slots=True)
class LLMSettings:
    default_provider: str = "google"
    openai_api_key: str | None = None
    openai_model: str = "gpt-4o-mini"
    openai_whisper_model: str = "whisper-1"
    openai_base_url: str = "https://api.openai.com/v1"
    google_api_key: str | None = None
    google_model: str = "gemini-1.5-flash"
    google_api_base: str = "https://generativelanguage.googleapis.com/v1beta"
    anthropic_api_key: str | None = None
    anthropic_model: str = "claude-3-5-sonnet-20241022"
    anthropic_api_base: str = "https://api.anthropic.com/v1"
    request_timeout: float = 30.0


@dataclass(slots=True)
class AppSettings:
    llm: LLMSettings


def _env(key: str, default: str | None = None) -> str | None:
    value = os.getenv(key)
    if value is None:
        return default
    return value


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    llm = LLMSettings(
        default_provider=_env("DEFAULT_LLM_PROVIDER", "openai"),
        openai_api_key=_env("OPENAI_API_KEY"),
        openai_model=_env("OPENAI_MODEL", "gpt-4o-mini"),
        openai_whisper_model=_env("OPENAI_WHISPER_MODEL", "whisper-1"),
        openai_base_url=_env("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        google_api_key=_env("GOOGLE_API_KEY"),
        google_model=_env("GOOGLE_MODEL", "gemini-1.5-flash"),
        google_api_base=_env("GOOGLE_API_BASE", "https://generativelanguage.googleapis.com/v1beta"),
        anthropic_api_key=_env("ANTHROPIC_API_KEY"),
        anthropic_model=_env("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022"),
        anthropic_api_base=_env("ANTHROPIC_API_BASE", "https://api.anthropic.com/v1"),
        request_timeout=float(_env("LLM_REQUEST_TIMEOUT", "30.0")),
    )
    return AppSettings(llm=llm)


def reload_settings() -> None:
    get_settings.cache_clear()


__all__ = ["AppSettings", "LLMSettings", "get_settings", "reload_settings"]

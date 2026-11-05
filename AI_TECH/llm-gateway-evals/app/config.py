from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

# USD per 1M tokens (input, output). Prices change: verify against your provider's pricing page
# and override with PRICING_JSON before relying on the cost figures.
DEFAULT_PRICING = {"gpt-4.1-mini": (0.40, 1.60)}


def _parse_keys(raw: str) -> dict[str, str]:
    """API_KEYS='team-a:secret1,team-b:secret2' -> {'secret1': 'team-a', 'secret2': 'team-b'}"""
    keys: dict[str, str] = {}
    for item in raw.split(","):
        if ":" in item:
            name, _, secret = item.partition(":")
            if name.strip() and secret.strip():
                keys[secret.strip()] = name.strip()
    return keys


@dataclass(frozen=True)
class Settings:
    api_keys: dict[str, str] = field(default_factory=dict)  # secret -> client name
    openai_api_key: str = ""
    allowed_models: tuple[str, ...] = ("gpt-4.1-mini",)
    default_model: str = "gpt-4.1-mini"
    pricing: dict[str, tuple[float, float]] = field(default_factory=lambda: dict(DEFAULT_PRICING))
    rate_limit_per_minute: int = 60
    cache_ttl_seconds: float = 300.0
    max_retries: int = 2
    backoff_base_seconds: float = 0.5
    db_path: str = "./data/usage.db"
    request_timeout: float = 30.0

    @classmethod
    def from_env(cls) -> "Settings":
        env = os.environ.get
        pricing = dict(DEFAULT_PRICING)
        if env("PRICING_JSON"):
            pricing.update({m: (float(p[0]), float(p[1])) for m, p in json.loads(env("PRICING_JSON", "{}")).items()})
        models = tuple(m.strip() for m in env("ALLOWED_MODELS", "gpt-4.1-mini").split(",") if m.strip())
        return cls(
            api_keys=_parse_keys(env("API_KEYS", "")),
            openai_api_key=env("OPENAI_API_KEY", ""),
            allowed_models=models,
            default_model=env("DEFAULT_MODEL", models[0] if models else "gpt-4.1-mini"),
            pricing=pricing,
            rate_limit_per_minute=int(env("RATE_LIMIT_PER_MINUTE", "60")),
            cache_ttl_seconds=float(env("CACHE_TTL_SECONDS", "300")),
            max_retries=int(env("MAX_RETRIES", "2")),
            backoff_base_seconds=float(env("BACKOFF_BASE_SECONDS", "0.5")),
            db_path=env("DB_PATH", "./data/usage.db"),
            request_timeout=float(env("REQUEST_TIMEOUT", "30")),
        )

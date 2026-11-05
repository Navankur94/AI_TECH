from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    api_keys: tuple[str, ...] = ()
    openai_api_key: str = ""
    chat_model: str = "gpt-4.1-mini"
    kb_dir: str = "data/kb"
    confidence_threshold: float = 0.6
    request_timeout: float = 30.0

    @classmethod
    def from_env(cls) -> "Settings":
        env = os.environ.get
        return cls(
            api_keys=tuple(k.strip() for k in env("API_KEYS", "").split(",") if k.strip()),
            openai_api_key=env("OPENAI_API_KEY", ""),
            chat_model=env("CHAT_MODEL", "gpt-4.1-mini"),
            kb_dir=env("KB_DIR", "data/kb"),
            confidence_threshold=float(env("CONFIDENCE_THRESHOLD", "0.6")),
            request_timeout=float(env("REQUEST_TIMEOUT", "30")),
        )

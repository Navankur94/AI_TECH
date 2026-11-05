"""Runtime settings, read once from environment variables."""
from __future__ import annotations

import os
from dataclasses import dataclass


def _csv(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


@dataclass(frozen=True)
class Settings:
    api_keys: tuple[str, ...] = ()
    openai_api_key: str = ""
    chat_model: str = "gpt-4.1-mini"
    embedding_model: str = "text-embedding-3-small"
    embedder: str = "openai"  # "openai" or "hashing" (offline, for demos and tests)
    chroma_dir: str = "./chroma_data"
    collection: str = "knowledge"
    chunk_chars: int = 800
    chunk_overlap: int = 120
    top_k: int = 4
    min_score: float = 0.30
    max_upload_bytes: int = 5_000_000
    request_timeout: float = 30.0

    @classmethod
    def from_env(cls) -> "Settings":
        env = os.environ.get
        return cls(
            api_keys=_csv(env("API_KEYS", "")),
            openai_api_key=env("OPENAI_API_KEY", ""),
            chat_model=env("CHAT_MODEL", "gpt-4.1-mini"),
            embedding_model=env("EMBEDDING_MODEL", "text-embedding-3-small"),
            embedder=env("EMBEDDER", "openai").lower(),
            chroma_dir=env("CHROMA_DIR", "./chroma_data"),
            collection=env("CHROMA_COLLECTION", "knowledge"),
            chunk_chars=int(env("CHUNK_CHARS", "800")),
            chunk_overlap=int(env("CHUNK_OVERLAP", "120")),
            top_k=int(env("TOP_K", "4")),
            min_score=float(env("MIN_SCORE", "0.30")),
            max_upload_bytes=int(env("MAX_UPLOAD_BYTES", "5000000")),
            request_timeout=float(env("REQUEST_TIMEOUT", "30")),
        )

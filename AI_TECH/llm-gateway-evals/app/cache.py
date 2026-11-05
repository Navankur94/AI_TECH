from __future__ import annotations

import hashlib
import json
import threading
import time
from typing import Callable

from .providers import ChatRequest


def request_key(request: ChatRequest) -> str:
    payload = json.dumps(
        {"m": request.model, "msgs": request.messages, "t": request.temperature,
         "max": request.max_tokens, "json": request.json_mode}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


class TTLCache:
    """Small in-process cache. Use Redis when running more than one replica."""

    def __init__(self, ttl_seconds: float, max_items: int = 1000, clock: Callable[[], float] = time.monotonic) -> None:
        self.ttl = ttl_seconds
        self.max_items = max_items
        self.clock = clock
        self._items: dict[str, tuple[float, object]] = {}
        self._lock = threading.Lock()

    def get(self, key: str):
        with self._lock:
            entry = self._items.get(key)
            if entry is None:
                return None
            expires, value = entry
            if self.clock() >= expires:
                del self._items[key]
                return None
            return value

    def set(self, key: str, value: object) -> None:
        with self._lock:
            if len(self._items) >= self.max_items:
                self._items.pop(next(iter(self._items)))  # evict the oldest insertion
            self._items[key] = (self.clock() + self.ttl, value)

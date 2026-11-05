"""SQLite usage log: one row per request, with tokens, cost and latency."""
from __future__ import annotations

import os
import sqlite3
import threading
import time
from dataclasses import dataclass

SCHEMA = """
CREATE TABLE IF NOT EXISTS requests (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    client TEXT NOT NULL,
    model TEXT NOT NULL,
    status TEXT NOT NULL,
    input_tokens INTEGER NOT NULL DEFAULT 0,
    output_tokens INTEGER NOT NULL DEFAULT 0,
    cost_usd REAL NOT NULL DEFAULT 0,
    latency_ms REAL NOT NULL DEFAULT 0,
    cache_hit INTEGER NOT NULL DEFAULT 0,
    attempts INTEGER NOT NULL DEFAULT 1
)
"""


@dataclass(frozen=True)
class UsageRecord:
    client: str
    model: str
    status: str  # ok | error | rate_limited
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    cache_hit: bool = False
    attempts: int = 1


class UsageStore:
    def __init__(self, path: str = ":memory:") -> None:
        if path != ":memory:":
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock:
            self._db.execute(SCHEMA)
            self._db.commit()

    def log(self, record: UsageRecord) -> None:
        with self._lock:
            self._db.execute(
                "INSERT INTO requests (ts, client, model, status, input_tokens, output_tokens, cost_usd,"
                " latency_ms, cache_hit, attempts) VALUES (?,?,?,?,?,?,?,?,?,?)",
                (time.time(), record.client, record.model, record.status, record.input_tokens,
                 record.output_tokens, record.cost_usd, record.latency_ms, int(record.cache_hit), record.attempts),
            )
            self._db.commit()

    def summary(self, client: str | None = None) -> dict:
        where, params = ("WHERE client = ?", (client,)) if client else ("", ())
        with self._lock:
            row = self._db.execute(
                "SELECT COUNT(*) AS requests,"
                " COALESCE(SUM(status='ok'),0) AS ok,"
                " COALESCE(SUM(status='error'),0) AS errors,"
                " COALESCE(SUM(status='rate_limited'),0) AS rate_limited,"
                " COALESCE(SUM(cache_hit),0) AS cache_hits,"
                " COALESCE(SUM(input_tokens),0) AS input_tokens,"
                " COALESCE(SUM(output_tokens),0) AS output_tokens,"
                " COALESCE(SUM(cost_usd),0) AS cost_usd,"
                " COALESCE(AVG(CASE WHEN status='ok' THEN latency_ms END),0) AS avg_latency_ms"
                f" FROM requests {where}", params).fetchone()
        result = dict(row)
        result["cost_usd"] = round(result["cost_usd"], 8)
        result["avg_latency_ms"] = round(result["avg_latency_ms"], 1)
        return result

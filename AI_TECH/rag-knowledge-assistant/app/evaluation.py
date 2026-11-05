"""Retrieval evaluation: hit@k and mean reciprocal rank over a labeled question set."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .service import RAGService


@dataclass
class EvalResult:
    total: int
    hit_at_k: float
    mrr: float
    misses: list[dict]


def load_golden(path: str | Path) -> list[dict]:
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def ingest_folder(service: RAGService, folder: str | Path) -> int:
    total = 0
    for path in sorted(Path(folder).glob("*.md")):
        total += service.ingest(path.name, path.read_text(encoding="utf-8"))
    return total


def evaluate(service: RAGService, golden: list[dict], k: int = 3) -> EvalResult:
    hits, reciprocal, misses = 0, 0.0, []
    for row in golden:
        retrieved = service.retrieve(row["question"], k)
        sources = [h.source for h in retrieved]
        if row["expected_source"] in sources:
            hits += 1
            reciprocal += 1.0 / (sources.index(row["expected_source"]) + 1)
        else:
            misses.append({"question": row["question"], "expected": row["expected_source"], "got": sources})
    n = len(golden) or 1
    return EvalResult(total=len(golden), hit_at_k=hits / n, mrr=reciprocal / n, misses=misses)

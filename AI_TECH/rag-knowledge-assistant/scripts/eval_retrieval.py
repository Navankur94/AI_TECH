"""Run retrieval evaluation against the sample documents.

Offline:   EMBEDDER=hashing python scripts/eval_retrieval.py
With real embeddings: OPENAI_API_KEY=... python scripts/eval_retrieval.py
Exits with code 1 when hit@k falls below --min-hit-rate, so CI can gate on it.
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import chromadb  # noqa: E402

from app.config import Settings  # noqa: E402
from app.embeddings import HashingEmbedder, OpenAIEmbedder  # noqa: E402
from app.evaluation import evaluate, ingest_folder, load_golden  # noqa: E402
from app.service import RAGService  # noqa: E402
from app.store import VectorStore  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs", default="data/sample_docs")
    parser.add_argument("--golden", default="data/eval/golden.jsonl")
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--min-hit-rate", type=float, default=0.75)
    args = parser.parse_args()

    settings = Settings.from_env()
    if settings.embedder == "hashing":
        embedder = HashingEmbedder()
    else:
        from openai import OpenAI

        embedder = OpenAIEmbedder(OpenAI(api_key=settings.openai_api_key), settings.embedding_model)

    with tempfile.TemporaryDirectory() as tmp:
        store = VectorStore(chromadb.PersistentClient(path=tmp), "eval")
        service = RAGService(store, embedder, llm=None, settings=settings)  # type: ignore[arg-type]
        chunks = ingest_folder(service, args.docs)
        result = evaluate(service, load_golden(args.golden), args.k)

    print(f"embedder={settings.embedder} chunks={chunks} questions={result.total}")
    print(f"hit@{args.k}={result.hit_at_k:.2f} mrr={result.mrr:.2f}")
    for miss in result.misses:
        print("MISS:", miss)
    return 0 if result.hit_at_k >= args.min_hit_rate else 1


if __name__ == "__main__":
    raise SystemExit(main())

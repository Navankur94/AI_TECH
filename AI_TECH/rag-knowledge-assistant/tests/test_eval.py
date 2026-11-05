from pathlib import Path

import chromadb

from app.config import Settings
from app.embeddings import HashingEmbedder
from app.evaluation import evaluate, ingest_folder, load_golden
from app.service import RAGService
from app.store import VectorStore

ROOT = Path(__file__).resolve().parents[1]


def test_sample_corpus_retrieval_quality_offline(tmp_path):
    settings = Settings(min_score=0.0, chunk_chars=800, chunk_overlap=120)
    store = VectorStore(chromadb.PersistentClient(path=str(tmp_path)), "eval")
    service = RAGService(store, HashingEmbedder(), llm=None, settings=settings)  # type: ignore[arg-type]
    assert ingest_folder(service, ROOT / "data/sample_docs") >= 3
    result = evaluate(service, load_golden(ROOT / "data/eval/golden.jsonl"), k=3)
    assert result.total == 12
    assert result.hit_at_k >= 0.75, result.misses

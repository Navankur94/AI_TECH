import chromadb
import pytest

from app.config import Settings
from app.embeddings import HashingEmbedder
from app.service import RAGService
from app.store import VectorStore


class FakeLLM:
    """Records prompts and returns a scripted reply."""

    def __init__(self, reply: str = "Answer from the sources [1].") -> None:
        self.reply = reply
        self.calls: list[tuple[str, str]] = []

    def complete(self, system: str, user: str) -> str:
        self.calls.append((system, user))
        return self.reply


@pytest.fixture
def settings():
    return Settings(api_keys=("test-key",), min_score=0.1, chunk_chars=400, chunk_overlap=60, top_k=3)


@pytest.fixture
def llm():
    return FakeLLM()


@pytest.fixture
def service(tmp_path, settings, llm):
    store = VectorStore(chromadb.PersistentClient(path=str(tmp_path / "chroma")), "test")
    return RAGService(store, HashingEmbedder(), llm, settings)

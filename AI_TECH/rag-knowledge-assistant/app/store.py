"""ChromaDB-backed vector store. Embeddings are always supplied by us, never by Chroma."""
from __future__ import annotations

from dataclasses import dataclass

from .chunking import Chunk


@dataclass(frozen=True)
class Hit:
    id: str
    source: str
    section: str
    text: str
    score: float  # cosine similarity, 1.0 = identical direction


class VectorStore:
    def __init__(self, client, collection: str) -> None:
        self.collection = client.get_or_create_collection(name=collection, metadata={"hnsw:space": "cosine"})

    def replace_source(self, source: str, chunks: list[Chunk], embeddings: list[list[float]]) -> int:
        """Re-ingesting a source replaces its chunks, so re-indexing never creates duplicates."""
        self.delete_source(source)
        if not chunks:
            return 0
        self.collection.add(
            ids=[c.id for c in chunks],
            documents=[c.text for c in chunks],
            embeddings=embeddings,
            metadatas=[{"source": c.source, "section": c.section, "index": c.index} for c in chunks],
        )
        return len(chunks)

    def delete_source(self, source: str) -> int:
        existing = self.collection.get(where={"source": source}, include=[])
        ids = existing["ids"]
        if ids:
            self.collection.delete(ids=ids)
        return len(ids)

    def sources(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for meta in self.collection.get(include=["metadatas"])["metadatas"] or []:
            counts[meta["source"]] = counts.get(meta["source"], 0) + 1
        return dict(sorted(counts.items()))

    def query(self, embedding: list[float], k: int) -> list[Hit]:
        total = self.collection.count()
        if total == 0:
            return []
        result = self.collection.query(
            query_embeddings=[embedding], n_results=min(k, total), include=["documents", "metadatas", "distances"]
        )
        hits = []
        for id_, doc, meta, dist in zip(
            result["ids"][0], result["documents"][0], result["metadatas"][0], result["distances"][0]
        ):
            hits.append(Hit(id=id_, source=meta["source"], section=meta["section"], text=doc, score=1.0 - float(dist)))
        return hits

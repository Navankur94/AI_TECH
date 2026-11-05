"""Embedding backends: OpenAI for real use, a hashing embedder for offline demos and tests."""
from __future__ import annotations

import hashlib
import math
import re
from typing import Protocol, Sequence

STOPWORDS = frozenset(
    "a an and are as at be by can do does for from has have how i in is it its of on or our "
    "that the their there this to was what when where which who will with you your".split()
)


def tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOPWORDS]


class Embedder(Protocol):
    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


class HashingEmbedder:
    """Deterministic bag-of-words embedding. Needs no network, so tests and demos run offline.

    It captures word overlap only, not meaning. Use OpenAIEmbedder for real semantic search.
    """

    def __init__(self, dim: int = 512) -> None:
        self.dim = dim

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            vec = [0.0] * self.dim
            for token in tokenize(text):
                digest = hashlib.blake2b(token.encode(), digest_size=8).digest()
                number = int.from_bytes(digest, "big")
                vec[number % self.dim] += 1.0 if (number >> 63) & 1 else -1.0
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            vectors.append([v / norm for v in vec])
        return vectors


class OpenAIEmbedder:
    def __init__(self, client, model: str = "text-embedding-3-small", batch_size: int = 64) -> None:
        self.client = client
        self.model = model
        self.batch_size = batch_size

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = list(texts[start:start + self.batch_size])
            response = self.client.embeddings.create(model=self.model, input=batch)
            ordered = sorted(response.data, key=lambda item: item.index)
            vectors.extend(list(item.embedding) for item in ordered)
        return vectors

from __future__ import annotations

from collections.abc import Sequence
from hashlib import sha256
from typing import Any

from rag.schema.model_protocols import Embedder


class FallbackEmbeddingRepo(Embedder):
    def __init__(self, dimension: int = 8) -> None:
        if dimension <= 0:
            raise ValueError("dimension must be positive")
        self._dimension = dimension

    @property
    def provider_name(self) -> str:
        return "fallback"

    @property
    def embedding_model_name(self) -> str:
        return f"hash-{self._dimension}"

    def embed(self, texts: Sequence[str], **_: Any) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            digest = sha256(text.encode("utf-8")).digest()
            vectors.append([digest[index] / 255.0 for index in range(self._dimension)])
        return vectors


__all__ = ["FallbackEmbeddingRepo"]

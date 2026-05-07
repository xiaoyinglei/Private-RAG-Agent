from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import httpx

from rag.schema.model_protocols import Embedder


class OllamaEmbedder(Embedder):
    """
    Ollama 向量化专员。
    只负责把文本转成向量，不负责生成和重排。
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        default_model: str = "qwen3-embedding:8b",
        timeout_seconds: float = 120.0,
        batch_size: int = 8,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        self._base_url = base_url.rstrip("/")
        self._default_model = default_model
        self._batch_size = batch_size
        self._client = httpx.Client(
            timeout=httpx.Timeout(timeout_seconds),
            trust_env=False,
        )

    @property
    def embedding_model_name(self) -> str:
        return self._default_model

    def embed(self, texts: Sequence[str], **kwargs: Any) -> list[list[float]]:
        if not texts:
            return []

        model = kwargs.pop("model", self._default_model)
        batch_size = int(kwargs.pop("batch_size", self._batch_size))

        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        all_vectors: list[list[float]] = []

        for start in range(0, len(texts), batch_size):
            batch = list(texts[start : start + batch_size])

            request_payload = {
                "model": model,
                "input": batch,
            }

            try:
                response = self._client.post(
                    f"{self._base_url}/api/embed",
                    json=request_payload,
                )
                response.raise_for_status()
                response_payload = response.json()
            except httpx.HTTPError as exc:
                raise RuntimeError(f"Ollama embedding request failed: {exc}") from exc

            embeddings = response_payload.get("embeddings")
            if not isinstance(embeddings, list):
                raise RuntimeError("Ollama embedding response missing 'embeddings'")

            batch_vectors: list[list[float]] = []
            for vector in embeddings:
                if not isinstance(vector, list):
                    raise RuntimeError("Ollama embedding response contains invalid vector")
                try:
                    batch_vectors.append([float(x) for x in vector])
                except (TypeError, ValueError) as exc:
                    raise RuntimeError("Ollama embedding response contains non-numeric vector values") from exc

            if len(batch_vectors) != len(batch):
                raise RuntimeError(
                    f"Ollama embedding count mismatch: expected {len(batch)}, got {len(batch_vectors)}"
                )

            all_vectors.extend(batch_vectors)

        if len(all_vectors) != len(texts):
            raise RuntimeError(
                f"Ollama embedding total count mismatch: expected {len(texts)}, got {len(all_vectors)}"
            )

        return all_vectors

    def close(self) -> None:
        self._client.close()

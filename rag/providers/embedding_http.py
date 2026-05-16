from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any

import httpx
from pydantic import BaseModel, Field


class EmbeddingHealthResponse(BaseModel):
    model: str
    embedding_space: str = "default"
    dimension: int


class EmbeddingRequest(BaseModel):
    texts: list[str]
    mode: str = "document"
    batch_size: int | None = None


class EmbeddingResponse(BaseModel):
    vectors: list[list[float]]
    dimension: int


class EmbeddingHttpClient:
    """HTTP client for a remote embedding service.

    Talks to a standalone FastAPI embedding-service process.
    Validates health on construction; enforces dimension and count
    invariants on every embed call.
    """

    def __init__(
        self,
        base_url: str,
        *,
        timeout: float = 120.0,
        client: httpx.Client | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._client = client or httpx.Client(timeout=httpx.Timeout(timeout), trust_env=False)
        self._model_name: str = ""
        self._embedding_space: str = "default"
        self._dim: int = 0
        self.last_latency_ms: float = 0.0
        self._health_check()

    # ── public properties ──

    @property
    def embedding_model_name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        return self._dim

    # ── Embedder protocol ──

    def embed(self, texts: Sequence[str], **kwargs: Any) -> list[list[float]]:
        if not texts:
            return []

        mode = str(kwargs.get("mode", "document"))
        batch_size = kwargs.get("batch_size")

        request_body = EmbeddingRequest(
            texts=list(texts),
            mode=mode,
            batch_size=int(batch_size) if batch_size is not None else None,
        )

        t0 = time.monotonic()
        response = self._client.post(
            f"{self._base_url}/v1/embeddings",
            json=request_body.model_dump(),
        )
        self.last_latency_ms = (time.monotonic() - t0) * 1000.0

        if response.status_code != 200:
            detail = _extract_detail(response)
            raise RuntimeError(
                f"Embedding service error (HTTP {response.status_code}): {detail}"
            )

        parsed = EmbeddingResponse.model_validate(response.json())

        if len(parsed.vectors) != len(texts):
            raise RuntimeError(
                f"Embedding count mismatch: expected {len(texts)}, got {len(parsed.vectors)}"
            )

        if parsed.dimension != self._dim:
            raise RuntimeError(
                f"Embedding dimension mismatch: expected {self._dim}, got {parsed.dimension}"
            )

        return parsed.vectors

    def embed_query(self, texts: Sequence[str]) -> list[list[float]]:
        return self.embed(texts, mode="query")

    def close(self) -> None:
        self._client.close()

    # ── internal ──

    def _health_check(self) -> None:
        try:
            response = self._client.get(f"{self._base_url}/health")
        except httpx.RequestError as exc:
            raise RuntimeError(
                f"Embedding service health check failed: {exc}"
            ) from exc

        if response.status_code != 200:
            detail = _extract_detail(response)
            raise RuntimeError(
                f"Embedding service health check failed (HTTP {response.status_code}): {detail}"
            )

        try:
            health = EmbeddingHealthResponse.model_validate(response.json())
        except Exception as exc:
            raise RuntimeError(
                f"Embedding service health check returned invalid response: {exc}"
            ) from exc

        if not health.model:
            raise RuntimeError("Embedding service health check missing 'model'")
        if health.dimension <= 0:
            raise RuntimeError(
                f"Embedding service health check invalid dimension: {health.dimension}"
            )

        self._model_name = health.model
        self._embedding_space = health.embedding_space
        self._dim = health.dimension


def _extract_detail(response: httpx.Response) -> str:
    try:
        body = response.json()
        if isinstance(body, dict):
            return str(body.get("detail", response.text))
    except Exception:
        pass
    return response.text[:500]

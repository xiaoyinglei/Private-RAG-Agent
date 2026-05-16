from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any

import httpx
from pydantic import BaseModel


class RerankHealthResponse(BaseModel):
    model: str


class RerankRequest(BaseModel):
    query: str
    documents: list[str]
    max_length: int | None = None
    batch_size: int | None = None


class RerankResponse(BaseModel):
    scores: list[float]


class RerankHttpClient:
    """HTTP client for a remote rerank service.

    Talks to a standalone FastAPI rerank-service process.
    Validates health on construction; enforces score count
    invariants on every rerank call.
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
        self.last_latency_ms: float = 0.0
        self._health_check()

    # ── public properties ──

    @property
    def rerank_model_name(self) -> str:
        return self._model_name

    # ── Reranker protocol ──

    def rerank(self, query: str, documents: Sequence[str], **kwargs: Any) -> list[float]:
        if not documents:
            return []

        max_length = kwargs.get("max_length")
        batch_size = kwargs.get("batch_size")

        request_body = RerankRequest(
            query=query,
            documents=list(documents),
            max_length=int(max_length) if max_length is not None else None,
            batch_size=int(batch_size) if batch_size is not None else None,
        )

        t0 = time.monotonic()
        response = self._client.post(
            f"{self._base_url}/v1/rerank",
            json=request_body.model_dump(),
        )
        self.last_latency_ms = (time.monotonic() - t0) * 1000.0

        if response.status_code != 200:
            detail = _extract_detail(response)
            raise RuntimeError(
                f"Rerank service error (HTTP {response.status_code}): {detail}"
            )

        parsed = RerankResponse.model_validate(response.json())

        if len(parsed.scores) != len(documents):
            raise RuntimeError(
                f"Rerank score count mismatch: expected {len(documents)}, got {len(parsed.scores)}"
            )

        return parsed.scores

    def close(self) -> None:
        self._client.close()

    # ── internal ──

    def _health_check(self) -> None:
        try:
            response = self._client.get(f"{self._base_url}/health")
        except httpx.RequestError as exc:
            raise RuntimeError(
                f"Rerank service health check failed: {exc}"
            ) from exc

        if response.status_code != 200:
            detail = _extract_detail(response)
            raise RuntimeError(
                f"Rerank service health check failed (HTTP {response.status_code}): {detail}"
            )

        try:
            health = RerankHealthResponse.model_validate(response.json())
        except Exception as exc:
            raise RuntimeError(
                f"Rerank service health check returned invalid response: {exc}"
            ) from exc

        if not health.model:
            raise RuntimeError("Rerank service health check missing 'model'")

        self._model_name = health.model


def _extract_detail(response: httpx.Response) -> str:
    try:
        body = response.json()
        if isinstance(body, dict):
            return str(body.get("detail", response.text))
    except Exception:
        pass
    return response.text[:500]

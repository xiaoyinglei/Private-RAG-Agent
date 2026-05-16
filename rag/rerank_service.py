from __future__ import annotations

import asyncio
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

_MAX_DOCUMENTS = 256
_MAX_LENGTH = 1024
_MAX_REQUEST_BYTES = 10 * 1024 * 1024  # 10 MB


class RerankRequest(BaseModel):
    query: str = Field(min_length=1)
    documents: list[str] = Field(max_length=_MAX_DOCUMENTS, min_length=1)
    max_length: int | None = Field(default=None, ge=1, le=_MAX_LENGTH)
    batch_size: int | None = Field(default=None, ge=1, le=64)


class RerankResponse(BaseModel):
    scores: list[float]


class HealthResponse(BaseModel):
    model: str


app = FastAPI(title="rag-rerank-service")
_reranker: Any = None
_model_name: str = ""
_semaphore = asyncio.Semaphore(1)


@app.on_event("startup")
async def _warmup() -> None:
    pass


def create_app(
    model_name_or_path: str = "BAAI/bge-reranker-v2-m3",
    *,
    model_path: str | None = None,
    batch_size: int = 8,
    max_length: int = _MAX_LENGTH,
    use_fp16: bool = False,
    local_files_only: bool = True,
    devices: str | None = None,
    normalize: bool = False,
) -> FastAPI:
    """Create and warmup the FastAPI app with a FlagEmbedding reranker."""
    from rag.providers.huggingface.rerank import FlagEmbeddingReranker

    global _reranker, _model_name

    _reranker = FlagEmbeddingReranker(
        model_name_or_path=model_name_or_path,
        model_path=model_path,
        batch_size=batch_size,
        max_length=max_length,
        use_fp16=use_fp16,
        local_files_only=local_files_only,
        devices=devices,
        normalize=normalize,
    )
    _model_name = _reranker.rerank_model_name

    # warmup
    _reranker.rerank("warmup", ["warmup document"], max_length=32)

    return app


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    if _reranker is None:
        raise HTTPException(status_code=503, detail="reranker not loaded")
    return HealthResponse(model=_model_name)


@app.post("/v1/rerank", response_model=RerankResponse)
async def rerank(request: RerankRequest) -> RerankResponse:
    if _reranker is None:
        raise HTTPException(status_code=503, detail="reranker not loaded")

    async with _semaphore:
        loop = asyncio.get_running_loop()
        try:
            scores: list[float] = await loop.run_in_executor(
                None,
                lambda: _reranker.rerank(
                    request.query,
                    request.documents,
                    max_length=request.max_length or _MAX_LENGTH,
                    batch_size=request.batch_size or 8,
                ),
            )
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    if len(scores) != len(request.documents):
        raise HTTPException(
            status_code=500,
            detail=f"count mismatch: expected {len(request.documents)}, got {len(scores)}",
        )

    return RerankResponse(scores=scores)

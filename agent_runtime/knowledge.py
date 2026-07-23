from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict


class RAGKnowledgeConfig(BaseModel):
    """Serializable, secret-free configuration for one RAG knowledge store."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    storage_root: Path = Path(".rag")
    embedding_model: str | None = None
    reranker_model: str | None = None
    vector_backend: Literal["milvus", "sqlite"] = "milvus"
    vector_namespace: str | None = None
    vector_collection_prefix: str | None = None


__all__ = ["RAGKnowledgeConfig"]

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class ModelCapability(StrEnum):
    CHAT = "chat"
    EMBEDDING = "embedding"
    RERANKER = "reranker"


@dataclass(frozen=True, slots=True)
class ModelSpec:
    alias: str
    capability: ModelCapability
    provider: str
    model: str
    base_url: str | None = None
    api_key_env: str | None = None
    embedding_space: str | None = None

    @property
    def requires_api_key(self) -> bool:
        return self.api_key_env is not None


@dataclass(frozen=True, slots=True)
class ModelRuntimeConfig:
    primary_model: ModelSpec
    embedding_model: ModelSpec
    reranker_model: ModelSpec | None = None

from __future__ import annotations

from dataclasses import dataclass, field
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
class GenerationTaskConfig:
    """Per-task generation parameters.

    model       — model alias in models.yaml; None = fallback to defaults.primary_model
    max_tokens  — max completion tokens; None = not configured, consumer decides fallback
    temperature — None = don't pass to LLM (use model default)
    """

    model: str | None = None
    max_tokens: int | None = None
    temperature: float | None = None


@dataclass(frozen=True, slots=True)
class GenerationConfig:
    summary: GenerationTaskConfig = field(default_factory=GenerationTaskConfig)
    answer: GenerationTaskConfig = field(default_factory=GenerationTaskConfig)
    planner: GenerationTaskConfig = field(default_factory=GenerationTaskConfig)
    synthesize: GenerationTaskConfig = field(default_factory=GenerationTaskConfig)
    factcheck: GenerationTaskConfig = field(default_factory=GenerationTaskConfig)


@dataclass(frozen=True, slots=True)
class ModelRuntimeConfig:
    primary_model: ModelSpec
    embedding_model: ModelSpec
    reranker_model: ModelSpec | None = None
    generation: GenerationConfig = field(default_factory=GenerationConfig)

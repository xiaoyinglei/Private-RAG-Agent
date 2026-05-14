from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

def _provider_name(provider: object) -> str:
    explicit = getattr(provider, "provider_name", None)
    if isinstance(explicit, str) and explicit:
        return explicit
    fallback = getattr(provider, "name", None)
    if isinstance(fallback, str) and fallback:
        return fallback
    normalized = provider.__class__.__name__.removesuffix("ProviderRepo").removesuffix("Repo")
    return normalized.replace("_", "-").lower() or "unknown"


def _provider_model(provider: object, capability: str) -> str | None:
    attribute_names = {
        "chat": ("chat_model_name", "model_name_or_path", "_chat_model", "_model", "_model_name_or_path"),
        "embedding": ("embedding_model_name", "model_name_or_path", "_embedding_model", "_model_name_or_path"),
        "rerank": ("rerank_model_name", "_rerank_model"),
    }.get(capability, ())
    for attribute_name in attribute_names:
        value = getattr(provider, attribute_name, None)
        if isinstance(value, str) and value:
            return value
    return None


def _supports_capability(provider: object, capability: str) -> bool:
    configured_name = {
        "chat": "is_chat_configured",
        "embedding": "is_embed_configured",
        "rerank": "is_rerank_configured",
    }[capability]
    if capability == "chat":
        supported = callable(getattr(provider, "generate_text", None)) or callable(getattr(provider, "chat", None))
    elif capability == "embedding":
        supported = callable(getattr(provider, "embed", None))
    else:
        supported = callable(getattr(provider, "rerank", None))
    if not supported:
        return False
    configured = getattr(provider, configured_name, True)
    return bool(configured)


@dataclass(frozen=True, slots=True)
class EmbeddingCapabilityBinding:
    backend: object
    space: str
    location: str = "runtime"
    provider_name: str = field(init=False)
    model_name: str | None = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "provider_name", _provider_name(self.backend))
        object.__setattr__(self, "model_name", _provider_model(self.backend, "embedding"))

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        embed = getattr(self.backend, "embed", None)
        if not callable(embed):
            raise RuntimeError("Embedding capability is not available")
        return list(embed(list(texts)))

    def embed_query(self, texts: Sequence[str]) -> list[list[float]]:
        embed_query = getattr(self.backend, "embed_query", None)
        if callable(embed_query):
            return list(embed_query(list(texts)))
        return self.embed(texts)

    def embed_query_sparse(self, texts: Sequence[str]) -> list[dict[int, float]]:
        embed_query_sparse = getattr(self.backend, "embed_query_sparse", None)
        if callable(embed_query_sparse):
            return [self._normalize_sparse_vector(item) for item in embed_query_sparse(list(texts))]
        embed_sparse = getattr(self.backend, "embed_sparse", None)
        if callable(embed_sparse):
            return [self._normalize_sparse_vector(item) for item in embed_sparse(list(texts))]
        raise RuntimeError("Sparse embedding capability is not available")

    def supports_sparse_embedding(self) -> bool:
        return callable(getattr(self.backend, "embed_query_sparse", None)) or callable(
            getattr(self.backend, "embed_sparse", None)
        )

    @staticmethod
    def _normalize_sparse_vector(value: object) -> dict[int, float]:
        if isinstance(value, dict):
            normalized: dict[int, float] = {}
            for key, item in value.items():
                try:
                    normalized[int(key)] = float(item)
                except (TypeError, ValueError):
                    continue
            if normalized:
                return normalized
        if isinstance(value, list):
            normalized = {}
            for item in value:
                if not isinstance(item, (tuple, list)) or len(item) != 2:
                    continue
                try:
                    normalized[int(item[0])] = float(item[1])
                except (TypeError, ValueError):
                    continue
            if normalized:
                return normalized
        raise RuntimeError(f"Unsupported sparse vector payload: {type(value)!r}")


@dataclass(frozen=True, slots=True)
class ChatCapabilityBinding:
    backend: object
    location: str = "runtime"
    provider_name: str = field(init=False)
    model_name: str | None = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "provider_name", _provider_name(self.backend))
        object.__setattr__(self, "model_name", _provider_model(self.backend, "chat"))

    def chat(self, prompt: str, **kwargs: Any) -> str:
        chat = getattr(self.backend, "chat", None)
        if callable(chat):
            return str(chat(prompt, **kwargs))
        generate_text = getattr(self.backend, "generate_text", None)
        if callable(generate_text):
            return str(generate_text(prompt=prompt, **kwargs))
        raise RuntimeError("Chat capability is not available")


@dataclass(frozen=True, slots=True)
class RerankCapabilityBinding:
    backend: object
    location: str = "runtime"
    provider_name: str = field(init=False)
    model_name: str | None = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "provider_name", _provider_name(self.backend))
        object.__setattr__(self, "model_name", _provider_model(self.backend, "rerank"))

    def rerank(self, query: str, documents: Sequence[str], **kwargs: object) -> list[float]:
        rerank = getattr(self.backend, "rerank", None)
        if not callable(rerank):
            raise RuntimeError("Rerank capability is not available")
        return [float(score) for score in rerank(query, list(documents), **kwargs)]


CapabilityBinding = EmbeddingCapabilityBinding | ChatCapabilityBinding | RerankCapabilityBinding

__all__ = [
    "CapabilityBinding",
    "ChatCapabilityBinding",
    "EmbeddingCapabilityBinding",
    "RerankCapabilityBinding",
    "_supports_capability",
]

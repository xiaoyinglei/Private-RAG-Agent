from __future__ import annotations

import os

from rag.assembly.models import AssemblyOverrides, ProviderConfig
from rag.models.config import ModelRuntimeConfig, ModelSpec


_PROVIDER_KIND_MAP: dict[str, str] = {
    "openai_compatible": "openai-compatible",
    "ollama": "ollama",
    "sentence_transformers": "local-bge",
    "mlx_embedding": "mlx-embedding",
}


def to_assembly_overrides(config: ModelRuntimeConfig) -> AssemblyOverrides:
    """Convert ModelRuntimeConfig to AssemblyOverrides.

    ONLY converts configuration — does NOT create provider instances.
    Provider instantiation stays in rag.assembly.support.build_provider.
    """
    return AssemblyOverrides(
        chat=_to_chat_provider_config(config.primary_model),
        embedding=_to_embedding_provider_config(config.embedding_model),
        rerank=_to_reranker_provider_config(config.reranker_model),
    )


def _to_chat_provider_config(spec: ModelSpec) -> ProviderConfig:
    return ProviderConfig(
        provider_kind=_map_kind(spec.provider),
        chat_model=spec.model,
        base_url=spec.base_url,
        api_key=_resolve_api_key(spec),
    )


def _to_embedding_provider_config(spec: ModelSpec) -> ProviderConfig:
    return ProviderConfig(
        provider_kind=_map_kind(spec.provider),
        embedding_model=spec.model,
        base_url=spec.base_url,
    )


def _to_reranker_provider_config(spec: ModelSpec | None) -> ProviderConfig | None:
    if spec is None:
        return None
    return ProviderConfig(
        provider_kind=_map_kind(spec.provider),
        rerank_model=spec.model,
    )


def _map_kind(provider: str) -> str:
    mapped = _PROVIDER_KIND_MAP.get(provider)
    if mapped is not None:
        return mapped
    raise ValueError(f"Unknown provider: {provider!r}")


def _resolve_api_key(spec: ModelSpec) -> str | None:
    if spec.api_key_env:
        return os.environ.get(spec.api_key_env)
    return None

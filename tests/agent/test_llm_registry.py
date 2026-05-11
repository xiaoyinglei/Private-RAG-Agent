from __future__ import annotations

import pytest

from rag.agent.core.llm_config import AgentModelsConfig, ModelProvider, ModelSpec
from rag.agent.core.llm_registry import (
    ModelRegistry,
    UnknownModelAliasError,
)


def _ollama_spec(model: str = "test-model") -> ModelSpec:
    return ModelSpec(
        provider=ModelProvider.OLLAMA,
        model=model,
        base_url="http://localhost:11434",
    )


def _make_config(
    *,
    default_model: str = "main",
    fallback_model: str | None = None,
) -> AgentModelsConfig:
    models: dict[str, ModelSpec] = {"main": _ollama_spec("main-model")}
    if fallback_model:
        models["fast"] = _ollama_spec("fast-model")
    return AgentModelsConfig(
        models=models,
        default_model=default_model,
        fallback_model=fallback_model,
    )


class TestModelRegistryProperties:
    def test_default_model(self) -> None:
        reg = ModelRegistry(_make_config(default_model="main"))
        assert reg.default_model == "main"

    def test_fallback_model_none(self) -> None:
        reg = ModelRegistry(_make_config())
        assert reg.fallback_model is None

    def test_fallback_model_set(self) -> None:
        reg = ModelRegistry(_make_config(fallback_model="fast"))
        assert reg.fallback_model == "fast"


class TestModelRegistryResolve:
    def test_unknown_alias_raises(self) -> None:
        reg = ModelRegistry(_make_config())
        with pytest.raises(UnknownModelAliasError):
            reg.resolve("nonexistent")

    def test_resolve_returns_generator(self) -> None:
        reg = ModelRegistry(_make_config())
        resolved = reg.resolve("main")
        assert resolved.generator is not None
        assert resolved.kwargs["max_tokens"] == 2048

    def test_caches_same_alias(self) -> None:
        reg = ModelRegistry(_make_config())
        r1 = reg.resolve("main")
        r2 = reg.resolve("main")
        assert r1.generator is r2.generator

    def test_kwargs_include_model_defaults(self) -> None:
        spec = ModelSpec(
            provider=ModelProvider.OLLAMA,
            model="x",
            max_tokens=512,
            defaults={"temperature": 0.3, "top_p": 0.8},
        )
        config = AgentModelsConfig(
            models={"test": spec},
            default_model="test",
        )
        reg = ModelRegistry(config)
        resolved = reg.resolve("test")
        assert resolved.kwargs["max_tokens"] == 512
        assert resolved.kwargs["temperature"] == 0.3
        assert resolved.kwargs["top_p"] == 0.8


class TestModelRegistryResolveOrFallback:
    def test_falls_back_when_alias_unknown(self) -> None:
        reg = ModelRegistry(_make_config(fallback_model="fast"))
        resolved = reg.resolve_or_fallback("missing")
        # 应该降级到 fallback_model="fast"
        assert resolved.generator is not None

    def test_does_not_infinite_loop_when_fallback_also_missing(self) -> None:
        reg = ModelRegistry(_make_config())
        with pytest.raises(UnknownModelAliasError):
            reg.resolve_or_fallback("missing")


class TestModelRegistryResolveForNode:
    def test_uses_explicit_node_model(self) -> None:
        config = _make_config(fallback_model="fast")
        reg = ModelRegistry(config)
        resolved = reg.resolve_for_node(node_model="fast", node_name="route")
        assert resolved.generator is not None

    def test_uses_default_when_node_model_is_none(self) -> None:
        reg = ModelRegistry(_make_config())
        resolved = reg.resolve_for_node(node_model=None, node_name="route")
        assert resolved.generator is not None

from __future__ import annotations

import pytest
import yaml

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


def test_load_configs_models_maps_openai_compatible_protocol(tmp_path) -> None:
    config_path = tmp_path / "models.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "models": {
                    "qwen3_8b_mlx_4bit": {
                        "capability": "chat",
                        "provider": "qwen",
                        "protocol": "openai_compatible",
                        "model": "Qwen/Qwen3-8B-MLX-4bit",
                        "base_url": "http://127.0.0.1:8080/v1",
                    }
                },
                "defaults": {"primary_model": "qwen3_8b_mlx_4bit"},
            }
        ),
        encoding="utf-8",
    )

    config = ModelRegistry._load_yaml_file(config_path)

    assert config.default_model == "qwen3_8b_mlx_4bit"
    assert config.models["qwen3_8b_mlx_4bit"].provider is ModelProvider.OPENAI_COMPATIBLE
    assert config.models["qwen3_8b_mlx_4bit"].base_url == "http://127.0.0.1:8080/v1"


def test_load_configs_models_preserves_api_key_env_for_cloud_models(tmp_path, monkeypatch) -> None:
    config_path = tmp_path / "models.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "models": {
                    "mimo_cloud": {
                        "capability": "chat",
                        "provider": "mimo",
                        "protocol": "openai_compatible",
                        "model": "mimo-v2-flash",
                        "base_url": "https://api.xiaomimimo.com/v1",
                        "api_key_env": "MIMO_API_KEY",
                    }
                },
                "defaults": {"primary_model": "mimo_cloud"},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("MIMO_API_KEY", "sk-test")

    config = ModelRegistry._load_yaml_file(config_path)
    provider_config = ModelRegistry(config)._spec_to_provider_config(config.models["mimo_cloud"])

    assert config.models["mimo_cloud"].api_key_env == "MIMO_API_KEY"
    assert provider_config.api_key == "sk-test"


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
        resolved = reg.resolve_for_node(node_model="fast", node_name="retrieval_hint")
        assert resolved.generator is not None

    def test_uses_default_when_node_model_is_none(self) -> None:
        reg = ModelRegistry(_make_config())
        resolved = reg.resolve_for_node(node_model=None, node_name="retrieval_hint")
        assert resolved.generator is not None

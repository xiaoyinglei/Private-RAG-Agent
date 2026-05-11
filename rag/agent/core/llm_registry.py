from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rag.agent.core.llm_config import AgentModelsConfig, ModelProvider, ModelSpec
from rag.assembly.models import ProviderConfig
from rag.assembly.support import build_provider


class UnknownModelAliasError(KeyError):
    """别名在 models 中不存在。"""


class ModelNotAvailableError(RuntimeError):
    """模型构造失败（加载出错等）。"""


@dataclass(slots=True)
class ResolvedModel:
    generator: object
    kwargs: dict[str, Any]


class ModelRegistry:
    """按 alias 解析并缓存 Generator 实例。

    加载顺序：RAG_AGENT_MODELS_PATH(YAML) > RAG_AGENT_MODELS(JSON) > models.yaml 内置默认
    """

    _BUNDLED_CONFIG_PATH = Path(__file__).resolve().parent.parent / "models.yaml"

    def __init__(self, config: AgentModelsConfig) -> None:
        self._config = config
        self._cache: dict[str, ResolvedModel] = {}

    @property
    def default_model(self) -> str:
        return self._config.default_model

    @property
    def fallback_model(self) -> str | None:
        return self._config.fallback_model

    @classmethod
    def from_env(cls, env_path: str = ".env") -> ModelRegistry:
        config = cls._load_config()
        return cls(config)

    @classmethod
    def _load_config(cls) -> AgentModelsConfig:
        # 1. RAG_AGENT_MODELS_PATH → YAML 文件
        yaml_path = os.environ.get("RAG_AGENT_MODELS_PATH")
        if yaml_path:
            return cls._load_yaml_file(Path(yaml_path))

        # 2. RAG_AGENT_MODELS → JSON 字符串
        json_text = os.environ.get("RAG_AGENT_MODELS")
        if json_text:
            return AgentModelsConfig.model_validate(json.loads(json_text))

        # 3. 内置 models.yaml（相对于 rag/agent/models.yaml）
        if cls._BUNDLED_CONFIG_PATH.is_file():
            return cls._load_yaml_file(cls._BUNDLED_CONFIG_PATH)

        raise FileNotFoundError(
            "No agent model config found. Set RAG_AGENT_MODELS_PATH, "
            "RAG_AGENT_MODELS, or ensure rag/agent/models.yaml exists."
        )

    @staticmethod
    def _load_yaml_file(path: Path) -> AgentModelsConfig:
        import yaml

        with open(path, encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        return AgentModelsConfig.model_validate(data)

    def resolve(self, alias: str) -> ResolvedModel:
        """别名 → (Generator, kwargs)。按 alias 缓存，同 alias 多次调用返回同一 Generator。"""
        if alias in self._cache:
            return self._cache[alias]

        spec = self._config.models.get(alias)
        if spec is None:
            raise UnknownModelAliasError(f"Model alias {alias!r} not found in config")

        provider_config = self._spec_to_provider_config(spec)
        try:
            provider = build_provider(provider_config)
        except Exception as exc:
            raise ModelNotAvailableError(f"Failed to build provider for {alias!r}: {exc}") from exc

        generator = getattr(provider, "generator", None)
        if generator is None:
            raise ModelNotAvailableError(f"Provider for {alias!r} does not support chat generation")

        kwargs: dict[str, Any] = {"max_tokens": spec.max_tokens, **spec.defaults}
        resolved = ResolvedModel(generator=generator, kwargs=kwargs)
        self._cache[alias] = resolved
        return resolved

    def resolve_or_fallback(self, alias: str) -> ResolvedModel:
        """尝试解析 alias，失败时降级到 fallback_model。"""
        try:
            return self.resolve(alias)
        except (UnknownModelAliasError, ModelNotAvailableError):
            if self._config.fallback_model and alias != self._config.fallback_model:
                return self.resolve(self._config.fallback_model)
            raise

    def resolve_for_node(
        self,
        *,
        node_model: str | None,
        node_name: str,
    ) -> ResolvedModel:
        """根据节点指定的 model alias（可为 None）解析 Generator。

        node_model 非空 → 直接用该 alias（失败降级到 fallback）
        node_model 为空 → 用 default_model（失败降级到 fallback）
        """
        alias = node_model or self._config.default_model
        return self.resolve_or_fallback(alias)

    @staticmethod
    def _spec_to_provider_config(spec: ModelSpec) -> ProviderConfig:
        if spec.provider == ModelProvider.MLX:
            return ProviderConfig(
                provider_kind="local-hf",
                chat_backend="mlx",
                chat_model=spec.model,
            )
        if spec.provider == ModelProvider.OLLAMA:
            return ProviderConfig(
                provider_kind="ollama",
                base_url=spec.base_url or "http://localhost:11434",
                chat_model=spec.model,
            )
        raise ValueError(f"Unsupported provider: {spec.provider}")

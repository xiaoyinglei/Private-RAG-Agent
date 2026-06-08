from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rag.agent.core.llm_config import AgentModelsConfig, ModelProvider, ModelSpec
from rag.assembly.models import ProviderConfig
from rag.assembly.support import build_provider
from rag.assembly.tokenizer import TokenAccountingService, TokenizerContract
from rag.providers.llm_gateway import LLMGateway
from rag.schema.llm import parse_llm_stage_budgets


class UnknownModelAliasError(KeyError):
    """别名在 models 中不存在。"""


class ModelNotAvailableError(RuntimeError):
    """模型构造失败（加载出错等）。"""


@dataclass(slots=True)
class ResolvedModel:
    generator: object
    kwargs: dict[str, Any]
    context_window_tokens: int = 32_768
    gateway: LLMGateway | None = None
    token_accounting: TokenAccountingService | None = None


class ModelRegistry:
    """按 alias 解析并缓存 Generator 实例。

    加载顺序：RAG_AGENT_MODELS_PATH(YAML) > RAG_AGENT_MODELS(JSON) > models.yaml 内置默认
    """

    _BUNDLED_CONFIG_PATH = Path("configs/models.yaml")

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
    def from_env(cls, env_path: str = ".env", *, default_model: str | None = None) -> ModelRegistry:
        _load_env_file(Path(env_path))
        config = cls._load_config()
        if default_model is not None:
            if default_model not in config.models:
                raise UnknownModelAliasError(f"Model alias {default_model!r} not found in config")
            config = config.model_copy(
                update={
                    "default_model": default_model,
                    "fallback_model": default_model,
                }
            )
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

        # Support configs/models.yaml: models keyed by alias plus defaults.
        raw_models = data.get("models", {})
        defaults = data.get("defaults", {})

        agent_models: dict[str, dict[str, object]] = {}
        for alias, entry in raw_models.items():
            if not isinstance(entry, dict):
                continue
            if entry.get("capability") != "chat":
                continue
            agent_models[alias] = {
                "provider": _agent_provider_kind(entry),
                "model": entry["model"],
                "max_tokens": entry.get("max_tokens", 2048),
                "base_url": entry.get("base_url"),
                "api_key_env": entry.get("api_key_env"),
                "context_window_tokens": entry.get("context_window_tokens", 32_768),
            }

        default_model = defaults.get("primary_model", "")
        if not default_model and agent_models:
            default_model = next(iter(agent_models))

        return AgentModelsConfig.model_validate({
            "version": data.get("version", 1),
            "models": agent_models,
            "default_model": default_model,
            "fallback_model": data.get("fallback_model", default_model),
            "llm_stage_budgets": parse_llm_stage_budgets(data.get("llm_budgets")),
        })

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
        token_accounting = TokenAccountingService(
            TokenizerContract(
                embedding_model_name=spec.model,
                tokenizer_model_name=spec.model,
                chunking_tokenizer_model_name=spec.model,
                tokenizer_backend="auto",
                max_context_tokens=spec.context_window_tokens,
                prompt_reserved_tokens=512,
                local_files_only=True,
            )
        )
        resolved = ResolvedModel(
            generator=generator,
            kwargs=kwargs,
            context_window_tokens=spec.context_window_tokens,
            gateway=LLMGateway(
                generator=generator,
                token_accounting=token_accounting,
                model_context_tokens=spec.context_window_tokens,
                stage_budgets=self._config.llm_stage_budgets,
            ),
            token_accounting=token_accounting,
        )
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
                provider_kind="openai-compatible",
                base_url=spec.base_url or "http://127.0.0.1:8080/v1",
                chat_model=spec.model,
                api_key=_api_key_from_env(spec.api_key_env),
            )
        if spec.provider == ModelProvider.OLLAMA:
            return ProviderConfig(
                provider_kind="ollama",
                base_url=spec.base_url or "http://localhost:11434",
                chat_model=spec.model,
            )
        if spec.provider == ModelProvider.OPENAI_COMPATIBLE:
            return ProviderConfig(
                provider_kind="openai-compatible",
                base_url=spec.base_url or "http://127.0.0.1:8080/v1",
                chat_model=spec.model,
                api_key=_api_key_from_env(spec.api_key_env),
            )
        raise ValueError(f"Unsupported provider: {spec.provider}")


def _agent_provider_kind(entry: dict[str, object]) -> str:
    protocol = _normalized_provider_value(entry.get("protocol"))
    provider = _normalized_provider_value(entry.get("provider"))
    if protocol == "openai_compatible":
        return ModelProvider.OPENAI_COMPATIBLE.value
    if provider in {"openai_compatible", "qwen", "deepseek"}:
        return ModelProvider.OPENAI_COMPATIBLE.value
    if provider == "ollama":
        return ModelProvider.OLLAMA.value
    if provider == "mlx":
        return ModelProvider.MLX.value
    return provider


def _normalized_provider_value(value: object) -> str:
    return str(value or "").strip().lower().replace("-", "_")


def _api_key_from_env(name: str | None) -> str | None:
    if not name:
        return None
    value = os.environ.get(name)
    return value.strip() if isinstance(value, str) and value.strip() else None


def _load_env_file(path: Path) -> None:
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, raw_value = stripped.split("=", maxsplit=1)
        key = key.strip()
        if not key or key in os.environ:
            continue
        os.environ[key] = _parse_env_value(raw_value.strip())


def _parse_env_value(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value

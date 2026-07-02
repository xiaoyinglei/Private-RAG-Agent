from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

from rag.agent.core.llm_config import ModelProvider
from rag.agent.core.llm_config import ModelSpec as InternalModelSpec
from rag.agent.core.llm_registry import (
    ModelRegistry,
    ModelResolver,
    ResolvedModel,
    UnknownModelAliasError,
)
from rag.models.config import GenerationConfig

ModelLocation = Literal["local", "cloud"]
ModelSwitchRequester = Literal["user", "agent", "system"]


class ModelPolicyError(ValueError):
    """A model switch request was rejected by policy."""


@dataclass(frozen=True, slots=True)
class ModelSpec:
    id: str
    provider: str
    provider_model: str
    context_window: int
    supports_tools: bool
    supports_structured_output: bool
    location: ModelLocation
    protocol: str | None = None
    base_url: str | None = None
    api_key_env: str | None = None
    max_output_tokens: int = 2048
    input_cost_per_1m: float | None = None
    output_cost_per_1m: float | None = None


class ModelCatalog:
    """Runtime-facing model catalog built from model declarations."""

    def __init__(
        self,
        *,
        specs: Mapping[str, ModelSpec],
        default_model_id: str,
    ) -> None:
        if not specs:
            raise ValueError("model catalog must not be empty")
        if default_model_id not in specs:
            raise UnknownModelAliasError(
                f"Default model {default_model_id!r} not found in catalog"
            )
        self._specs = dict(specs)
        self.default_model_id = default_model_id

    @classmethod
    def from_config_file(cls, path: Path) -> ModelCatalog:
        return cls.from_registry(
            ModelRegistry(ModelRegistry._load_yaml_file(path))
        )

    @classmethod
    def from_env(cls, env_path: str = ".env") -> ModelCatalog:
        return cls.from_registry(ModelRegistry.from_env(env_path=env_path))

    @classmethod
    def from_registry(cls, registry: ModelRegistry) -> ModelCatalog:
        specs = {
            model_id: _to_public_spec(model_id, registry.get_model_spec(model_id))
            for model_id in registry.model_ids
        }
        return cls(specs=specs, default_model_id=registry.default_model)

    def list_models(self) -> list[ModelSpec]:
        return [self._specs[model_id] for model_id in self._specs]

    def get(self, model_id: str) -> ModelSpec:
        try:
            return self._specs[model_id]
        except KeyError as exc:
            raise UnknownModelAliasError(f"Model alias {model_id!r} not found in catalog") from exc

    def has(self, model_id: str) -> bool:
        return model_id in self._specs


@dataclass(slots=True)
class ModelSessionState:
    """Mutable model choice for one runtime session."""

    current_model_id: str

    @classmethod
    def load(cls, path: Path, *, default_model_id: str) -> ModelSessionState:
        if not path.is_file():
            return cls(current_model_id=default_model_id)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return cls(current_model_id=default_model_id)
        current = payload.get("current_model_id")
        if isinstance(current, str) and current.strip():
            return cls(current_model_id=current.strip())
        return cls(current_model_id=default_model_id)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {"current_model_id": self.current_model_id},
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )


@dataclass(frozen=True, slots=True)
class ModelPolicy:
    """Policy gate for model switches. This is not a router."""

    allowed_user_model_ids: frozenset[str] | None = None
    allowed_agent_model_ids: frozenset[str] | None = None
    allowed_system_model_ids: frozenset[str] | None = None

    def review_switch(
        self,
        *,
        catalog: ModelCatalog,
        target_model_id: str,
        requested_by: ModelSwitchRequester,
    ) -> ModelSpec:
        spec = catalog.get(target_model_id)
        allowed = self._allowed_ids_for(requested_by)
        if allowed is not None and target_model_id not in allowed:
            raise ModelPolicyError(
                f"Model {target_model_id!r} is not allowed for {requested_by} requests"
            )
        return spec

    def _allowed_ids_for(
        self,
        requested_by: ModelSwitchRequester,
    ) -> frozenset[str] | None:
        if requested_by == "agent":
            return self.allowed_agent_model_ids
        if requested_by == "system":
            return self.allowed_system_model_ids
        return self.allowed_user_model_ids


class ModelControlPlane:
    """Shared facade for model catalog, session choice, policy, and resolution."""

    def __init__(
        self,
        *,
        catalog: ModelCatalog,
        state: ModelSessionState,
        policy: ModelPolicy | None = None,
        registry: ModelResolver | None = None,
        session_path: Path | None = None,
    ) -> None:
        if not catalog.has(state.current_model_id):
            raise UnknownModelAliasError(
                f"Model alias {state.current_model_id!r} not found in catalog"
            )
        self.catalog = catalog
        self.state = state
        self.policy = policy or ModelPolicy()
        self._registry = registry
        self._session_path = session_path

    @classmethod
    def from_config_file(
        cls,
        path: Path,
        *,
        initial_model_id: str | None = None,
        session_path: Path | None = None,
        policy: ModelPolicy | None = None,
    ) -> ModelControlPlane:
        registry = ModelRegistry(ModelRegistry._load_yaml_file(path))
        return cls.from_registry(
            registry,
            initial_model_id=initial_model_id,
            session_path=session_path,
            policy=policy,
        )

    @classmethod
    def from_env(
        cls,
        env_path: str = ".env",
        *,
        initial_model_id: str | None = None,
        session_path: Path | None = None,
        policy: ModelPolicy | None = None,
    ) -> ModelControlPlane:
        registry = ModelRegistry.from_env(env_path=env_path)
        return cls.from_registry(
            registry,
            initial_model_id=initial_model_id,
            session_path=session_path,
            policy=policy,
        )

    @classmethod
    def from_registry(
        cls,
        registry: ModelRegistry,
        *,
        initial_model_id: str | None = None,
        session_path: Path | None = None,
        policy: ModelPolicy | None = None,
    ) -> ModelControlPlane:
        catalog = ModelCatalog.from_registry(registry)
        state = _load_session_state(
            catalog=catalog,
            initial_model_id=initial_model_id,
            session_path=session_path,
        )
        return cls(
            catalog=catalog,
            state=state,
            policy=policy,
            registry=registry,
            session_path=session_path,
        )

    @property
    def default_model(self) -> str:
        return self.state.current_model_id

    @property
    def fallback_model(self) -> str | None:
        if self._registry is None:
            return None
        return self._registry.fallback_model

    @property
    def generation_config(self) -> GenerationConfig:
        if self._registry is None:
            raise RuntimeError("Model resolver is not configured")
        return self._registry.generation_config

    def list_models(self) -> list[ModelSpec]:
        return self.catalog.list_models()

    def current_model(self) -> ModelSpec:
        return self.catalog.get(self.state.current_model_id)

    def switch_model(
        self,
        model_id: str,
        *,
        requested_by: ModelSwitchRequester,
        persist: bool = True,
    ) -> ModelSpec:
        spec = self.policy.review_switch(
            catalog=self.catalog,
            target_model_id=model_id,
            requested_by=requested_by,
        )
        self.state.current_model_id = spec.id
        if persist and self._session_path is not None:
            self.state.save(self._session_path)
        return spec

    def request_model_switch(self, model_id: str) -> ModelSpec:
        return self.switch_model(model_id, requested_by="agent")

    def resolve_or_fallback(self, alias: str) -> ResolvedModel:
        if self._registry is None:
            raise RuntimeError("Model resolver is not configured")
        return self._registry.resolve_or_fallback(alias)

    def resolve_for_node(
        self,
        *,
        node_model: str | None,
        node_name: str,
    ) -> ResolvedModel:
        del node_name
        model_id = node_model or self.state.current_model_id
        return self.resolve_or_fallback(model_id)


def _load_session_state(
    *,
    catalog: ModelCatalog,
    initial_model_id: str | None,
    session_path: Path | None,
) -> ModelSessionState:
    if initial_model_id is not None:
        state = ModelSessionState(current_model_id=initial_model_id)
    elif session_path is not None:
        state = ModelSessionState.load(
            session_path,
            default_model_id=catalog.default_model_id,
        )
    else:
        state = ModelSessionState(current_model_id=catalog.default_model_id)
    if not catalog.has(state.current_model_id):
        raise UnknownModelAliasError(
            f"Model alias {state.current_model_id!r} not found in catalog"
        )
    return state


def _to_public_spec(
    model_id: str,
    spec: InternalModelSpec,
) -> ModelSpec:
    provider = str(spec.provider_name or spec.provider)
    provider_model = str(spec.model)
    base_url = spec.base_url
    location = spec.location or _infer_location(
        base_url=base_url,
        provider=spec.provider,
    )
    return ModelSpec(
        id=model_id,
        provider=provider,
        provider_model=provider_model,
        context_window=int(spec.context_window_tokens),
        supports_tools=bool(spec.supports_tools),
        supports_structured_output=bool(spec.supports_structured_output),
        location=location,
        protocol=spec.protocol,
        base_url=base_url,
        api_key_env=spec.api_key_env,
        max_output_tokens=int(spec.max_tokens),
        input_cost_per_1m=spec.input_cost_per_1m,
        output_cost_per_1m=spec.output_cost_per_1m,
    )


def _infer_location(
    *,
    base_url: str | None,
    provider: object,
) -> ModelLocation:
    if provider in {ModelProvider.MLX, ModelProvider.OLLAMA}:
        return "local"
    if base_url:
        host = urlparse(base_url).hostname or ""
        if host in {"localhost", "127.0.0.1", "::1"}:
            return "local"
    return "cloud"


def format_model_rows(
    specs: Iterable[ModelSpec],
    *,
    current_model_id: str,
) -> list[str]:
    lines = []
    for spec in specs:
        marker = "*" if spec.id == current_model_id else " "
        caps = []
        if spec.supports_tools:
            caps.append("tools")
        if spec.supports_structured_output:
            caps.append("structured")
        cap_text = ",".join(caps) if caps else "-"
        cost = "-"
        if spec.input_cost_per_1m is not None or spec.output_cost_per_1m is not None:
            cost = f"{spec.input_cost_per_1m or 0:g}/{spec.output_cost_per_1m or 0:g}"
        lines.append(
            f"{marker} {spec.id}  provider={spec.provider}  "
            f"model={spec.provider_model}  ctx={spec.context_window}  "
            f"{spec.location}  caps={cap_text}  cost={cost}"
        )
    return lines


__all__ = [
    "ModelCatalog",
    "ModelControlPlane",
    "ModelLocation",
    "ModelPolicy",
    "ModelPolicyError",
    "ModelSessionState",
    "ModelSpec",
    "ModelSwitchRequester",
    "format_model_rows",
]

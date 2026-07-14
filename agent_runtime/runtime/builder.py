from __future__ import annotations

import logging
import os
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, cast

from agent_runtime.models import ModelControlPlane

if TYPE_CHECKING:
    from rag.agent.core.definition import AgentRuntimePolicy
    from rag.agent.core.registry import AgentRegistry
    from rag.agent.core.runtime_diagnostics import RuntimeDiagnostic
    from rag.agent.service import AgentService
    from rag.agent.skills.runtime import SkillRuntime
    from rag.agent.streaming.sink import StreamEventSink
    from rag.agent.tools.tool import Tool

logger = logging.getLogger(__name__)

DEFAULT_VECTOR_BACKEND = "milvus"
CLI_AGENT_CHOICES = ("generic",)


@dataclass(frozen=True)
class AutoRAGConfig:
    storage_root: Path
    vector_backend: str
    vector_dsn: str | None
    vector_namespace: str | None
    vector_collection_prefix: str | None
    explicit: bool


def resolve_auto_rag_config(
    *,
    storage_root: Path,
    vector_backend: str,
    vector_dsn: str | None,
    vector_namespace: str | None,
    vector_collection_prefix: str | None,
) -> AutoRAGConfig:
    effective_storage_root = storage_root
    env_storage_root = (
        os.environ.get("AGENT_RAG_STORAGE_ROOT")
        or os.environ.get("RAG_STORAGE_ROOT")
        or os.environ.get("STORAGE_ROOT")
    )
    if storage_root == Path(".rag") and env_storage_root:
        effective_storage_root = Path(env_storage_root)

    env_vector_backend = os.environ.get("AGENT_VECTOR_BACKEND") or os.environ.get("VECTOR_BACKEND")
    env_vector_dsn = os.environ.get("AGENT_VECTOR_DSN") or os.environ.get("VECTOR_DSN")
    env_vector_namespace = os.environ.get("AGENT_VECTOR_NAMESPACE") or os.environ.get("VECTOR_NAMESPACE")
    env_vector_prefix = os.environ.get("AGENT_VECTOR_PREFIX") or os.environ.get("VECTOR_PREFIX")
    return AutoRAGConfig(
        storage_root=effective_storage_root,
        vector_backend=env_vector_backend or vector_backend,
        vector_dsn=vector_dsn or env_vector_dsn,
        vector_namespace=vector_namespace or env_vector_namespace,
        vector_collection_prefix=vector_collection_prefix or env_vector_prefix,
        explicit=(
            storage_root != Path(".rag")
            or bool(env_storage_root)
            or vector_backend != DEFAULT_VECTOR_BACKEND
            or bool(env_vector_backend)
            or bool(vector_dsn)
            or bool(env_vector_dsn)
            or bool(vector_namespace)
            or bool(env_vector_namespace)
            or bool(vector_collection_prefix)
            or bool(env_vector_prefix)
        ),
    )


def looks_like_rag_storage(storage_root: Path) -> bool:
    return any(
        (storage_root / marker).exists()
        for marker in ("metadata.sqlite3", "vectors.sqlite3", "index.sqlite")
    )


def build_model_control_plane(
    *,
    model_alias: str | None = None,
    session_path: Path | None = None,
) -> ModelControlPlane:
    return ModelControlPlane.from_env(
        initial_model_id=model_alias,
        session_path=session_path,
    )


def resolve_cli_agent_definition(
    agent_registry: AgentRegistry,
    agent_type: str,
) -> AgentRuntimePolicy:
    if agent_type not in CLI_AGENT_CHOICES:
        allowed = ", ".join(CLI_AGENT_CHOICES)
        raise ValueError(f"{agent_type!r} is not a supported CLI agent. Allowed: {allowed}")
    return agent_registry.get(agent_type)


def build_agent_service(
    runtime: Any | None,
    *,
    checkpoint_db: Path | None = None,
    agent_type: str = "generic",
    model_alias: str | None = None,
    model_control_plane: ModelControlPlane | None = None,
    runtime_diagnostics: Sequence[RuntimeDiagnostic] = (),
    knowledge_runner: Callable[..., object] | None = None,
    mcp_tools: Sequence[Tool] = (),
    skill_tools: Sequence[Tool] = (),
    subagent_tools: Sequence[Tool] = (),
    discoverable_tools: Sequence[Tool] = (),
    skill_runtime: SkillRuntime | None = None,
    stream_sink: StreamEventSink | None = None,
    strict_model_provider: bool = True,
    startup_ms: float = 0.0,
) -> AgentService:
    """Build one CLI/SDK runtime from ordinary canonical Tool values."""
    from rag.agent.builtin import create_builtin_agent_registry
    from rag.agent.core.checkpointing import create_agent_checkpointer
    from rag.agent.core.runtime_diagnostics import (
        AgentLatencyProfile,
        RuntimeDiagnostic,
    )
    from rag.agent.service import AgentService
    from rag.agent.tools.builtins import create_resident_coding_tools
    from rag.agent.tools.integrations.knowledge import (
        KnowledgeSearchInput,
        create_knowledge_tools,
    )
    from rag.agent.tools.permissions import ToolExecutionContext
    from rag.agent.tools.registry import build_tool_registry
    from rag.agent.tools.selection import (
        create_find_tools_tool,
        find_tools,
    )
    from rag.agent.workspace import WorkspaceRuntime, create_temp_workspace

    build_started_at = time.perf_counter()
    model_ready_ms = 0.0
    agent_registry = create_builtin_agent_registry()
    definition = resolve_cli_agent_definition(agent_registry, agent_type)
    workspace = (
        runtime
        if isinstance(runtime, WorkspaceRuntime)
        else create_temp_workspace()
    )
    plan_revision = 0

    def update_plan(arguments: Mapping[str, object]) -> dict[str, object]:
        nonlocal plan_revision
        del arguments
        plan_revision += 1
        return {
            "accepted": True,
            "revision": plan_revision,
            "message": "Plan updated.",
        }

    resident_tools = create_resident_coding_tools(
        workspace,
        plan_updater=cast(Any, update_plan),
    )

    async def search_knowledge(arguments: Mapping[str, object]) -> object:
        if knowledge_runner is None:
            raise RuntimeError("knowledge search is not configured")
        payload = KnowledgeSearchInput.model_validate(arguments)
        result = knowledge_runner(
            payload,
            execution_context=ToolExecutionContext(),
        )
        if hasattr(result, "__await__"):
            return await result
        return result

    knowledge_tools = create_knowledge_tools(
        search_knowledge if knowledge_runner is not None else None
    )
    tool_registry = build_tool_registry(
        resident_tools,
        knowledge_tools,
        mcp_tools,
        skill_tools,
        subagent_tools,
        discoverable_tools,
    )
    configured_resident_names = tuple(
        tool.definition.name
        for source in (knowledge_tools, skill_tools)
        for tool in source
    )
    discoverable_tool_names = tuple(
        tool.definition.name
        for source in (mcp_tools, subagent_tools, discoverable_tools)
        for tool in source
    )
    if discoverable_tool_names:
        discovery_snapshot = MappingProxyType(
            {
                tool.definition.name: tool
                for tool in tool_registry.list_all()
            }
        )

        def search_hidden_tools(query: str, limit: int) -> object:
            return find_tools(
                discovery_snapshot,
                query=query,
                discoverable_names=discoverable_tool_names,
                limit=limit,
                max_active_tools=definition.max_active_deferred_tools,
            )

        tool_registry.register(create_find_tools_tool(search_hidden_tools))
    definition = replace(
        definition,
        core_tool_names=tuple(
            tool.definition.name for tool in resident_tools
        ),
        deferred_tool_names=(
            *configured_resident_names,
            *discoverable_tool_names,
        ),
    )
    diagnostics: tuple[RuntimeDiagnostic, ...] = tuple(runtime_diagnostics)
    try:
        model_ready_started_at = time.perf_counter()
        model_registry = model_control_plane or build_model_control_plane(
            model_alias=model_alias,
        )
        model_ready_ms = (time.perf_counter() - model_ready_started_at) * 1000
    except Exception as exc:
        if strict_model_provider:
            raise
        model_registry = None
        diagnostics = (
            *diagnostics,
            RuntimeDiagnostic.from_exception(
                code="model_registry_initialization_failed",
                component="model_registry",
                error=exc,
            ),
        )

    return AgentService(
        definition=definition,
        tool_registry=tool_registry,
        model_registry=model_registry,
        checkpointer=create_agent_checkpointer(checkpoint_db),
        runtime_diagnostics=diagnostics,
        strict_model_provider=strict_model_provider,
        latency_profile=AgentLatencyProfile(
            startup_ms=startup_ms,
            build_service_ms=(time.perf_counter() - build_started_at) * 1000,
            model_ready_ms=model_ready_ms,
        ),
        workspace=workspace,
        configured_resident_tool_names=configured_resident_names,
        discoverable_tool_names=discoverable_tool_names,
        skill_runtime=skill_runtime,
        stream_sink=stream_sink,
    )


def build_optional_rag_runtime(
    *,
    storage_root: Path,
    model_alias: str | None,
    embedding_model_alias: str | None,
    reranker_model_alias: str | None,
    vector_backend: str,
    vector_dsn: str | None,
    vector_namespace: str | None,
    vector_collection_prefix: str | None,
    explicit: bool = False,
) -> tuple[Any | None, tuple[RuntimeDiagnostic, ...]]:
    from rag.agent.core.runtime_diagnostics import RuntimeDiagnostic

    if not explicit:
        return None, ()
    rag_config = resolve_auto_rag_config(
        storage_root=storage_root,
        vector_backend=vector_backend,
        vector_dsn=vector_dsn,
        vector_namespace=vector_namespace,
        vector_collection_prefix=vector_collection_prefix,
    )
    if not rag_config.storage_root.exists():
        return None, ()
    if not rag_config.explicit and not looks_like_rag_storage(rag_config.storage_root):
        return None, ()
    try:
        from rag import AssemblyRequest, CapabilityRequirements, RAGRuntime
        from rag.models.assembly_adapter import to_assembly_overrides
        from rag.models.runtime import RuntimeOverrides, resolve_runtime_config
        from rag.retrieval import QueryOptions
        from rag.storage.runtime_config import runtime_storage_config

        runtime_config = resolve_runtime_config(
            RuntimeOverrides(
                model_alias=model_alias,
                embedding_model_alias=embedding_model_alias,
                reranker_model_alias=reranker_model_alias or "none",
            )
        )
        assembly_overrides = to_assembly_overrides(runtime_config)
        storage = runtime_storage_config(
            rag_config.storage_root,
            vector_backend=rag_config.vector_backend,
            vector_dsn=rag_config.vector_dsn,
            vector_namespace=rag_config.vector_namespace,
            vector_collection_prefix=rag_config.vector_collection_prefix,
        )
        requirements = CapabilityRequirements(
            require_chat=True,
            default_context_tokens=QueryOptions().max_context_tokens,
        )
        runtime = RAGRuntime.from_request(
            storage=storage,
            request=AssemblyRequest(
                requirements=requirements,
                overrides=assembly_overrides,
            ),
            generation_config=runtime_config.generation,
            chat_context_window_tokens=(
                runtime_config.primary_model.context_window_tokens or 32_768
            ),
            llm_stage_budgets=runtime_config.llm_stage_budgets,
        )
        return runtime, ()
    except Exception as exc:
        logger.warning("RAG auto-attach failed; continuing without RAG", exc_info=True)
        return None, (
            RuntimeDiagnostic.from_exception(
                code="rag_auto_attach_failed",
                component="rag_runtime",
                error=exc,
            ),
        )

from __future__ import annotations

import logging
import os
import time
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from agent_runtime.models import ModelControlPlane

if TYPE_CHECKING:
    from rag.agent.core.definition import AgentRuntimePolicy
    from rag.agent.core.registry import AgentRegistry
    from rag.agent.core.runtime_diagnostics import RuntimeDiagnostic
    from rag.agent.service import AgentService
    from rag.agent.tools.registry import ContextualToolRunner

logger = logging.getLogger(__name__)

DEFAULT_VECTOR_BACKEND = "milvus"
CLI_AGENT_CHOICES = ("generic",)
_SEMANTIC_RAG_TOOLS = frozenset({"search_knowledge", "search_assets"})


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


def build_llm_tool_runners(
    primary_chat: Any,
    *,
    token_accounting: object | None = None,
    model_context_tokens: int = 32_768,
    stage_budgets: object | None = None,
) -> dict[str, ContextualToolRunner]:
    from rag.agent.core.llm_registry import ResolvedModel
    from rag.agent.core.llm_tool_runners import create_model_llm_tool_runners
    from rag.assembly.tokenizer import TokenAccountingService, TokenizerContract
    from rag.providers.llm_gateway import LLMGateway
    from rag.schema.llm import DEFAULT_LLM_STAGE_BUDGETS

    if primary_chat is None:
        return {}

    accounting = token_accounting or TokenAccountingService(
        TokenizerContract(
            embedding_model_name="cli-chat",
            tokenizer_model_name="cli-chat",
            chunking_tokenizer_model_name="cli-chat",
            tokenizer_backend="simple",
            max_context_tokens=model_context_tokens,
            prompt_reserved_tokens=512,
            local_files_only=True,
        )
    )
    gateway = LLMGateway(
        generator=primary_chat,
        token_accounting=cast(Any, accounting),
        model_context_tokens=model_context_tokens,
        stage_budgets=cast(
            Any,
            stage_budgets or DEFAULT_LLM_STAGE_BUDGETS,
        ),
    )

    class _Registry:
        def resolve_for_node(
            self,
            *,
            node_model: str | None,
            node_name: str,
        ) -> ResolvedModel:
            del node_model, node_name
            return ResolvedModel(
                generator=primary_chat,
                kwargs={},
                context_window_tokens=model_context_tokens,
                gateway=gateway,
                token_accounting=cast(Any, accounting),
            )

    return create_model_llm_tool_runners(cast(Any, _Registry()))


def resolve_cli_agent_definition(
    agent_registry: AgentRegistry,
    agent_type: str,
) -> AgentRuntimePolicy:
    if agent_type not in CLI_AGENT_CHOICES:
        allowed = ", ".join(CLI_AGENT_CHOICES)
        raise ValueError(f"{agent_type!r} is not a supported CLI agent. Allowed: {allowed}")
    return agent_registry.get(agent_type)


def _without_unavailable_deferred_tools(
    definition: AgentRuntimePolicy,
    unavailable_tools: set[str],
) -> AgentRuntimePolicy:
    if not unavailable_tools:
        return definition
    filt = definition.tool_catalog_filter
    return replace(
        definition,
        deferred_tool_names=tuple(
            name for name in definition.deferred_tool_names
            if name not in unavailable_tools
        ),
        tool_catalog_filter=replace(filt, deny=filt.deny | frozenset(unavailable_tools)),
    )


def build_agent_service(
    runtime: Any | None,
    *,
    checkpoint_db: Path | None = None,
    agent_type: str = "generic",
    model_alias: str | None = None,
    model_control_plane: ModelControlPlane | None = None,
    runtime_diagnostics: Sequence[RuntimeDiagnostic] = (),
    knowledge_runner: ContextualToolRunner | None = None,
    knowledge_asset_runner: ContextualToolRunner | None = None,
    strict_model_provider: bool = True,
    startup_ms: float = 0.0,
) -> AgentService:
    """Build the product Agent service.

    RAG is an optional attached tool provider. Pure Agent runs should not
    require RAG storage/vector configuration, and unavailable RAG tools should
    not be visible to the model.
    """
    from rag.agent.builtin import create_builtin_agent_registry
    from rag.agent.builtin_registry import create_builtin_tool_registry
    from rag.agent.core.agent_service_factory import AgentServiceFactory
    from rag.agent.core.checkpointing import create_agent_checkpointer
    from rag.agent.core.runtime_diagnostics import AgentLatencyProfile, RuntimeDiagnostic
    from rag.agent.core.subagent_runner import BuiltinSubAgentRunner
    from rag.agent.tools.registry import ToolRunner

    build_started_at = time.perf_counter()
    model_ready_ms = 0.0
    agent_registry = create_builtin_agent_registry()
    definition = resolve_cli_agent_definition(agent_registry, agent_type)

    runners: dict[str, ToolRunner] = {}
    contextual_runners: dict[str, ContextualToolRunner] = {}

    if runtime is not None:
        bundle = getattr(runtime, "capability_bundle", None)
        chat_bindings = list(getattr(bundle, "chat_bindings", ()) or ())
        primary_chat = chat_bindings[0] if chat_bindings else None

        retrieval_service = getattr(runtime, "retrieval_service", None)
        if retrieval_service is not None:
            from rag.agent.tools.rag_answer_tools import RAGSearchAnswerRunner
            from rag.agent.tools.rag_tool_runner import AsyncRAGToolRunner

            rag_runner = AsyncRAGToolRunner(
                runtime=runtime,
                retrieval_service=retrieval_service,
                max_context_tokens=4096,
            )
            for name in ("vector_search", "keyword_search", "grounding", "rerank", "graph_expand"):
                contextual_runners[name] = cast(
                    Any,
                    rag_runner.retrieve_evidence,
                )
            rag_answer_runner = RAGSearchAnswerRunner(runtime=runtime)
            contextual_runners["rag_search_answer"] = cast(
                Any,
                rag_answer_runner.answer,
            )
            contextual_runners["search_knowledge"] = cast(
                Any,
                rag_answer_runner.answer,
            )

        from rag.agent.tools.asset_tools import AssetToolRunner

        stores = getattr(runtime, "stores", None)
        metadata_repo = getattr(stores, "metadata_repo", None)
        object_store = getattr(stores, "object_store", None)
        if metadata_repo is not None and object_store is not None:
            from rag.agent.tools.asset_tools import AssetInspectInput, AssetListInput

            asset_runner = AssetToolRunner(
                metadata_repo=metadata_repo,
                object_store=object_store,
            )
            runners["asset_list"] = cast(ToolRunner, asset_runner.list_assets)
            runners["asset_inspect"] = cast(ToolRunner, asset_runner.inspect_asset)
            runners["asset_read_slice"] = cast(ToolRunner, asset_runner.read_slice)
            runners["asset_analyze"] = cast(ToolRunner, asset_runner.analyze_asset)

            async def _search_assets_runner(payload: Any) -> Any:
                from rag.agent.tools.rag_semantic_tools import (
                    AssetResult,
                    AssetSearchInput,
                    AssetSearchOutput,
                )

                if isinstance(payload, dict):
                    inp = AssetSearchInput(**payload)
                else:
                    inp = payload
                list_out = asset_runner.list_assets(
                    AssetListInput(
                        doc_id=inp.doc_id,
                        asset_type=inp.asset_type,
                        limit=inp.max_results,
                    )
                )
                results: list[AssetResult] = []
                for a in list_out.assets[:inp.max_results]:
                    ar = AssetResult(
                        asset_id=a.asset_id,
                        doc_id=a.doc_id,
                        asset_type=a.asset_type,
                        sheet_name=a.sheet_name,
                        caption=a.caption,
                        columns=list(a.columns or []),
                        row_count=a.row_count,
                        column_count=a.column_count,
                    )
                    if inp.include_preview and a.asset_id:
                        try:
                            insp = asset_runner.inspect_asset(
                                AssetInspectInput(asset_id=a.asset_id, head_rows=3)
                            )
                            if insp.head_rows:
                                ar.preview_rows = insp.head_rows[:3]
                            ar.analysis_capabilities = list(insp.analysis_capabilities or [])
                        except Exception:
                            pass
                    results.append(ar)
                return AssetSearchOutput(assets=results, total_found=len(list_out.assets))

            contextual_runners["search_assets"] = cast(Any, _search_assets_runner)

        contextual_runners.update(
            build_llm_tool_runners(
                primary_chat,
                token_accounting=getattr(runtime, "token_accounting", None),
                model_context_tokens=getattr(
                    runtime,
                    "chat_context_window_tokens",
                    32_768,
                ),
                stage_budgets=getattr(
                    runtime,
                    "llm_stage_budgets",
                    None,
                ),
            )
        )

    if knowledge_runner is not None:
        contextual_runners["search_knowledge"] = knowledge_runner
    if knowledge_asset_runner is not None:
        contextual_runners["search_assets"] = knowledge_asset_runner

    unavailable_rag_tools = {
        name for name in _SEMANTIC_RAG_TOOLS
        if name not in contextual_runners
    }
    definition = _without_unavailable_deferred_tools(
        definition,
        unavailable_rag_tools,
    )

    tool_registry = create_builtin_tool_registry(
        runners=runners,
        contextual_runners=contextual_runners,
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

    skill_catalog = None
    try:
        from rag.agent.skills.catalog import SkillCatalog
        from rag.agent.skills.loader import scan_and_load_skills

        manifests = scan_and_load_skills(Path.cwd())
        if manifests:
            skill_catalog = SkillCatalog(manifests)
    except Exception as exc:
        logger.warning(
            "Skill catalog loading failed; continuing without skills",
            exc_info=True,
        )
        diagnostics = (
            *diagnostics,
            RuntimeDiagnostic.from_exception(
                code="skill_catalog_load_failed",
                component="skill_catalog",
                error=exc,
            ),
        )

    service_factory = AgentServiceFactory(
        tool_registry=tool_registry,
        model_registry=model_registry,
        checkpointer=create_agent_checkpointer(checkpoint_db),
        runtime_diagnostics=diagnostics,
        skill_catalog=skill_catalog,
        strict_model_provider=strict_model_provider,
        latency_profile=AgentLatencyProfile(
            startup_ms=startup_ms,
            build_service_ms=(time.perf_counter() - build_started_at) * 1000,
            model_ready_ms=model_ready_ms,
        ),
    )
    subagent_runner = BuiltinSubAgentRunner(
        agent_registry=agent_registry,
        service_factory=service_factory,
    )
    service_factory.bind_subagent_runner(subagent_runner)
    return service_factory.create(definition)


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

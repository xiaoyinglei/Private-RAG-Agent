from __future__ import annotations

import asyncio
import logging
import time
import warnings
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid4

from agent_runtime.models import ModelControlPlane, ModelSpec
from agent_runtime.result import AgentResult

if TYPE_CHECKING:
    from agent_runtime.knowledge_providers.rag import LazyRAGKnowledgeProvider
    from rag.agent.core.runtime_diagnostics import RuntimeDiagnostic
    from rag.agent.streaming.sink import StreamEventSink
    from rag.agent.tools.tool import Tool

DEFAULT_VECTOR_BACKEND = "milvus"
_RUNTIME_CLOSE_GRACE_SECONDS = 5.0
logger = logging.getLogger(__name__)


class Agent:
    def __init__(
        self,
        *,
        model: str | None = None,
        agent_type: str = "generic",
        checkpoint_db: Path | None = None,
        workspace_path: Path | str | None = None,
        model_session_path: Path | None = None,
        knowledge: tuple[str, ...] | list[str] | None = None,
        rag_storage_root: Path = Path(".rag"),
        embedding_model: str | None = None,
        reranker_model: str | None = None,
        vector_backend: str = DEFAULT_VECTOR_BACKEND,
        vector_dsn: str | None = None,
        vector_namespace: str | None = None,
        vector_collection_prefix: str | None = None,
    ) -> None:
        self.model = model
        self.agent_type = agent_type
        self.checkpoint_db = checkpoint_db
        self.workspace_path = (
            None
            if workspace_path is None
            else Path(workspace_path).expanduser().resolve()
        )
        self.model_session_path = model_session_path
        self.knowledge = tuple(knowledge or ())
        self.rag_storage_root = rag_storage_root
        self.embedding_model = embedding_model
        self.reranker_model = reranker_model
        self.vector_backend = vector_backend
        self.vector_dsn = vector_dsn
        self.vector_namespace = vector_namespace
        self.vector_collection_prefix = vector_collection_prefix
        self._model_control_plane: ModelControlPlane | None = None
        self._session_store: Any | None = None
        self._checkpointer: Any | None = None

    def models(self) -> list[ModelSpec]:
        return self._get_model_control_plane().list_models()

    def current_model(self) -> ModelSpec:
        return self._get_model_control_plane().current_model()

    def switch_model(self, model_id: str) -> ModelSpec:
        return self._get_model_control_plane().switch_model(
            model_id,
            requested_by="user",
            persist=self.model_session_path is not None,
        )

    def run(
        self,
        task: str,
        *,
        files: list[str] | tuple[str, ...] | None = None,
        run_id: str | None = None,
        max_turns: int | None = None,
        max_tokens_total: int | None = None,
        tools: list[str] | tuple[str, ...] | None = None,
        disabled_tools: list[str] | tuple[str, ...] | None = None,
        allow_write_tools: bool = False,
        allow_execute_tools: bool = False,
        allow_discovery_tools: bool | None = None,
    ) -> AgentResult:
        return asyncio.run(
            self.arun(
                task,
                files=files,
                run_id=run_id,
                max_turns=max_turns,
                max_tokens_total=max_tokens_total,
                tools=tools,
                disabled_tools=disabled_tools,
                allow_write_tools=allow_write_tools,
                allow_execute_tools=allow_execute_tools,
                allow_discovery_tools=allow_discovery_tools,
            )
        )

    async def arun(
        self,
        task: str,
        *,
        files: list[str] | tuple[str, ...] | None = None,
        run_id: str | None = None,
        max_turns: int | None = None,
        max_tokens_total: int | None = None,
        tools: list[str] | tuple[str, ...] | None = None,
        disabled_tools: list[str] | tuple[str, ...] | None = None,
        allow_write_tools: bool = False,
        allow_execute_tools: bool = False,
        allow_discovery_tools: bool | None = None,
    ) -> AgentResult:
        from rag.agent.service import AgentRunRequest

        _warn_deprecated_tool_options(
            tools=tools,
            disabled_tools=disabled_tools,
            allow_discovery_tools=allow_discovery_tools,
        )
        effective_discovery = _effective_discovery_option(
            tools=tools,
            disabled_tools=disabled_tools,
            allow_discovery_tools=allow_discovery_tools,
        )
        async with self._open_product_runtime() as service:
            turn_id = _public_turn_id(run_id)
            raw = await service.chat(
                AgentRunRequest(
                    task=task,
                    session_id=None,
                    run_id=turn_id,
                    thread_id=turn_id,
                    max_turns=max_turns,
                    llm_budget_total=max_tokens_total,
                    input_files=list(files or ()),
                    workspace_path=(
                        None
                        if self.workspace_path is None
                        else str(self.workspace_path)
                    ),
                    tools=None if tools is None else tuple(tools),
                    disabled_tools=tuple(disabled_tools or ()),
                    allow_write_tools=allow_write_tools,
                    allow_execute_tools=allow_execute_tools,
                    allow_discovery_tools=effective_discovery,
                )
            )
            return AgentResult.from_internal(
                raw,
                files=tuple(files or ()),
            )

    def chat(
        self,
        message: str,
        *,
        session_id: str | None = None,
        files: list[str] | tuple[str, ...] | None = None,
        max_turns: int | None = None,
        max_tokens_total: int | None = None,
        allow_write_tools: bool = False,
        allow_execute_tools: bool = False,
    ) -> AgentResult:
        return asyncio.run(
            self.achat(
                message,
                session_id=session_id,
                files=files,
                max_turns=max_turns,
                max_tokens_total=max_tokens_total,
                allow_write_tools=allow_write_tools,
                allow_execute_tools=allow_execute_tools,
            )
        )

    async def achat(
        self,
        message: str,
        *,
        session_id: str | None = None,
        files: list[str] | tuple[str, ...] | None = None,
        max_turns: int | None = None,
        max_tokens_total: int | None = None,
        allow_write_tools: bool = False,
        allow_execute_tools: bool = False,
    ) -> AgentResult:
        from rag.agent.service import AgentRunRequest

        turn_id = str(uuid4())
        runtime_agent = (
            self
            if session_id is None
            else self._agent_for_session(session_id)
        )
        async with runtime_agent._open_product_runtime() as service:
            raw = await service.chat(
                AgentRunRequest(
                    task=message,
                    session_id=session_id,
                    run_id=turn_id,
                    thread_id=turn_id,
                    max_turns=max_turns,
                    llm_budget_total=max_tokens_total,
                    input_files=list(files or ()),
                    workspace_path=(
                        None
                        if runtime_agent.workspace_path is None
                        else str(runtime_agent.workspace_path)
                    ),
                    allow_write_tools=allow_write_tools,
                    allow_execute_tools=allow_execute_tools,
                )
            )
            return AgentResult.from_internal(
                raw,
                files=tuple(files or ()),
            )

    def resume(
        self,
        turn_id: str,
        action: str,
        *,
        user_input: str | None = None,
    ) -> AgentResult:
        return asyncio.run(
            self.aresume(
                turn_id,
                action,
                user_input=user_input,
            )
        )

    async def aresume(
        self,
        turn_id: str,
        action: str,
        *,
        user_input: str | None = None,
    ) -> AgentResult:
        runtime_agent = self._agent_for_turn(turn_id)
        async with runtime_agent._open_product_runtime() as service:
            raw = await service.resume_turn(
                turn_id=turn_id,
                action=action,
                user_input=user_input,
            )
            return AgentResult.from_internal(raw)

    def _agent_for_session(self, session_id: str) -> Agent:
        """Load durable metadata before assembling any product runtime."""

        from rag.agent.sessions import SessionBusyError

        session = self._get_session_store().get_session(session_id)
        if session.active_turn_id is not None:
            raise SessionBusyError(
                f"Session {session_id} already has active Turn "
                f"{session.active_turn_id}"
            )
        return self._agent_for_binding(session.runtime)

    def _agent_for_turn(self, turn_id: str) -> Agent:
        from rag.agent.sessions import TurnStateError, TurnStatus

        turn = self._get_session_store().get_turn(turn_id)
        if turn.status in {TurnStatus.COMPLETED, TurnStatus.FAILED}:
            raise TurnStateError(
                f"Turn {turn_id} is {turn.status.value} and cannot resume"
            )
        if (
            turn.status is TurnStatus.RUNNING
            and turn.lease_owner is not None
            and turn.lease_expires_at is not None
            and turn.lease_expires_at > time.time()
        ):
            raise TurnStateError(
                f"Turn {turn_id} is still running under an active lease"
            )
        return self._agent_for_binding(turn.runtime)

    def _agent_for_binding(self, binding: Any) -> Agent:
        restored = Agent(
            model=binding.model_alias,
            agent_type=binding.agent_type,
            checkpoint_db=self.checkpoint_db,
            workspace_path=binding.workspace_path,
            knowledge=binding.knowledge,
            rag_storage_root=Path(binding.rag_storage_root),
            embedding_model=binding.embedding_model_alias,
            reranker_model=binding.reranker_model_alias,
            vector_backend=binding.vector_backend,
            vector_dsn=self.vector_dsn,
            vector_namespace=binding.vector_namespace,
            vector_collection_prefix=binding.vector_collection_prefix,
        )
        restored._session_store = self._get_session_store()
        restored._checkpointer = self._get_checkpointer()
        return restored

    async def stream(
        self,
        task: str,
        *,
        session_id: str | None = None,
        files: list[str] | tuple[str, ...] | None = None,
        run_id: str | None = None,
        max_turns: int | None = None,
        max_tokens_total: int | None = None,
        tools: list[str] | tuple[str, ...] | None = None,
        disabled_tools: list[str] | tuple[str, ...] | None = None,
        allow_write_tools: bool = False,
        allow_execute_tools: bool = False,
        allow_discovery_tools: bool | None = None,
    ) -> AsyncIterator[Any]:
        from rag.agent.service import AgentRunRequest

        _warn_deprecated_tool_options(
            tools=tools,
            disabled_tools=disabled_tools,
            allow_discovery_tools=allow_discovery_tools,
        )
        effective_discovery = _effective_discovery_option(
            tools=tools,
            disabled_tools=disabled_tools,
            allow_discovery_tools=allow_discovery_tools,
        )
        runtime_agent = (
            self
            if session_id is None
            else self._agent_for_session(session_id)
        )
        async with runtime_agent._open_product_runtime() as service:
            effective_run_id = _public_turn_id(run_id)
            request = AgentRunRequest(
                task=task,
                session_id=session_id,
                run_id=effective_run_id,
                thread_id=effective_run_id,
                max_turns=max_turns,
                llm_budget_total=max_tokens_total,
                input_files=list(files or ()),
                workspace_path=(
                    None
                    if runtime_agent.workspace_path is None
                    else str(runtime_agent.workspace_path)
                ),
                tools=None if tools is None else tuple(tools),
                disabled_tools=tuple(disabled_tools or ()),
                allow_write_tools=allow_write_tools,
                allow_execute_tools=allow_execute_tools,
                allow_discovery_tools=effective_discovery,
            )
            async for event in service.chat_streaming(request):
                yield event

    @asynccontextmanager
    async def _open_product_runtime(
        self,
        *,
        stream_sink: StreamEventSink | None = None,
    ) -> AsyncIterator[Any]:
        """Own one SDK call's resources and release them in reverse order."""

        from agent_runtime.runtime.mcp import (
            open_product_mcp_tools,
            resolve_product_mcp_config,
        )

        config_path = resolve_product_mcp_config(self.workspace_path)
        runtime_diagnostics: list[RuntimeDiagnostic] = []
        async with open_product_mcp_tools(
            config_path,
            diagnostics=runtime_diagnostics,
        ) as mcp_tools:
            service, provider = self._build_service(
                mcp_tools=mcp_tools,
                runtime_diagnostics=tuple(runtime_diagnostics),
                stream_sink=stream_sink,
            )
            try:
                yield service
            finally:
                try:
                    close_method = getattr(service, "aclose", None)
                    if callable(close_method):
                        try:
                            await asyncio.wait_for(
                                close_method(),
                                timeout=_RUNTIME_CLOSE_GRACE_SECONDS,
                            )
                        except TimeoutError:
                            logger.warning(
                                "Agent runtime close exceeded %.1fs grace period",
                                _RUNTIME_CLOSE_GRACE_SECONDS,
                            )
                finally:
                    if provider is not None:
                        await _close_owned_sync_resource(
                            provider,
                            label="knowledge provider",
                        )

    def _build_service(
        self,
        *,
        mcp_tools: tuple[Tool, ...] = (),
        runtime_diagnostics: Sequence[RuntimeDiagnostic] = (),
        stream_sink: StreamEventSink | None = None,
    ) -> tuple[Any, LazyRAGKnowledgeProvider | None]:
        from agent_runtime.runtime.builder import build_agent_service
        from rag.agent.skills.catalog import SkillCatalog
        from rag.agent.skills.loader import scan_and_load_skills
        from rag.agent.skills.policy import SkillPolicy
        from rag.agent.skills.runtime import SkillRuntime
        from rag.agent.tools.integrations.skills import create_skill_tools
        from rag.agent.tools.integrations.subagent import (
            SubagentInput,
            create_subagent_tool,
        )
        from rag.agent.workspace import open_workspace
        from rag.utils.text import load_env_file

        startup_started_at = time.perf_counter()
        load_env_file()
        try:
            model_control_plane = self._get_model_control_plane()
        except Exception:
            if self.model is not None:
                raise
            model_control_plane = None
        provider: LazyRAGKnowledgeProvider | None = None
        knowledge_runner = None
        if self.knowledge:
            from agent_runtime.knowledge_providers.rag import LazyRAGKnowledgeProvider

            provider = LazyRAGKnowledgeProvider(
                storage_root=self.rag_storage_root,
                model_alias=self.model,
                embedding_model_alias=self.embedding_model,
                reranker_model_alias=self.reranker_model,
                vector_backend=self.vector_backend,
                vector_dsn=self.vector_dsn,
                vector_namespace=self.vector_namespace,
                vector_collection_prefix=self.vector_collection_prefix,
            )
            knowledge_runner = provider.search_knowledge

        workspace = (
            None
            if self.workspace_path is None
            else open_workspace(self.workspace_path, create=True)
        )
        skill_runtime = None
        skill_tools: tuple[Tool, ...] = ()
        subagent_tools: tuple[Tool, ...] = ()
        if workspace is not None:
            policy = SkillPolicy()
            manifests = [
                manifest
                for manifest in scan_and_load_skills(
                    workspace.root,
                    repo_root=workspace.root,
                )
                if policy.is_skill_enabled(manifest)
            ]
            catalog = SkillCatalog(manifests)
            candidate_runtime = SkillRuntime(catalog, policy=policy)
            if candidate_runtime.has_model_invocable_skills:
                skill_runtime = candidate_runtime
                skill_tools = create_skill_tools(
                    workspace,
                    invoke_skill=skill_runtime.invoke_skill,
                    active_skill_root=skill_runtime.skill_root,
                    invoke_execution_revision=skill_runtime.catalog_revision,
                )

            async def run_subagent(arguments: object) -> dict[str, object]:
                from rag.agent.service import AgentRunRequest

                payload = SubagentInput.model_validate(arguments)
                child_task = payload.task
                if payload.context_summary:
                    child_task += (
                        "\n\nContext supplied by the parent agent:\n"
                        + payload.context_summary
                    )
                child_service = build_agent_service(
                    workspace,
                    checkpoint_db=None,
                    agent_type=self.agent_type,
                    model_alias=self.model,
                    model_control_plane=model_control_plane,
                    runtime_diagnostics=(),
                    knowledge_runner=knowledge_runner,
                    mcp_tools=mcp_tools,
                    skill_tools=skill_tools,
                    skill_runtime=skill_runtime,
                )
                try:
                    child = await child_service.run(
                        AgentRunRequest(
                            task=child_task,
                            max_turns=payload.max_turns,
                            llm_budget_total=payload.llm_budget_total,
                            max_depth=0,
                            workspace_path=str(workspace.root),
                        )
                    )
                finally:
                    close_child = getattr(child_service, "aclose", None)
                    if callable(close_child):
                        try:
                            await asyncio.wait_for(
                                close_child(),
                                timeout=_RUNTIME_CLOSE_GRACE_SECONDS,
                            )
                        except TimeoutError:
                            logger.warning(
                                "Subagent close exceeded %.1fs grace period",
                                _RUNTIME_CLOSE_GRACE_SECONDS,
                            )
                status = (
                    child.status
                    if child.status in {"done", "failed", "paused"}
                    else "failed"
                )
                return {
                    "conclusion": child.final_answer or child.needs_user_input or "",
                    "key_facts": [
                        item.text[:2000] for item in child.evidence[:10]
                    ],
                    "evidence_refs": [
                        {
                            "evidence_id": item.evidence_id,
                            "doc_id": item.doc_id,
                            "citation_anchor": item.citation_anchor,
                        }
                        for item in child.evidence[:20]
                    ],
                    "citations": [
                        {
                            **item.model_dump(mode="json"),
                            "citation_anchor": item.citation_anchor or "",
                        }
                        for item in child.citations[:20]
                    ],
                    "status": status,
                    "child_run_id": child.run_id,
                    "stop_reason": child.stop_reason,
                }

            subagent_tools = (create_subagent_tool(run_subagent),)
        service = build_agent_service(
            workspace,
            checkpoint_db=self.checkpoint_db,
            checkpointer=self._get_checkpointer(),
            agent_type=self.agent_type,
            model_alias=self.model,
            model_control_plane=model_control_plane,
            runtime_diagnostics=runtime_diagnostics,
            knowledge_runner=knowledge_runner,
            mcp_tools=mcp_tools,
            skill_tools=skill_tools,
            subagent_tools=subagent_tools,
            skill_runtime=skill_runtime,
            stream_sink=stream_sink,
            startup_ms=(time.perf_counter() - startup_started_at) * 1000,
            session_store=self._get_session_store(),
            runtime_binding=self._runtime_binding(),
        )
        return service, provider

    def _get_session_store(self) -> Any:
        if self._session_store is None:
            from rag.agent.sessions import SessionStore

            self._session_store = SessionStore(self.checkpoint_db)
        return self._session_store

    def _get_checkpointer(self) -> Any:
        if self._checkpointer is None:
            from rag.agent.core.checkpointing import create_agent_checkpointer

            self._checkpointer = create_agent_checkpointer(self.checkpoint_db)
        return self._checkpointer

    def _runtime_binding(self) -> Any:
        from rag.agent.sessions import RuntimeBinding

        model_alias = self.model
        if self._model_control_plane is not None:
            model_alias = self._model_control_plane.current_model().id
        return RuntimeBinding(
            agent_type=self.agent_type,
            model_alias=model_alias,
            workspace_path=(
                None
                if self.workspace_path is None
                else str(self.workspace_path)
            ),
            knowledge=self.knowledge,
            rag_storage_root=str(self.rag_storage_root),
            embedding_model_alias=self.embedding_model,
            reranker_model_alias=self.reranker_model,
            vector_backend=self.vector_backend,
            vector_namespace=self.vector_namespace,
            vector_collection_prefix=self.vector_collection_prefix,
        )

    def _get_model_control_plane(self) -> ModelControlPlane:
        if self._model_control_plane is None:
            self._model_control_plane = ModelControlPlane.from_env(
                initial_model_id=self.model,
                session_path=self.model_session_path,
            )
        return self._model_control_plane


def _warn_deprecated_tool_options(
    *,
    tools: list[str] | tuple[str, ...] | None,
    disabled_tools: list[str] | tuple[str, ...] | None,
    allow_discovery_tools: bool | None,
) -> None:
    if (
        tools is None
        and not disabled_tools
        and allow_discovery_tools is None
    ):
        return
    warnings.warn(
        "explicit tool selection options are deprecated; product capability "
        "assembly selects tools automatically",
        DeprecationWarning,
        stacklevel=3,
    )


def _effective_discovery_option(
    *,
    tools: list[str] | tuple[str, ...] | None,
    disabled_tools: list[str] | tuple[str, ...] | None,
    allow_discovery_tools: bool | None,
) -> bool | None:
    if allow_discovery_tools is not None:
        return allow_discovery_tools
    if tools is not None or disabled_tools is not None:
        return False
    return None


def _public_turn_id(run_id: str | None) -> str:
    if run_id is None:
        return str(uuid4())
    warnings.warn(
        "run_id is deprecated; it is interpreted as the UUID Turn ID",
        DeprecationWarning,
        stacklevel=3,
    )
    try:
        return str(UUID(run_id))
    except (TypeError, ValueError) as exc:
        raise ValueError("run_id must be a UUID Turn ID") from exc


async def _close_owned_sync_resource(
    resource: object,
    *,
    label: str,
) -> None:
    close_method = getattr(resource, "close", None)
    if not callable(close_method):
        return
    try:
        await asyncio.wait_for(
            asyncio.to_thread(close_method),
            timeout=_RUNTIME_CLOSE_GRACE_SECONDS,
        )
    except TimeoutError:
        logger.warning(
            "%s close exceeded %.1fs grace period",
            label,
            _RUNTIME_CLOSE_GRACE_SECONDS,
        )
    except Exception:
        logger.warning("%s close failed", label, exc_info=True)

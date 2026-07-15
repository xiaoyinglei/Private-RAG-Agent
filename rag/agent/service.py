from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncGenerator, Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

from langchain_core.messages import BaseMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from pydantic import BaseModel, ConfigDict, Field, field_validator

from rag.agent.core.checkpointing import (
    LangGraphCheckpointStore,
    aclose_agent_checkpointer,
    create_agent_checkpointer,
    reconcile_tool_manifest,
)
from rag.agent.core.context import AgentRunConfig, RunRegistry
from rag.agent.core.definition import AgentRuntimePolicy
from rag.agent.core.finalization import FinishCandidateBuilder
from rag.agent.core.goal_contract import GoalCompatibilityConfig, GoalSpec
from rag.agent.core.human_input import (
    HumanInputRequest,
    HumanInputResponse,
)
from rag.agent.core.llm_registry import ModelResolver
from rag.agent.core.messages import ModelMessage, snapshot_model_message
from rag.agent.core.model_provider_runtime import ModelProviderResolver
from rag.agent.core.model_request import (
    ModelCallRecord,
    build_stable_context,
    build_tool_manifest,
)
from rag.agent.core.output_finalizer import (
    StructuredOutputFinalizer,
    create_model_structured_output_finalizer,
)
from rag.agent.core.output_models import ValidatedFinalOutput, output_model_path
from rag.agent.core.runtime_diagnostics import (
    AgentLatencyProfile,
    RuntimeDiagnostic,
    merge_runtime_diagnostics,
)
from rag.agent.core.turn_contracts import (
    ToolCallPlan,
    ToolManifestDriftStatus,
)
from rag.agent.file_manifest import FileManifest, build_file_manifest
from rag.agent.loop.runtime import AgentLoop, ModelTurnProvider
from rag.agent.loop.state import (
    LoopPause,
    LoopState,
    LoopTerminal,
    LoopTransition,
    create_loop_state,
)
from rag.agent.loop.stop_hooks import StopHookRunner, build_stop_hooks
from rag.agent.memory.compactor import LoopContextCompactor, MessageCompactor
from rag.agent.memory.models import MemoryPolicy
from rag.agent.memory.store import WorkspaceMemoryStore
from rag.agent.planning import AgentPlan, PlanEvent
from rag.agent.sessions import (
    RuntimeBinding,
    SessionRecord,
    SessionStore,
    TurnStateError,
    TurnStatus,
)
from rag.agent.skills.catalog import SkillCatalog
from rag.agent.skills.runtime import SkillRuntime
from rag.agent.streaming.events import StreamEvent
from rag.agent.streaming.sink import StreamEventSink
from rag.agent.tools.builtins import RESIDENT_CODING_TOOL_NAMES
from rag.agent.tools.executor import ToolExecutor
from rag.agent.tools.permissions import ToolExecutionContext
from rag.agent.tools.registry import ToolRegistry
from rag.agent.tools.selection import resolve_tool_options, select_tools
from rag.agent.tools.tool import JsonValue, ToolCall, ToolCallOrigin, ToolResult
from rag.agent.workspace import (
    WorkspaceRuntime,
    create_temp_workspace,
    import_files,
    open_workspace,
)
from rag.schema.query import AnswerCitation, EvidenceItem
from rag.schema.runtime import AccessPolicy

logger = logging.getLogger(__name__)
_TURN_LEASE_SECONDS = 300.0
_TURN_LEASE_HEARTBEAT_SECONDS = 60.0


class AgentRunRequest(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    task: str = Field(min_length=1)
    session_id: str | None = None
    run_id: str | None = None
    thread_id: str | None = None
    max_turns: int | None = None
    max_context_tokens: int | None = Field(default=None, gt=0)
    llm_budget_total: int | None = Field(default=None, gt=0)
    max_depth: int | None = Field(default=None, ge=0)
    pending_tool_calls: list[ToolCallPlan] = Field(default_factory=list)
    approved_tool_call_ids: list[str] = Field(default_factory=list)
    denied_tool_call_ids: list[str] = Field(default_factory=list)
    messages: list[BaseMessage] = Field(default_factory=list)
    history_messages: list[ModelMessage] = Field(default_factory=list)
    turn_messages: list[ModelMessage] = Field(default_factory=list)
    input_files: list[str] = Field(default_factory=list)
    workspace_path: str | None = None
    memory_policy: MemoryPolicy | None = None
    goal_spec: GoalSpec | None = None
    tools: tuple[str, ...] | None = None
    disabled_tools: tuple[str, ...] = ()
    allow_write_tools: bool = False
    allow_execute_tools: bool = False
    allow_discovery_tools: bool | None = None

    @field_validator("tools", "disabled_tools", mode="before")
    @classmethod
    def _tuple_names(cls, value: object) -> object:
        if value is None:
            return None
        if isinstance(value, (str, bytes)):
            raise TypeError("tool options must be sequences of names")
        return tuple(value)  # type: ignore[arg-type]

    def to_run_config(self, definition: AgentRuntimePolicy) -> AgentRunConfig:
        run_id = self.run_id or str(uuid4())
        return AgentRunConfig(
            run_id=run_id,
            thread_id=run_id,
            session_id=self.session_id,
            max_turns=self.max_turns,
            agent_type=definition.agent_type,
            max_context_tokens=self.max_context_tokens,
            llm_budget_total=self.llm_budget_total,
            max_depth=(
                definition.max_depth
                if self.max_depth is None
                else self.max_depth
            ),
            access_policy=(
                definition.access_policy_ceiling or AccessPolicy.default()
            ),
            tool_policy=definition.tool_policy,
            memory_policy=self.memory_policy or MemoryPolicy(),
        )


class AgentRunResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    run_id: str
    thread_id: str
    session_id: str = ""
    status: str
    final_answer: str | None = None
    final_output: BaseModel | None = None
    output_validation_errors: list[dict[str, object]] = Field(
        default_factory=list
    )
    stop_reason: str | None = None
    tool_results: list[ToolResult] = Field(default_factory=list)
    model_call_records: list[ModelCallRecord] = Field(default_factory=list)
    evidence: list[EvidenceItem] = Field(default_factory=list)
    citations: list[AnswerCitation] = Field(default_factory=list)
    iteration: int = 0
    groundedness_flag: bool = False
    insufficient_evidence_flag: bool = False
    needs_user_input: str | None = None
    human_input_request: object | None = None
    pending_tool_calls_summary: list[dict[str, object]] = Field(
        default_factory=list
    )
    workspace_path: str | None = None
    runtime_diagnostics: list[RuntimeDiagnostic] = Field(default_factory=list)
    tool_call_metrics: object | None = None
    latency_profile: AgentLatencyProfile | None = None
    plan: AgentPlan | None = None
    plan_events: list[PlanEvent] = Field(default_factory=list)

    @classmethod
    def from_state(
        cls,
        state: LoopState,
        *,
        definition: AgentRuntimePolicy | None = None,
        workspace_path: str | None = None,
    ) -> AgentRunResult:
        return cls.from_loop_result(
            state,
            definition=definition,
            workspace_path=workspace_path,
        )

    @classmethod
    def from_loop_result(
        cls,
        state: LoopState,
        *,
        definition: AgentRuntimePolicy | None = None,
        workspace_path: str | None = None,
    ) -> AgentRunResult:
        run_config = state["run_config"]
        terminal = state["terminal"]
        pause = state["pause"]
        evidence, citations = _result_provenance(state["tool_results"])
        is_terminal = state["status"] in {"completed", "failed"}
        return cls(
            run_id=run_config.run_id,
            thread_id=run_config.thread_id,
            session_id=run_config.session_id or "",
            status=(
                "done" if state["status"] == "completed" else state["status"]
            ),
            final_answer=state["finish_state"].final_answer,
            final_output=_restore_final_output(
                state["finish_state"].final_output,
                definition=definition,
            ),
            output_validation_errors=list(
                state["finish_state"].output_validation_errors
            ),
            stop_reason=None if terminal is None else terminal.stop_reason,
            tool_results=list(state["tool_results"]),
            model_call_records=list(state.get("model_call_records", ())),
            evidence=evidence,
            citations=citations,
            iteration=state["iteration"],
            groundedness_flag=_result_flag(
                state["tool_results"],
                "groundedness_flag",
            ),
            insufficient_evidence_flag=(
                _result_flag(state["tool_results"], "insufficient_evidence")
                or _result_flag(
                    state["tool_results"],
                    "insufficient_evidence_flag",
                )
            ),
            needs_user_input=(
                None if is_terminal or pause is None else pause.reason
            ),
            human_input_request=(
                None if is_terminal else state["approval_request"]
            ),
            pending_tool_calls_summary=[
                {
                    "tool_call_id": call.tool_call_id,
                    "tool_name": call.tool_name,
                }
                for call in state["pending_tool_calls"]
            ],
            workspace_path=workspace_path,
            runtime_diagnostics=list(state["runtime_diagnostics"]),
            tool_call_metrics=state.get("tool_call_metrics"),
            latency_profile=state.get("latency_profile"),
            plan=state["plan_state"].agent_plan,
            plan_events=list(state["plan_state"].plan_events),
        )


class AgentService:
    """Assemble one frozen Tool snapshot and reuse one executor everywhere."""

    def __init__(
        self,
        *,
        definition: AgentRuntimePolicy | None = None,
        policy: AgentRuntimePolicy | None = None,
        tool_registry: ToolRegistry,
        model_turn_provider: ModelTurnProvider | None = None,
        retrieval_hint_provider: object | None = None,
        subagent_runner: object | None = None,
        output_finalizer: StructuredOutputFinalizer | None = None,
        model_registry: ModelResolver | None = None,
        checkpointer: BaseCheckpointSaver[str] | None = None,
        runtime_diagnostics: Sequence[RuntimeDiagnostic] = (),
        catalog: object | None = None,
        stream_sink: StreamEventSink | None = None,
        mcp_registry: object | None = None,
        skill_catalog: SkillCatalog | None = None,
        skill_runtime: SkillRuntime | None = None,
        strict_model_provider: bool = True,
        latency_profile: AgentLatencyProfile | None = None,
        workspace: WorkspaceRuntime | None = None,
        configured_resident_tool_names: Sequence[str] = (),
        discoverable_tool_names: Sequence[str] = (),
        session_store: SessionStore | None = None,
        runtime_binding: RuntimeBinding | None = None,
    ) -> None:
        del (
            retrieval_hint_provider,
            subagent_runner,
            catalog,
            mcp_registry,
        )
        if definition is not None and policy is not None:
            raise ValueError("Provide either 'definition' or 'policy', not both")
        effective_policy = definition or policy
        if effective_policy is None:
            raise ValueError("Provide either 'definition' or 'policy'")
        self._policy: AgentRuntimePolicy = effective_policy
        self._tool_registry = tool_registry
        self._tool_snapshot = tool_registry.freeze()
        self._tool_executor = ToolExecutor(self._tool_snapshot)
        self._configured_resident_tool_names = tuple(
            configured_resident_tool_names
        )
        self._discoverable_tool_names = tuple(discoverable_tool_names)
        self._skill_runtime = skill_runtime or (
            None if skill_catalog is None else SkillRuntime(skill_catalog)
        )
        self._model_turn_provider = model_turn_provider
        self._model_registry = model_registry
        self._strict_model_provider = strict_model_provider
        self._output_finalizer = output_finalizer
        self._checkpointer = checkpointer or create_agent_checkpointer(None)
        self._runtime_diagnostics = tuple(
            merge_runtime_diagnostics([], runtime_diagnostics)
        )
        self._stream_sink = stream_sink
        self._latency_profile = latency_profile or AgentLatencyProfile()
        self._workspace = workspace
        self._workspace_by_run: dict[str, WorkspaceRuntime] = {}
        self._session_store = session_store
        self._runtime_binding = runtime_binding
        self._lease_owner = str(uuid4())

    async def aclose(self) -> None:
        await aclose_agent_checkpointer(self._checkpointer)

    def initial_state(self, request: AgentRunRequest) -> LoopState:
        return self._initial_state(
            request,
            run_config=request.to_run_config(self._policy),
        )

    def initial_state_from_config(
        self,
        *,
        task: str,
        run_config: AgentRunConfig,
        pending_tool_calls: list[ToolCallPlan] | None = None,
        approved_tool_call_ids: list[str] | None = None,
        denied_tool_call_ids: list[str] | None = None,
        messages: list[BaseMessage] | None = None,
        memory_store: WorkspaceMemoryStore | None = None,
    ) -> LoopState:
        request = AgentRunRequest(
            task=task,
            run_id=run_config.run_id,
            thread_id=run_config.thread_id,
            pending_tool_calls=pending_tool_calls or [],
            approved_tool_call_ids=approved_tool_call_ids or [],
            denied_tool_call_ids=denied_tool_call_ids or [],
            messages=messages or [],
        )
        return self._initial_state(
            request,
            run_config=run_config,
            memory_store=memory_store,
        )

    async def run(self, request: AgentRunRequest) -> AgentRunResult:
        return await self._run_request(request, streaming=False)

    async def chat(self, request: AgentRunRequest) -> AgentRunResult:
        if self._session_store is None or self._runtime_binding is None:
            raise RuntimeError("Session runtime is not configured")
        session_id = request.session_id
        if session_id is None:
            session = self._session_store.create_session(
                self._runtime_binding
            )
        else:
            session = self._session_store.get_session(session_id)
        session = self._ensure_session_workspace(
            session,
            requested_workspace=request.workspace_path,
        )
        session_id = session.session_id
        turn = self._session_store.begin_turn(
            session_id,
            request.task,
            turn_id=request.run_id,
            lease_owner=self._lease_owner,
            lease_seconds=_TURN_LEASE_SECONDS,
        )
        try:
            effective_request = self._session_request(
                request,
                session_id=session_id,
                turn_id=turn.turn_id,
            )
            return await self._run_request(
                effective_request,
                streaming=False,
            )
        except asyncio.CancelledError:
            self._interrupt_session_turn(turn.turn_id)
            raise
        except Exception:
            self._fail_session_turn(turn.turn_id)
            raise

    async def chat_streaming(
        self,
        request: AgentRunRequest,
    ) -> AsyncGenerator[StreamEvent, None]:
        if self._session_store is None or self._runtime_binding is None:
            raise RuntimeError("Session runtime is not configured")
        session_id = request.session_id
        if session_id is None:
            session = self._session_store.create_session(
                self._runtime_binding
            )
        else:
            session = self._session_store.get_session(session_id)
        session = self._ensure_session_workspace(
            session,
            requested_workspace=request.workspace_path,
        )
        session_id = session.session_id
        turn = self._session_store.begin_turn(
            session_id,
            request.task,
            turn_id=request.run_id,
            lease_owner=self._lease_owner,
            lease_seconds=_TURN_LEASE_SECONDS,
        )
        try:
            effective_request = self._session_request(
                request,
                session_id=session_id,
                turn_id=turn.turn_id,
            )
            async for event in self.run_streaming(effective_request):
                yield replace(
                    event,
                    session_id=session_id,
                    run_id=event.run_id or turn.turn_id,
                )
        except (asyncio.CancelledError, GeneratorExit):
            self._interrupt_session_turn(turn.turn_id)
            raise
        except Exception:
            self._fail_session_turn(turn.turn_id)
            raise

    def _session_request(
        self,
        request: AgentRunRequest,
        *,
        session_id: str,
        turn_id: str,
    ) -> AgentRunRequest:
        if self._session_store is None:
            raise RuntimeError("Session runtime is not configured")
        session = self._session_store.get_session(session_id)
        history = self._session_store.history_before_turn(turn_id)
        if history:
            first = history[0]
            if first.role != "user":
                raise RuntimeError(
                    f"Session {session_id} canonical history does not begin "
                    "with a user message"
                )
            initial_task = first.content
            canonical_history = [
                *history[1:],
                ModelMessage(role="user", content=request.task),
            ]
        else:
            initial_task = request.task
            canonical_history = []
        return request.model_copy(
            update={
                "task": initial_task,
                "session_id": session_id,
                "run_id": turn_id,
                "thread_id": turn_id,
                "workspace_path": session.runtime.workspace_path,
                "history_messages": canonical_history,
                "turn_messages": [
                    ModelMessage(role="user", content=request.task)
                ],
            }
        )

    def _ensure_session_workspace(
        self,
        session: SessionRecord,
        *,
        requested_workspace: str | None,
    ) -> SessionRecord:
        if self._session_store is None:
            raise RuntimeError("Session runtime is not configured")
        if session.runtime.workspace_path is not None:
            return session
        if requested_workspace is not None:
            workspace = open_workspace(requested_workspace, create=True)
        elif self._workspace is not None:
            workspace = self._workspace
        elif self._session_store.path is not None:
            database = self._session_store.path.expanduser().resolve()
            workspace = open_workspace(
                database.parent
                / ".agent-workspaces"
                / database.stem
                / session.session_id,
                create=True,
            )
        else:
            workspace = create_temp_workspace()
        runtime = session.runtime.model_copy(
            update={"workspace_path": str(workspace.root)}
        )
        return self._session_store.initialize_session_runtime(
            session.session_id,
            runtime,
        )

    async def run_with_config(
        self,
        *,
        task: str,
        run_config: AgentRunConfig,
        pending_tool_calls: list[ToolCallPlan] | None = None,
        approved_tool_call_ids: list[str] | None = None,
        denied_tool_call_ids: list[str] | None = None,
        messages: list[BaseMessage] | None = None,
        goal_spec: GoalSpec | None = None,
        input_files: list[str] | None = None,
        workspace_path: str | None = None,
        tools: Sequence[str] | None = None,
        disabled_tools: Sequence[str] = (),
        allow_write_tools: bool = False,
        allow_execute_tools: bool = False,
        allow_discovery_tools: bool = False,
    ) -> AgentRunResult:
        request = AgentRunRequest(
            task=task,
            run_id=run_config.run_id,
            thread_id=run_config.thread_id,
            pending_tool_calls=pending_tool_calls or [],
            approved_tool_call_ids=approved_tool_call_ids or [],
            denied_tool_call_ids=denied_tool_call_ids or [],
            messages=messages or [],
            goal_spec=goal_spec,
            input_files=input_files or [],
            workspace_path=workspace_path,
            tools=None if tools is None else tuple(tools),
            disabled_tools=tuple(disabled_tools),
            allow_write_tools=allow_write_tools,
            allow_execute_tools=allow_execute_tools,
            allow_discovery_tools=allow_discovery_tools,
        )
        return await self._run_request(
            request,
            streaming=False,
            run_config=run_config,
        )

    async def run_streaming(
        self,
        request: AgentRunRequest,
    ) -> AsyncGenerator[StreamEvent, None]:
        run_config = request.to_run_config(self._policy)
        lease_task = self._start_lease_heartbeat(run_config)
        state: LoopState | None = None
        workspace: WorkspaceRuntime | None = None
        try:
            (
                state,
                loop,
                workspace,
                started_at,
                checkpoint_store,
                tool_trace_start,
            ) = self._prepare_execution(
                request,
                run_config=run_config,
            )
            async for event in loop.run_streaming(state):
                yield event
        except (asyncio.CancelledError, GeneratorExit):
            if run_config.session_id is not None:
                self._interrupt_session_turn(run_config.run_id)
            RunRegistry.remove(run_config.run_id)
            raise
        except Exception:
            if run_config.session_id is not None:
                self._fail_session_turn(run_config.run_id)
            RunRegistry.remove(run_config.run_id)
            raise
        finally:
            try:
                if state is not None:
                    self._finalize_state(
                        state,
                        started_at=started_at,
                        tool_trace_start=tool_trace_start,
                    )
                    if state["status"] == "paused":
                        await checkpoint_store.save_snapshot(
                            state,
                            reason="service_pause_finalized",
                        )
                    if state["status"] in {"completed", "failed"}:
                        RunRegistry.remove(run_config.run_id)
                    if state["status"] != "running":
                        self._sync_session_turn(state)
                    if workspace is not None:
                        self._workspace_by_run[run_config.run_id] = workspace
            finally:
                await self._stop_lease_heartbeat(lease_task)

    async def _run_request(
        self,
        request: AgentRunRequest,
        *,
        streaming: bool,
        run_config: AgentRunConfig | None = None,
    ) -> AgentRunResult:
        del streaming
        effective_config = run_config or request.to_run_config(self._policy)
        lease_task = self._start_lease_heartbeat(effective_config)
        try:
            return await self._run_request_with_config(
                request,
                run_config=effective_config,
            )
        finally:
            await self._stop_lease_heartbeat(lease_task)

    async def _run_request_with_config(
        self,
        request: AgentRunRequest,
        *,
        run_config: AgentRunConfig,
    ) -> AgentRunResult:
        try:
            (
                state,
                loop,
                workspace,
                started_at,
                checkpoint_store,
                tool_trace_start,
            ) = self._prepare_execution(
                request,
                run_config=run_config,
            )
            result_state = await loop.run(state)
        except asyncio.CancelledError:
            if run_config.session_id is not None:
                self._interrupt_session_turn(run_config.run_id)
            RunRegistry.remove(run_config.run_id)
            raise
        except Exception:
            if run_config.session_id is not None:
                self._fail_session_turn(run_config.run_id)
            RunRegistry.remove(run_config.run_id)
            raise
        self._finalize_state(
            result_state,
            started_at=started_at,
            tool_trace_start=tool_trace_start,
        )
        if result_state["status"] == "paused":
            await checkpoint_store.save_snapshot(
                result_state,
                reason="service_pause_finalized",
            )
        if result_state["status"] in {"completed", "failed"}:
            RunRegistry.remove(run_config.run_id)
        self._sync_session_turn(result_state)
        self._workspace_by_run[run_config.run_id] = workspace
        return AgentRunResult.from_loop_result(
            result_state,
            definition=self._policy,
            workspace_path=str(workspace.root),
        )

    def _sync_session_turn(self, state: LoopState) -> None:
        run_config = state["run_config"]
        if self._session_store is None or run_config.session_id is None:
            return
        self._session_store.sync_turn_messages(
            run_config.run_id,
            state["turn_transcript"],
        )
        if state["status"] == "paused":
            self._session_store.mark_paused(run_config.run_id)
        elif state["status"] == "completed":
            self._session_store.mark_terminal(
                run_config.run_id,
                TurnStatus.COMPLETED,
            )
        elif state["status"] == "failed":
            self._session_store.mark_terminal(
                run_config.run_id,
                TurnStatus.FAILED,
            )

    def _sync_checkpoint_history(self, state: LoopState) -> None:
        run_config = state["run_config"]
        if self._session_store is None or run_config.session_id is None:
            return
        self._session_store.sync_turn_messages(
            run_config.run_id,
            state["turn_transcript"],
        )

    def _interrupt_session_turn(self, turn_id: str) -> None:
        if self._session_store is None:
            return
        turn = self._session_store.get_turn(turn_id)
        if turn.status is TurnStatus.RUNNING:
            self._session_store.mark_interrupted(turn_id)

    def _fail_session_turn(self, turn_id: str) -> None:
        if self._session_store is None:
            return
        turn = self._session_store.get_turn(turn_id)
        if turn.status is TurnStatus.RUNNING:
            self._session_store.mark_terminal(turn_id, TurnStatus.FAILED)

    def _start_lease_heartbeat(
        self,
        run_config: AgentRunConfig,
    ) -> asyncio.Task[None] | None:
        if run_config.session_id is None:
            return None
        return self._start_turn_lease_heartbeat(run_config.run_id)

    def _start_turn_lease_heartbeat(
        self,
        turn_id: str,
    ) -> asyncio.Task[None] | None:
        if self._session_store is None:
            return None
        owner_task = asyncio.current_task()
        heartbeat = asyncio.create_task(
            self._renew_turn_lease(turn_id),
            name=f"turn-lease-heartbeat:{turn_id}",
        )

        def cancel_owner_on_failure(task: asyncio.Task[None]) -> None:
            if task.cancelled() or task.exception() is None:
                return
            if owner_task is not None and not owner_task.done():
                owner_task.cancel()

        heartbeat.add_done_callback(cancel_owner_on_failure)
        return heartbeat

    async def _renew_turn_lease(self, turn_id: str) -> None:
        if self._session_store is None:
            return
        while True:
            await asyncio.sleep(_TURN_LEASE_HEARTBEAT_SECONDS)
            try:
                self._session_store.renew_lease(
                    turn_id,
                    lease_owner=self._lease_owner,
                    lease_seconds=_TURN_LEASE_SECONDS,
                )
            except TurnStateError:
                if self._session_store.get_turn(turn_id).status is not TurnStatus.RUNNING:
                    return
                raise

    @staticmethod
    async def _stop_lease_heartbeat(
        heartbeat: asyncio.Task[None] | None,
    ) -> None:
        if heartbeat is None:
            return
        if not heartbeat.done():
            heartbeat.cancel()
        try:
            await heartbeat
        except asyncio.CancelledError:
            pass

    def _prepare_execution(
        self,
        request: AgentRunRequest,
        *,
        run_config: AgentRunConfig,
    ) -> tuple[
        LoopState,
        AgentLoop,
        WorkspaceRuntime,
        float,
        LangGraphCheckpointStore,
        int,
    ]:
        started_at = time.perf_counter()
        tool_trace_start = len(self._tool_executor.traces)
        workspace = self._workspace_for_request(request)
        if request.input_files:
            import_files(workspace, [Path(item) for item in request.input_files])
        file_manifest = build_file_manifest(workspace)
        memory_store = WorkspaceMemoryStore(
            workspace=workspace,
            policy=run_config.memory_policy,
        )
        state = self._initial_state(
            request,
            run_config=run_config,
            memory_store=memory_store,
            file_manifest=file_manifest,
        )
        checkpoint_store = LangGraphCheckpointStore(
            self._checkpointer,
            run_config=run_config,
            compatibility_config=GoalCompatibilityConfig(
                goal_spec=request.goal_spec
            ),
            snapshot_sink=(
                self._sync_checkpoint_history
                if run_config.session_id is not None
                else None
            ),
        )
        loop = self._build_loop(
            state=state,
            checkpoint_store=checkpoint_store,
            memory_store=memory_store,
            goal_spec=request.goal_spec,
            workspace=workspace,
        )
        return (
            state,
            loop,
            workspace,
            started_at,
            checkpoint_store,
            tool_trace_start,
        )

    def _initial_state(
        self,
        request: AgentRunRequest,
        *,
        run_config: AgentRunConfig,
        memory_store: WorkspaceMemoryStore | None = None,
        file_manifest: FileManifest | None = None,
    ) -> LoopState:
        RunRegistry.remove(run_config.run_id)
        handles = RunRegistry.get_or_create(run_config)
        if memory_store is not None:
            handles.memory_store = memory_store
        state = create_loop_state(
            task=request.task,
            run_config=run_config,
            pending_tool_calls=request.pending_tool_calls,
            messages=request.messages,
            runtime_diagnostics=self._runtime_diagnostics,
            file_manifest=file_manifest,
        )
        state["canonical_transcript"] = [
            snapshot_model_message(message)
            for message in request.history_messages
        ]
        state["turn_transcript"] = [
            snapshot_model_message(message)
            for message in request.turn_messages
        ]
        allow_discovery_tools = (
            bool(self._discoverable_tool_names)
            if request.allow_discovery_tools is None
            else request.allow_discovery_tools
        )
        options = resolve_tool_options(
            self._tool_snapshot,
            default_resident_names=self._default_resident_names(),
            configured_resident_names=self._configured_resident_tool_names,
            discoverable_names=self._discoverable_tool_names,
            tools=request.tools,
            disabled_tools=request.disabled_tools,
            allow_discovery_tools=allow_discovery_tools,
        )
        if options.uses_default_tools:
            state["resident_tool_names"] = list(options.resident_names)
            state["explicit_tool_names"] = []
        else:
            state["resident_tool_names"] = []
            state["explicit_tool_names"] = list(options.resident_names)
        state["disabled_tool_names"] = list(options.disabled_names)
        state["allow_write_tools"] = request.allow_write_tools
        state["allow_execute_tools"] = request.allow_execute_tools
        state["allow_discovery_tools"] = allow_discovery_tools
        state["approved_tool_call_ids"] = list(
            request.approved_tool_call_ids
        )
        state["denied_tool_call_ids"] = list(request.denied_tool_call_ids)
        selected_names = (
            *state["resident_tool_names"],
            *state["explicit_tool_names"],
        )
        selected = select_tools(
            self._tool_snapshot,
            resident_names=selected_names,
            disabled_names=state["disabled_tool_names"],
        )
        state["tool_manifest"] = build_tool_manifest(
            tools=selected,
            resident_tool_names=state["resident_tool_names"],
            explicit_tool_names=state["explicit_tool_names"],
            provider_serializer_revision=state[
                "provider_serializer_revision"
            ],
        )
        context = build_stable_context(
            instructions=(
                self._policy.system_instructions or "You are a helpful agent.",
            ),
            initial_user_task=request.task,
        )
        state["context_revision"] = context.context_revision
        exposed_names = tuple(tool.definition.name for tool in selected)
        manifest = state["tool_manifest"]
        if manifest is None:
            raise RuntimeError("initial tool manifest was not built")
        origin = ToolCallOrigin(
            request_id=f"{run_config.run_id}:initial",
            toolset_revision=manifest.toolset_revision,
            exposed_tool_names=exposed_names,
        )
        for pending in state["pending_tool_calls"]:
            effective_origin = pending.plan.origin or origin
            pending.plan.origin = effective_origin
            state["canonical_tool_calls"][pending.tool_call_id] = ToolCall(
                tool_call_id=pending.tool_call_id,
                tool_name=pending.tool_name,
                arguments=cast(
                    Mapping[str, JsonValue],
                    pending.plan.arguments,
                ),
                origin=effective_origin,
            )
        compacted = MessageCompactor(
            policy=run_config.memory_policy,
            store=memory_store,
        ).compact_initial_state(dict(state))
        result = cast(LoopState, compacted)
        result["latency_profile"] = self._latency_profile
        return result

    def _build_loop(
        self,
        *,
        state: LoopState,
        checkpoint_store: LangGraphCheckpointStore,
        memory_store: WorkspaceMemoryStore | None,
        goal_spec: GoalSpec | None,
        workspace: WorkspaceRuntime,
    ) -> AgentLoop:
        provider = ModelProviderResolver(
            model_turn_provider=self._model_turn_provider,
            model_registry=self._model_registry,
            policy=self._policy,
            registry_snapshot=self._tool_snapshot,
            strict_model_provider=self._strict_model_provider,
            stream_sink=self._stream_sink,
            skill_runtime=self._skill_runtime,
        ).resolve(state)
        output_finalizer = self._resolve_output_finalizer(state)
        execution_context = ToolExecutionContext(
            workspace_root=workspace.root,
            cwd=workspace.root,
            allow_write_tools=state["allow_write_tools"],
            allow_execute_tools=state["allow_execute_tools"],
        )
        return AgentLoop(
            definition=self._policy,
            model_provider=provider,
            context_manager=LoopContextCompactor(store=memory_store),
            tool_executor=self._tool_executor,
            registry_snapshot=self._tool_snapshot,
            execution_context=execution_context,
            checkpoint_store=checkpoint_store,
            stop_hook_runner=StopHookRunner(
                hooks=build_stop_hooks(
                    definition=self._policy,
                    output_finalizer=output_finalizer,
                    goal_spec=goal_spec,
                ),
                max_blocks=self._policy.max_stop_hook_blocks,
            ),
            finish_candidate_builder=FinishCandidateBuilder(),
            stream_sink=self._stream_sink,
            skill_runtime=self._skill_runtime,
            discoverable_tool_names=self._discoverable_tool_names,
        )

    def _default_resident_names(self) -> tuple[str, ...]:
        installed = tuple(self._tool_snapshot)
        baseline = tuple(
            name for name in RESIDENT_CODING_TOOL_NAMES if name in installed
        )
        if baseline:
            return baseline
        return tuple(
            name for name in self._policy.allowed_tools if name in installed
        )

    def _workspace_for_request(
        self,
        request: AgentRunRequest,
    ) -> WorkspaceRuntime:
        if request.workspace_path is not None:
            return open_workspace(request.workspace_path, create=True)
        if self._workspace is not None:
            return self._workspace
        return create_temp_workspace()

    def _resolve_output_finalizer(
        self,
        state: LoopState,
    ) -> StructuredOutputFinalizer | None:
        if self._output_finalizer is not None:
            return self._output_finalizer
        if self._policy.output_model is None or self._model_registry is None:
            return None
        try:
            return create_model_structured_output_finalizer(
                self._model_registry
            )
        except Exception as exc:
            state["runtime_diagnostics"].append(
                RuntimeDiagnostic.from_exception(
                    code="structured_output_finalizer_initialization_failed",
                    component="structured_output_finalizer",
                    error=exc,
                )
            )
            return None

    def _finalize_state(
        self,
        state: LoopState,
        *,
        started_at: float,
        tool_trace_start: int,
    ) -> None:
        phase_tool_latency = sum(
            trace.duration_ms
            for trace in self._tool_executor.traces[tool_trace_start:]
        )
        profile = state.get("latency_profile") or AgentLatencyProfile()
        tool_latency = profile.tool_latency_ms + phase_tool_latency
        total_ms = profile.total_ms + (time.perf_counter() - started_at) * 1000
        if profile.total_ms == 0:
            total_ms += profile.startup_ms + profile.build_service_ms
        total_ms = max(
            total_ms,
            profile.startup_ms
            + profile.build_service_ms
            + profile.model_latency_ms
            + tool_latency,
        )
        state["latency_profile"] = profile.model_copy(
            update={
                "tool_latency_ms": tool_latency,
                "total_ms": total_ms,
            }
        )

    async def resume_turn(
        self,
        *,
        turn_id: str,
        action: str,
        user_input: str | None = None,
    ) -> AgentRunResult:
        if self._session_store is None:
            raise RuntimeError("Session runtime is not configured")
        turn = self._session_store.get_turn(turn_id)
        if turn.status in {TurnStatus.COMPLETED, TurnStatus.FAILED}:
            raise TurnStateError(
                f"Turn {turn_id} is {turn.status.value} and cannot resume"
            )
        started_at = time.perf_counter()
        checkpoint_store = LangGraphCheckpointStore(
            self._checkpointer,
            run_config=self._checkpoint_lookup_config(turn_id),
            snapshot_sink=self._sync_checkpoint_history,
        )
        restored = await checkpoint_store.load_for_resume()
        if restored is None:
            raise KeyError(f"No checkpoint found for turn_id={turn_id}")
        if restored["run_config"].session_id != turn.session_id:
            raise RuntimeError(
                f"Checkpoint for Turn {turn_id} does not belong to Session "
                f"{turn.session_id}"
            )
        drift_result = await self._reconcile_manifest(
            restored,
            checkpoint_store=checkpoint_store,
        )
        if drift_result is not None:
            return drift_result
        response, abort = _response_for_resume_action(
            restored,
            action=action,
            user_input=user_input,
        )
        self._session_store.claim_for_resume(
            turn_id,
            lease_owner=self._lease_owner,
            lease_seconds=_TURN_LEASE_SECONDS,
        )
        lease_task = self._start_turn_lease_heartbeat(turn_id)
        try:
            if response is not None:
                state = await checkpoint_store.apply_human_response(response)
            else:
                state = restored
                state["status"] = "running"
                state["pause"] = None
                state["approval_request"] = None
                state["approval_response"] = None
            self._hydrate_session_state(state, turn_id=turn_id)
            if user_input is not None and not abort:
                message = ModelMessage(role="user", content=user_input)
                state["canonical_transcript"] = [
                    *state["canonical_transcript"],
                    message,
                ]
                state["turn_transcript"] = [
                    *state["turn_transcript"],
                    message,
                ]
            if abort:
                state["status"] = "failed"
                state["pause"] = None
                state["terminal"] = LoopTerminal(
                    status="failed",
                    stop_reason="user_aborted",
                    error="The user aborted this Turn.",
                )
                state["latest_transition"] = LoopTransition(
                    reason="failed",
                    iteration=state["iteration"],
                    detail={"stop_reason": "user_aborted"},
                )
                await checkpoint_store.save_snapshot(
                    state,
                    reason="user_aborted",
                )
                self._sync_session_turn(state)
                RunRegistry.remove(turn_id)
                workspace = self._workspace or create_temp_workspace()
                return AgentRunResult.from_loop_result(
                    state,
                    definition=self._policy,
                    workspace_path=str(workspace.root),
                )
            await checkpoint_store.save_snapshot(
                state,
                reason="session_resume_prepared",
            )
            workspace = self._workspace or create_temp_workspace()
            return await self._continue_resumed_state(
                state,
                checkpoint_store=checkpoint_store,
                workspace=workspace,
                started_at=started_at,
            )
        except BaseException:
            self._interrupt_session_turn(turn_id)
            RunRegistry.remove(turn_id)
            raise
        finally:
            await self._stop_lease_heartbeat(lease_task)

    async def resume(
        self,
        *,
        run_id: str,
        response: HumanInputResponse,
        workspace_path: str | None = None,
    ) -> AgentRunResult:
        started_at = time.perf_counter()
        checkpoint_store = LangGraphCheckpointStore(
            self._checkpointer,
            run_config=self._checkpoint_lookup_config(run_id),
            snapshot_sink=(
                self._sync_checkpoint_history
                if self._session_store is not None
                else None
            ),
        )
        restored = await checkpoint_store.load_for_resume()
        if restored is None:
            raise KeyError(f"No checkpoint found for run_id={run_id}")
        drift_result = await self._reconcile_manifest(
            restored,
            checkpoint_store=checkpoint_store,
        )
        if drift_result is not None:
            return drift_result
        state = await checkpoint_store.apply_human_response(response)
        workspace = (
            open_workspace(workspace_path)
            if workspace_path is not None
            else self._workspace_by_run.get(run_id)
            or self._workspace
            or create_temp_workspace()
        )
        return await self._continue_resumed_state(
            state,
            checkpoint_store=checkpoint_store,
            workspace=workspace,
            started_at=started_at,
        )

    async def _continue_resumed_state(
        self,
        state: LoopState,
        *,
        checkpoint_store: LangGraphCheckpointStore,
        workspace: WorkspaceRuntime,
        started_at: float,
    ) -> AgentRunResult:
        run_config = state["run_config"]
        run_id = run_config.run_id
        RunRegistry.get_or_create(run_config)
        memory_store = WorkspaceMemoryStore(
            workspace=workspace,
            policy=run_config.memory_policy,
        )
        RunRegistry.get(run_id).memory_store = memory_store
        loop = self._build_loop(
            state=state,
            checkpoint_store=checkpoint_store,
            memory_store=memory_store,
            goal_spec=checkpoint_store.compatibility_config.goal_spec,
            workspace=workspace,
        )
        tool_trace_start = len(self._tool_executor.traces)
        result_state = await loop.run(state)
        self._finalize_state(
            result_state,
            started_at=started_at,
            tool_trace_start=tool_trace_start,
        )
        if result_state["status"] == "paused":
            await checkpoint_store.save_snapshot(
                result_state,
                reason="service_pause_finalized",
            )
        if result_state["status"] in {"completed", "failed"}:
            RunRegistry.remove(run_id)
        self._sync_session_turn(result_state)
        self._workspace_by_run[run_id] = workspace
        return AgentRunResult.from_loop_result(
            result_state,
            definition=self._policy,
            workspace_path=str(workspace.root),
        )

    def _hydrate_session_state(
        self,
        state: LoopState,
        *,
        turn_id: str,
    ) -> None:
        if self._session_store is None:
            raise RuntimeError("Session runtime is not configured")
        turn = self._session_store.get_turn(turn_id)
        before = self._session_store.history_before_turn(turn_id)
        persisted_turn = self._session_store.turn_history(turn_id)
        checkpoint_turn = tuple(state.get("turn_transcript", ()))
        if checkpoint_turn[: len(persisted_turn)] == persisted_turn:
            current = checkpoint_turn
        elif persisted_turn[: len(checkpoint_turn)] == checkpoint_turn:
            current = persisted_turn
        else:
            raise RuntimeError(
                f"Checkpoint and canonical history conflict for Turn {turn_id}"
            )
        initial = ModelMessage(role="user", content=turn.user_message)
        if not current:
            current = (initial,)
        if current[0] != initial:
            raise RuntimeError(
                f"Canonical history for Turn {turn_id} does not begin with "
                "its user message"
            )
        if before:
            first = before[0]
            if first.role != "user":
                raise RuntimeError(
                    f"Session {turn.session_id} canonical history does not "
                    "begin with a user message"
                )
            state["task"] = first.content
            state["canonical_transcript"] = [*before[1:], *current]
        else:
            state["task"] = turn.user_message
            state["canonical_transcript"] = list(current[1:])
        state["turn_transcript"] = list(current)

    async def _reconcile_manifest(
        self,
        state: LoopState,
        *,
        checkpoint_store: LangGraphCheckpointStore,
    ) -> AgentRunResult | None:
        persisted = state.get("tool_manifest")
        if persisted is None:
            return None
        resident = tuple(
            name
            for name in state.get("resident_tool_names", ())
            if name in self._tool_snapshot
        )
        explicit = tuple(
            name
            for name in state.get("explicit_tool_names", ())
            if name in self._tool_snapshot
        )
        active = tuple(
            name
            for name in state.get("active_tool_names", ())
            if name in self._tool_snapshot
        )
        tools = tuple(
            self._tool_snapshot[name]
            for name in (*resident, *explicit, *active)
        )
        rebuilt = build_tool_manifest(
            tools=tools,
            resident_tool_names=resident,
            explicit_tool_names=explicit,
            active_tool_names=active,
            provider_serializer_revision=state[
                "provider_serializer_revision"
            ],
        )
        calls = state.get("canonical_tool_calls", {})
        dependent = tuple(
            calls[item.tool_call_id]
            for item in state["pending_tool_calls"]
            if item.tool_call_id in calls
        )
        decision = reconcile_tool_manifest(
            persisted=persisted,
            rebuilt=rebuilt,
            pending_tool_calls=dependent,
            paused_tool_calls=(),
        )
        if decision.status is ToolManifestDriftStatus.MATCH:
            return None
        if decision.status is ToolManifestDriftStatus.NEW_REVISION_REQUIRED:
            state["resident_tool_names"] = list(resident)
            state["explicit_tool_names"] = list(explicit)
            state["active_tool_names"] = list(decision.active_tool_names)
            state["tool_manifest"] = rebuilt
            await checkpoint_store.save_snapshot(
                state,
                reason="tool_manifest_revision",
            )
            return None
        request = HumanInputRequest(
            request_id=f"hir_{uuid4().hex[:12]}",
            kind="tool_reconciliation",
            question=(
                "A pending tool definition changed; reconcile it before "
                "execution."
            ),
            context={
                "reason": decision.reason,
                "error_code": "tool_definition_changed",
                "tool_call_id": (
                    decision.dependent_tool_calls[0].tool_call_id
                    if decision.dependent_tool_calls
                    else ""
                ),
                "tool_call_ids": [
                    call.tool_call_id for call in decision.dependent_tool_calls
                ],
            },
            options=["mark_failed"],
        )
        state["status"] = "paused"
        state["approval_request"] = request
        state["pause"] = LoopPause(
            reason="tool_definition_changed",
            request=request,
        )
        state["latest_transition"] = LoopTransition(
            reason="approval_required",
            iteration=state["iteration"],
            detail={"reason": "tool_definition_changed"},
        )
        await checkpoint_store.save_snapshot(
            state,
            reason="tool_definition_changed",
        )
        return AgentRunResult.from_loop_result(
            state,
            definition=self._policy,
        )

    def pending_human_input_request(
        self,
        *,
        run_id: str,
    ) -> HumanInputRequest:
        state = LangGraphCheckpointStore(
            self._checkpointer,
            run_config=self._checkpoint_lookup_config(run_id),
        ).load_latest_sync()
        return _pending_request(state, run_id=run_id)

    async def apending_human_input_request(
        self,
        *,
        run_id: str,
    ) -> HumanInputRequest:
        checkpoint_store = LangGraphCheckpointStore(
            self._checkpointer,
            run_config=self._checkpoint_lookup_config(run_id),
            snapshot_sink=(
                self._sync_checkpoint_history
                if self._session_store is not None
                else None
            ),
        )
        state = await checkpoint_store.load_for_resume()
        if state is not None:
            await self._reconcile_manifest(
                state,
                checkpoint_store=checkpoint_store,
            )
        return _pending_request(state, run_id=run_id)

    def _checkpoint_lookup_config(self, run_id: str) -> AgentRunConfig:
        return AgentRunConfig(
            run_id=run_id,
            thread_id=run_id,
            agent_type=self._policy.agent_type,
            max_depth=self._policy.max_depth,
            access_policy=(
                self._policy.access_policy_ceiling or AccessPolicy.default()
            ),
            tool_policy=self._policy.tool_policy,
        )


def _response_for_resume_action(
    state: LoopState,
    *,
    action: str,
    user_input: str | None,
) -> tuple[HumanInputResponse | None, bool]:
    if action == "abort":
        return None, True
    request = state["approval_request"]
    if request is None and state["pause"] is not None:
        request = state["pause"].request
    if request is None:
        if action != "continue":
            raise ValueError(
                "An interrupted Turn without a human-input request only "
                "supports action='continue' or action='abort'"
            )
        return None, False
    if request.kind == "tool_approval":
        if action not in {"allow_once", "deny"}:
            raise ValueError(
                "A tool approval Turn supports allow_once, deny, or abort"
            )
        tool_call_ids = [
            item.approval_id or item.tool_call_id
            for item in request.tool_calls
        ]
        return (
            HumanInputResponse(
                request_id=request.request_id,
                decision=cast(Any, action),
                approved_tool_call_ids=(
                    tool_call_ids if action == "allow_once" else []
                ),
                denied_tool_call_ids=(
                    tool_call_ids if action == "deny" else []
                ),
                user_message=user_input,
            ),
            False,
        )
    if request.kind == "tool_reconciliation":
        if action not in {"mark_completed", "mark_failed"}:
            raise ValueError(
                "An outcome-unknown tool only supports mark_completed, "
                "mark_failed, or abort; replay is forbidden"
            )
        return (
            HumanInputResponse(
                request_id=request.request_id,
                decision=cast(Any, action),
                user_message=user_input,
            ),
            False,
        )
    if action == "continue":
        if request.kind == "clarification" and (
            user_input is None or not user_input.strip()
        ):
            raise ValueError("Clarification resume requires non-empty user_input")
        message = user_input
    elif action in request.options:
        message = user_input or action
    else:
        raise ValueError(
            f"Unsupported action {action!r} for {request.kind} request"
        )
    return (
        HumanInputResponse(
            request_id=request.request_id,
            decision="continue",
            user_message=message,
        ),
        False,
    )


def _pending_request(
    state: LoopState | None,
    *,
    run_id: str,
) -> HumanInputRequest:
    if state is None:
        raise KeyError(f"No checkpoint found for run_id={run_id}")
    request = state["approval_request"]
    if request is None and state["pause"] is not None:
        request = state["pause"].request
    if request is None:
        raise KeyError(f"No pending human input request for run_id={run_id}")
    return request


def _result_mapping(result: ToolResult) -> Mapping[str, object] | None:
    value = result.structured_content
    return value if isinstance(value, Mapping) else None


def _result_provenance(
    results: Sequence[ToolResult],
) -> tuple[list[EvidenceItem], list[AnswerCitation]]:
    evidence: list[EvidenceItem] = []
    citations: list[AnswerCitation] = []
    for result in results:
        if result.is_error:
            continue
        payload = _result_mapping(result)
        if payload is None:
            continue
        raw_evidence = payload.get("evidence", ()) or ()
        if isinstance(raw_evidence, Sequence) and not isinstance(
            raw_evidence,
            (str, bytes),
        ):
            for item in raw_evidence:
                evidence.append(EvidenceItem.model_validate(item))
        raw_citations = payload.get("citations", ()) or ()
        if isinstance(raw_citations, Sequence) and not isinstance(
            raw_citations,
            (str, bytes),
        ):
            for item in raw_citations:
                if not isinstance(item, (Mapping, AnswerCitation)):
                    continue
                citations.append(AnswerCitation.model_validate(item))
    return evidence, citations


def _result_flag(results: Sequence[ToolResult], name: str) -> bool:
    for result in reversed(results):
        payload = _result_mapping(result)
        if payload is not None and bool(payload.get(name, False)):
            return True
    return False


def _restore_final_output(
    raw_output: ValidatedFinalOutput | dict[str, object] | None,
    *,
    definition: AgentRuntimePolicy | None,
) -> BaseModel | None:
    if (
        raw_output is None
        or definition is None
        or definition.output_model is None
    ):
        return None
    envelope = ValidatedFinalOutput.model_validate(raw_output)
    expected_path = output_model_path(definition.output_model)
    if envelope.model_path != expected_path:
        raise ValueError(
            "Checkpoint final output model does not match configured output model"
        )
    return definition.output_model.model_validate(envelope.data)


__all__ = ["AgentRunRequest", "AgentRunResult", "AgentService"]

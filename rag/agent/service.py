from __future__ import annotations

import logging
import time
from collections.abc import AsyncGenerator, Mapping, Sequence
from pathlib import Path
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
from rag.agent.loop.runtime import AgentLoop, ModelTurnProvider
from rag.agent.loop.state import (
    LoopPause,
    LoopState,
    LoopTransition,
    create_loop_state,
)
from rag.agent.loop.stop_hooks import StopHookRunner, build_stop_hooks
from rag.agent.memory.compactor import LoopContextCompactor, MessageCompactor
from rag.agent.memory.models import MemoryPolicy
from rag.agent.memory.store import WorkspaceMemoryStore
from rag.agent.tools.builtins import RESIDENT_CODING_TOOL_NAMES
from rag.agent.tools.executor import ToolExecutor
from rag.agent.tools.permissions import ToolExecutionContext
from rag.agent.tools.registry import ToolRegistry
from rag.agent.tools.selection import resolve_tool_options, select_tools
from rag.agent.tools.tool import ToolCall, ToolCallOrigin, ToolResult
from rag.agent.workspace import (
    WorkspaceRuntime,
    create_temp_workspace,
    import_files,
    open_workspace,
)
from rag.schema.query import AnswerCitation, EvidenceItem
from rag.schema.runtime import AccessPolicy

logger = logging.getLogger(__name__)


class AgentRunRequest(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    task: str = Field(min_length=1)
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
    input_files: list[str] = Field(default_factory=list)
    workspace_path: str | None = None
    memory_policy: MemoryPolicy | None = None
    goal_spec: GoalSpec | None = None
    tools: tuple[str, ...] | None = None
    disabled_tools: tuple[str, ...] = ()
    allow_write_tools: bool = False
    allow_execute_tools: bool = False
    allow_discovery_tools: bool = False

    @field_validator("tools", "disabled_tools", mode="before")
    @classmethod
    def _tuple_names(cls, value: object) -> object:
        if value is None:
            return None
        if isinstance(value, (str, bytes)):
            raise TypeError("tool options must be sequences of names")
        return tuple(value)  # type: ignore[arg-type]

    def to_run_config(self, definition: AgentRuntimePolicy) -> AgentRunConfig:
        run_id = self.run_id or f"run_{uuid4().hex[:12]}"
        return AgentRunConfig(
            run_id=run_id,
            thread_id=self.thread_id or run_id,
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
        stream_sink: object | None = None,
        mcp_registry: object | None = None,
        skill_catalog: object | None = None,
        strict_model_provider: bool = True,
        latency_profile: AgentLatencyProfile | None = None,
        workspace: WorkspaceRuntime | None = None,
        configured_resident_tool_names: Sequence[str] = (),
    ) -> None:
        del (
            retrieval_hint_provider,
            subagent_runner,
            catalog,
            mcp_registry,
            skill_catalog,
        )
        if definition is not None and policy is not None:
            raise ValueError("Provide either 'definition' or 'policy', not both")
        self._policy = definition or policy
        if self._policy is None:
            raise ValueError("Provide either 'definition' or 'policy'")
        self._tool_registry = tool_registry
        self._tool_snapshot = tool_registry.freeze()
        self._tool_executor = ToolExecutor(self._tool_snapshot)
        self._configured_resident_tool_names = tuple(
            configured_resident_tool_names
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
    ) -> AsyncGenerator[object, None]:
        run_config = request.to_run_config(self._policy)
        state, loop, workspace, started_at = self._prepare_execution(
            request,
            run_config=run_config,
        )
        try:
            async for event in loop.run_streaming(state):
                yield event
        finally:
            self._finalize_state(state, started_at=started_at)
            if state["status"] in {"completed", "failed"}:
                RunRegistry.remove(run_config.run_id)
            self._workspace_by_run[run_config.run_id] = workspace

    async def _run_request(
        self,
        request: AgentRunRequest,
        *,
        streaming: bool,
        run_config: AgentRunConfig | None = None,
    ) -> AgentRunResult:
        del streaming
        effective_config = run_config or request.to_run_config(self._policy)
        state, loop, workspace, started_at = self._prepare_execution(
            request,
            run_config=effective_config,
        )
        try:
            result_state = await loop.run(state)
        except Exception:
            RunRegistry.remove(effective_config.run_id)
            raise
        self._finalize_state(result_state, started_at=started_at)
        if result_state["status"] in {"completed", "failed"}:
            RunRegistry.remove(effective_config.run_id)
        self._workspace_by_run[effective_config.run_id] = workspace
        return AgentRunResult.from_loop_result(
            result_state,
            definition=self._policy,
            workspace_path=str(workspace.root),
        )

    def _prepare_execution(
        self,
        request: AgentRunRequest,
        *,
        run_config: AgentRunConfig,
    ) -> tuple[LoopState, AgentLoop, WorkspaceRuntime, float]:
        started_at = time.perf_counter()
        workspace = self._workspace_for_request(request)
        if request.input_files:
            import_files(workspace, [Path(item) for item in request.input_files])
        memory_store = WorkspaceMemoryStore(
            workspace=workspace,
            policy=run_config.memory_policy,
        )
        state = self._initial_state(
            request,
            run_config=run_config,
            memory_store=memory_store,
        )
        checkpoint_store = LangGraphCheckpointStore(
            self._checkpointer,
            run_config=run_config,
            compatibility_config=GoalCompatibilityConfig(
                goal_spec=request.goal_spec
            ),
        )
        loop = self._build_loop(
            state=state,
            checkpoint_store=checkpoint_store,
            memory_store=memory_store,
            goal_spec=request.goal_spec,
            workspace=workspace,
        )
        return state, loop, workspace, started_at

    def _initial_state(
        self,
        request: AgentRunRequest,
        *,
        run_config: AgentRunConfig,
        memory_store: WorkspaceMemoryStore | None = None,
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
        )
        options = resolve_tool_options(
            self._tool_snapshot,
            default_resident_names=self._default_resident_names(),
            configured_resident_names=self._configured_resident_tool_names,
            tools=request.tools,
            disabled_tools=request.disabled_tools,
            allow_discovery_tools=request.allow_discovery_tools,
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
        state["allow_discovery_tools"] = request.allow_discovery_tools
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
        origin = ToolCallOrigin(
            request_id=f"{run_config.run_id}:initial",
            toolset_revision=state["tool_manifest"].toolset_revision,
            exposed_tool_names=exposed_names,
        )
        for pending in state["pending_tool_calls"]:
            effective_origin = pending.plan.origin or origin
            pending.plan.origin = effective_origin
            state["canonical_tool_calls"][pending.tool_call_id] = ToolCall(
                tool_call_id=pending.tool_call_id,
                tool_name=pending.tool_name,
                arguments=pending.plan.arguments,
                origin=effective_origin,
            )
        compacted = MessageCompactor(
            policy=run_config.memory_policy,
            store=memory_store,
        ).compact_initial_state(dict(state))
        result = state.__class__(compacted)
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
    ) -> None:
        tool_latency = sum(
            trace.duration_ms for trace in self._tool_executor.traces
        )
        profile = state.get("latency_profile") or AgentLatencyProfile()
        state["latency_profile"] = profile.model_copy(
            update={
                "tool_latency_ms": tool_latency,
                "total_ms": self._latency_profile.startup_ms
                + self._latency_profile.build_service_ms
                + (time.perf_counter() - started_at) * 1000,
            }
        )

    async def resume(
        self,
        *,
        run_id: str,
        response: HumanInputResponse,
        workspace_path: str | None = None,
    ) -> AgentRunResult:
        started_at = time.perf_counter()
        lookup_config = self._checkpoint_lookup_config(run_id)
        checkpoint_store = LangGraphCheckpointStore(
            self._checkpointer,
            run_config=lookup_config,
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
        run_config = state["run_config"]
        RunRegistry.get_or_create(run_config)
        workspace = (
            open_workspace(workspace_path)
            if workspace_path is not None
            else self._workspace_by_run.get(run_id)
            or self._workspace
            or create_temp_workspace()
        )
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
        result_state = await loop.run(state)
        self._finalize_state(result_state, started_at=started_at)
        if result_state["status"] in {"completed", "failed"}:
            RunRegistry.remove(run_id)
        return AgentRunResult.from_loop_result(
            result_state,
            definition=self._policy,
            workspace_path=str(workspace.root),
        )

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
            options=["mark_failed", "retry_new_operation"],
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
        for item in payload.get("evidence", ()) or ():
            evidence.append(EvidenceItem.model_validate(item))
        for item in payload.get("citations", ()) or ():
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

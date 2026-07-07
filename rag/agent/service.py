from __future__ import annotations

import logging
import time
from collections.abc import AsyncGenerator, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast
from uuid import uuid4

from langchain_core.messages import BaseMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from pydantic import BaseModel, ConfigDict, Field

from rag.agent.capabilities.catalog import (
    DeferredToolStore,
    SearchCandidate,
    ToolCatalog,
)
from rag.agent.capabilities.context import deferred_store_var, iteration_var
from rag.agent.capabilities.tool_search import (
    ActivateToolsInput,
    ActivateToolsOutput,
    ToolSearchInput,
    ToolSearchOutput,
    execute_activate_tools,
    execute_tool_search,
)
from rag.agent.core.checkpointing import (
    LangGraphCheckpointStore,
    aclose_agent_checkpointer,
    create_agent_checkpointer,
)
from rag.agent.core.context import AgentRunConfig, RunRegistry, derive_child_config
from rag.agent.core.definition import AgentRuntimePolicy
from rag.agent.core.delegation import AgentDelegationRequest, DelegatedAgentRunner, ParentAgentContext
from rag.agent.core.finalization import FinishCandidateBuilder
from rag.agent.core.goal_contract import GoalCompatibilityConfig, GoalSpec
from rag.agent.core.human_input import HumanInputRequest, HumanInputResponse
from rag.agent.core.llm_registry import ModelResolver
from rag.agent.core.model_provider_runtime import ModelProviderResolver
from rag.agent.core.output_finalizer import (
    StructuredOutputFinalizer,
    create_model_structured_output_finalizer,
)
from rag.agent.core.output_models import (
    ValidatedFinalOutput,
    output_model_path,
)
from rag.agent.core.runtime_diagnostics import (
    AgentLatencyProfile,
    RuntimeDiagnostic,
    merge_runtime_diagnostics,
)
from rag.agent.core.runtime_ports import RetrievalHintProvider
from rag.agent.core.tool_execution import ToolExecutionService
from rag.agent.core.turn_contracts import ToolCallPlan
from rag.agent.loop.runtime import AgentLoop, ModelTurnProvider
from rag.agent.loop.state import (
    LoopState,
    append_loop_diagnostic,
    create_loop_state,
)
from rag.agent.loop.stop_hooks import StopHookRunner, build_stop_hooks
from rag.agent.memory.compactor import (
    LoopContextCompactor,
    MessageCompactor,
)
from rag.agent.memory.models import MemoryPolicy
from rag.agent.memory.persistent import PersistentMemoryStore
from rag.agent.memory.persistent.runtime import PersistentMemoryRuntime
from rag.agent.memory.store import WorkspaceMemoryStore
from rag.agent.tooling import (
    ToolDiscoveryState,
    ToolExecutor as NewToolExecutor,
    ToolExecutorLoopAdapter,
    ToolRegistry as NewToolRegistry,
    ToolSurfaceRequest,
    install_minimal_workspace_tools,
)
from rag.agent.tools.catalog_assembly import build_tool_catalog
from rag.agent.tools.mcp_adapter import MCPToolRegistry
from rag.agent.tools.registry import ToolRegistry, ToolRunner
from rag.agent.tools.runtime_registry_builder import RuntimeToolRegistryBuilder
from rag.agent.tools.spec import ExecutionCategory, ToolPermissions, ToolResult, ToolSpec
from rag.agent.tools.workspace_tools import create_workspace_tools
from rag.schema.query import AnswerCitation, EvidenceItem
from rag.schema.runtime import AccessPolicy

if TYPE_CHECKING:
    from rag.agent.file_manifest import FileManifest

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
    tool_surface_request: ToolSurfaceRequest | None = None
    tool_discovery_state: ToolDiscoveryState | None = None

    def to_run_config(self, definition: AgentRuntimePolicy) -> AgentRunConfig:
        run_id = self.run_id or f"run_{uuid4().hex[:12]}"
        return AgentRunConfig(
            run_id=run_id,
            thread_id=self.thread_id or run_id,
            max_turns=self.max_turns,
            agent_type=definition.agent_type,
            max_context_tokens=self.max_context_tokens,
            llm_budget_total=self.llm_budget_total,
            max_depth=definition.max_depth if self.max_depth is None else self.max_depth,
            access_policy=definition.access_policy_ceiling or AccessPolicy.default(),
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
    output_validation_errors: list[dict[str, object]] = Field(default_factory=list)
    stop_reason: str | None = None
    tool_results: list[ToolResult] = Field(default_factory=list)
    evidence: list[EvidenceItem] = Field(default_factory=list)
    citations: list[AnswerCitation] = Field(default_factory=list)
    iteration: int = 0
    groundedness_flag: bool = False
    insufficient_evidence_flag: bool = False
    needs_user_input: str | None = None
    human_input_request: object | None = None
    pending_tool_calls_summary: list[dict[str, object]] = Field(default_factory=list)
    workspace_path: str | None = None
    runtime_diagnostics: list[RuntimeDiagnostic] = Field(default_factory=list)
    tool_call_metrics: object | None = None  # ToolCallMetrics | None
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
        is_terminal = state["status"] in {"completed", "failed"}
        terminal = state["terminal"]
        pause = state["pause"]
        pending = state["pending_tool_calls"]

        # PR2: derive evidence/citations from tool_results, not deprecated state fields
        evidence: list[EvidenceItem] = []
        citations: list[AnswerCitation] = []
        for result in state.get("tool_results", []):
            if result.status == "ok" and result.output is not None:
                ev = getattr(result.output, "evidence", None)
                if isinstance(ev, list):
                    for item in ev:
                        if isinstance(item, EvidenceItem):
                            evidence.append(item)
                        else:
                            evidence.append(EvidenceItem.model_validate(item))
                ct = getattr(result.output, "citations", None)
                if isinstance(ct, list):
                    for item in ct:
                        if isinstance(item, AnswerCitation):
                            citations.append(item)
                        else:
                            citations.append(AnswerCitation.model_validate(item))

        return cls(
            run_id=run_config.run_id,
            thread_id=run_config.thread_id,
            status=("done" if state["status"] == "completed" else state["status"]),
            final_answer=state["finish_state"].final_answer,
            final_output=_restore_final_output(
                state["finish_state"].final_output,
                definition=definition,
            ),
            output_validation_errors=list(state["finish_state"].output_validation_errors),
            stop_reason=(None if terminal is None else terminal.stop_reason),
            tool_results=list(state["tool_results"]),
            evidence=evidence,
            citations=citations,
            iteration=state["iteration"],
            groundedness_flag=_derive_groundedness(list(state["tool_results"])),
            insufficient_evidence_flag=_derive_insufficient_evidence(list(state["tool_results"])),
            needs_user_input=(None if is_terminal or pause is None else pause.reason),
            human_input_request=(None if is_terminal else state["approval_request"]),
            pending_tool_calls_summary=[
                {
                    "tool_call_id": call.tool_call_id,
                    "tool_name": call.tool_name,
                }
                for call in pending
            ],
            workspace_path=workspace_path,
            runtime_diagnostics=list(state["runtime_diagnostics"]),
            tool_call_metrics=state.get("tool_call_metrics"),
            latency_profile=state.get("latency_profile"),
        )


@dataclass
class _RunContext:
    """Prepared per-run context — the single source of truth for a run's setup.

    Extracted from the duplicated setup logic in run_with_config / run_streaming.
    Following Claude Code's "thin wrapper" philosophy: one path, clean assembly.
    """

    task: str
    workspace: Any
    manifest: Any  # FileManifest | None
    workspace_tools: list[Any]  # list[BaseTool]
    file_tools_to_activate: set[str]
    runtime_registry: ToolRegistry
    tooling_registry: NewToolRegistry
    memory_store: WorkspaceMemoryStore
    persistent_store: PersistentMemoryStore


def _elapsed_ms(started_at: float) -> float:
    return (time.perf_counter() - started_at) * 1000


def _profile_with_run_timings(
    profile: AgentLatencyProfile,
    *,
    prepare_latency_ms: float,
    tool_latency_ms: float,
    finalize_latency_ms: float,
    total_ms: float,
) -> AgentLatencyProfile:
    return profile.model_copy(
        update={
            "prepare_latency_ms": profile.prepare_latency_ms + prepare_latency_ms,
            "tool_latency_ms": tool_latency_ms,
            "finalize_latency_ms": profile.finalize_latency_ms + finalize_latency_ms,
            "total_ms": total_ms,
        }
    )


def _default_tool_surface_request() -> ToolSurfaceRequest:
    """Default service surface: no model-visible tools without explicit config."""
    return ToolSurfaceRequest(force_empty=True)


class _TaskChildRunner:
    """Service-layer runner for the generic task tool.

    Tools may request delegation, but service owns child-loop construction.
    """

    def __init__(
        self,
        *,
        policy: AgentRuntimePolicy,
        tool_registry: ToolRegistry,
        model_turn_provider: ModelTurnProvider | None,
        retrieval_hint_provider: RetrievalHintProvider | None,
    ) -> None:
        self._policy = policy
        self._tool_registry = tool_registry
        self._model_turn_provider = model_turn_provider
        self._retrieval_hint_provider = retrieval_hint_provider

    async def run_delegated_task(
        self,
        *,
        request: AgentDelegationRequest,
        parent_state: ParentAgentContext,
    ) -> AgentRunResult:
        child_policy = self._child_policy()
        child_config = self._derive_child_config(
            parent_state["run_config"],
            child_policy,
            request,
        )
        child_service = AgentService(
            policy=child_policy,
            tool_registry=self._tool_registry,
            model_turn_provider=self._model_turn_provider,
            retrieval_hint_provider=self._retrieval_hint_provider,
        )
        return await child_service.run_with_config(
            task=request.prompt,
            run_config=child_config,
        )

    def _child_policy(self) -> AgentRuntimePolicy:
        return replace(
            self._policy,
            max_depth=max(self._policy.max_depth - 1, 0),
            core_tool_names=tuple(tool for tool in self._policy.core_tool_names if tool != "task"),
            deferred_tool_names=tuple(tool for tool in self._policy.deferred_tool_names if tool != "task"),
        )

    def _derive_child_config(
        self,
        parent_config: AgentRunConfig,
        child_policy: AgentRuntimePolicy,
        request: AgentDelegationRequest,
    ) -> AgentRunConfig:
        child_definition = AgentRuntimePolicy(
            agent_type="task_child",
            description="Generic task child",
            system_instructions=child_policy.system_instructions,
            core_tool_names=child_policy.core_tool_names,
            deferred_tool_names=child_policy.deferred_tool_names,
            max_iterations=child_policy.max_iterations,
            max_depth=child_policy.max_depth,
            model_selection=child_policy.model_selection,
            tool_policy=child_policy.tool_policy,
        )
        child_config = derive_child_config(parent_config, child_definition)
        if request.llm_budget_total is not None:
            child_config = replace(
                child_config,
                llm_budget_total=request.llm_budget_total,
            )
        if request.max_turns is not None:
            child_config = replace(
                child_config,
                max_turns=request.max_turns,
            )
        return child_config


class AgentService:
    def __init__(
        self,
        *,
        definition: AgentRuntimePolicy | None = None,
        policy: AgentRuntimePolicy | None = None,
        tool_registry: ToolRegistry,
        model_turn_provider: ModelTurnProvider | None = None,
        retrieval_hint_provider: RetrievalHintProvider | None = None,
        subagent_runner: DelegatedAgentRunner | None = None,
        output_finalizer: StructuredOutputFinalizer | None = None,
        model_registry: ModelResolver | None = None,
        checkpointer: BaseCheckpointSaver[str] | None = None,
        runtime_diagnostics: Sequence[RuntimeDiagnostic] = (),
        catalog: ToolCatalog | None = None,
        stream_sink: Any = None,  # StreamEventSink | None
        mcp_registry: MCPToolRegistry | None = None,
        skill_catalog: Any = None,  # SkillCatalog | None
        strict_model_provider: bool = True,
        latency_profile: AgentLatencyProfile | None = None,
    ) -> None:
        if definition is not None and policy is not None:
            raise ValueError("Provide either 'definition' or 'policy', not both")
        if definition is not None:
            self._policy = definition
        elif policy is not None:
            self._policy = policy
        else:
            raise ValueError("Provide either 'definition' or 'policy'")
        self._base_tool_registry = tool_registry
        self._catalog = catalog or build_tool_catalog(tool_registry, self._policy)
        self._skill_catalog = skill_catalog  # SkillCatalog | None
        self._skill_runtime = None
        if skill_catalog is not None:
            from rag.agent.skills.runtime import SkillRuntime

            self._skill_runtime = SkillRuntime(skill_catalog)
            self._register_skill_tool()
        self._model_turn_provider = model_turn_provider
        self._retrieval_hint_provider = retrieval_hint_provider
        self._subagent_runner = subagent_runner
        self._output_finalizer = output_finalizer
        self._model_registry = model_registry
        self._strict_model_provider = strict_model_provider
        self._runtime_diagnostics = tuple(merge_runtime_diagnostics([], runtime_diagnostics))
        self._checkpointer = checkpointer or create_agent_checkpointer(None)
        self._stream_sink = stream_sink
        self._mcp_registry = mcp_registry
        self._latency_profile = latency_profile or AgentLatencyProfile()
        self._persistent_memory_runtime = PersistentMemoryRuntime(
            model_registry=model_registry,
        )
        # Register core tools: tool_search, activate_tools, task
        self._register_discovery_tools()
        self._register_task_tool()

    async def aclose(self) -> None:
        await aclose_agent_checkpointer(self._checkpointer)

    def initial_state(self, request: AgentRunRequest) -> LoopState:
        run_config = request.to_run_config(self._policy)
        return self.initial_state_from_config(
            task=request.task,
            run_config=run_config,
            pending_tool_calls=request.pending_tool_calls,
            approved_tool_call_ids=request.approved_tool_call_ids,
            denied_tool_call_ids=request.denied_tool_call_ids,
            messages=request.messages,
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
        RunRegistry.remove(run_config.run_id)
        handles = RunRegistry.get_or_create(run_config)
        if memory_store is not None:
            handles.memory_store = memory_store
        state = create_loop_state(
            task=task,
            run_config=run_config,
            pending_tool_calls=pending_tool_calls or (),
            messages=messages or (),
            runtime_diagnostics=self._runtime_diagnostics,
        )
        state["approved_tool_call_ids"] = list(approved_tool_call_ids or ())
        state["denied_tool_call_ids"] = list(denied_tool_call_ids or ())
        compacted = cast(
            LoopState,
            MessageCompactor(
                policy=run_config.memory_policy,
                store=memory_store,
            ).compact_initial_state(dict(state)),
        )
        compacted["latency_profile"] = self._latency_profile
        return compacted

    async def run(self, request: AgentRunRequest) -> AgentRunResult:
        run_config = request.to_run_config(self._policy)
        return await self.run_with_config(
            task=request.task,
            run_config=run_config,
            pending_tool_calls=request.pending_tool_calls,
            approved_tool_call_ids=request.approved_tool_call_ids,
            denied_tool_call_ids=request.denied_tool_call_ids,
            messages=request.messages,
            goal_spec=request.goal_spec,
            input_files=request.input_files,
            workspace_path=request.workspace_path,
            tool_surface_request=request.tool_surface_request,
            tool_discovery_state=request.tool_discovery_state,
        )

    def _prepare_run(
        self,
        *,
        task: str,
        run_config: AgentRunConfig,
        input_files: list[str] | None = None,
        workspace_path: str | None = None,
    ) -> _RunContext:
        """Single entry point for per-run setup.  One path, no duplication."""
        from rag.agent.file_manifest import build_file_manifest
        from rag.agent.workspace import create_temp_workspace, import_files, open_workspace

        if workspace_path:
            workspace = open_workspace(workspace_path, create=True)
        else:
            workspace = create_temp_workspace()

        if input_files:
            import_files(workspace, [Path(f) for f in input_files])

        manifest = build_file_manifest(workspace)
        workspace_tools = create_workspace_tools(workspace)
        tooling_registry = NewToolRegistry()
        install_minimal_workspace_tools(
            tooling_registry,
            workspace,
            allowed_commands={"echo"},
        )
        memory_store = WorkspaceMemoryStore(
            workspace=workspace,
            policy=run_config.memory_policy,
        )

        file_tools_to_activate: set[str] = {"structured_probe"} if manifest.has_probeable_files else set()
        runtime_registry = self._runtime_tool_registry(
            run_config,
            tools=workspace_tools,
        )
        self._validate_allowed_tools(runtime_registry)
        self._validate_workspace_core_runners(
            runtime_registry,
            allowed_tool_names=self._policy.allowed_tools,
        )

        enriched_task = self._inject_manifest_into_task(task, manifest)

        return _RunContext(
            task=enriched_task,
            workspace=workspace,
            manifest=manifest,
            workspace_tools=workspace_tools,
            file_tools_to_activate=file_tools_to_activate,
            runtime_registry=runtime_registry,
            tooling_registry=tooling_registry,
            memory_store=memory_store,
            persistent_store=PersistentMemoryStore(workspace),
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
        tool_surface_request: ToolSurfaceRequest | None = None,
        tool_discovery_state: ToolDiscoveryState | None = None,
    ) -> AgentRunResult:
        run_started_at = time.perf_counter()
        prepare_started_at = run_started_at
        ctx = self._prepare_run(
            task=task,
            run_config=run_config,
            input_files=input_files,
            workspace_path=workspace_path,
        )
        prepare_latency_ms = _elapsed_ms(prepare_started_at)

        state = self._initial_loop_state_from_config(
            task=ctx.task,
            run_config=run_config,
            pending_tool_calls=pending_tool_calls,
            approved_tool_call_ids=approved_tool_call_ids,
            denied_tool_call_ids=denied_tool_call_ids,
            messages=messages,
            memory_store=ctx.memory_store,
            file_manifest=ctx.manifest,
        )
        await self._persistent_memory_runtime.load(
            state,
            ctx.persistent_store,
            task=ctx.task,
        )
        has_pending_tool_calls = bool(pending_tool_calls)
        uses_service_model_provider = self._model_turn_provider is None
        effective_tool_surface_request = (
            tool_surface_request if uses_service_model_provider else None
        )
        effective_tool_discovery_state = (
            tool_discovery_state if uses_service_model_provider else None
        )
        if (
            effective_tool_surface_request is None
            and uses_service_model_provider
            and not has_pending_tool_calls
        ):
            effective_tool_surface_request = _default_tool_surface_request()

        checkpoint_store = LangGraphCheckpointStore(
            self._checkpointer,
            run_config=run_config,
            compatibility_config=GoalCompatibilityConfig(goal_spec=goal_spec),
        )
        try:
            loop = self._build_loop(
                runtime_registry=ctx.runtime_registry,
                checkpoint_store=checkpoint_store,
                memory_store=ctx.memory_store,
                goal_spec=goal_spec,
                state=state,
                auto_activate_tools=ctx.file_tools_to_activate,
                scratch_dir=ctx.workspace.root / "scratch",
                tooling_registry=ctx.tooling_registry,
                tool_surface_request=effective_tool_surface_request,
                tool_discovery_state=effective_tool_discovery_state,
            )
            result_state = await loop.run(state)
        except Exception:
            RunRegistry.remove(run_config.run_id)
            raise
        if result_state["status"] in {"completed", "failed"}:
            RunRegistry.remove(run_config.run_id)

        finalize_started_at = time.perf_counter()
        if result_state["status"] == "completed":
            try:
                await self._persistent_memory_runtime.extract(
                    result_state,
                    ctx.persistent_store,
                )
            except Exception:
                logger.warning("Persistent memory extraction failed", exc_info=True)
        finalize_latency_ms = _elapsed_ms(finalize_started_at)
        self._record_result_latency_profile(
            result_state,
            run_started_at=run_started_at,
            prepare_latency_ms=prepare_latency_ms,
            finalize_latency_ms=finalize_latency_ms,
        )

        return AgentRunResult.from_loop_result(
            result_state,
            definition=self._policy,
            workspace_path=str(ctx.workspace.root),
        )

    def _record_result_latency_profile(
        self,
        result_state: LoopState,
        *,
        run_started_at: float,
        prepare_latency_ms: float,
        finalize_latency_ms: float,
    ) -> None:
        result_state["latency_profile"] = _profile_with_run_timings(
            result_state["latency_profile"],
            prepare_latency_ms=prepare_latency_ms,
            tool_latency_ms=sum(result.latency_ms for result in result_state["tool_results"]),
            finalize_latency_ms=finalize_latency_ms,
            total_ms=self._latency_profile.startup_ms
            + self._latency_profile.build_service_ms
            + _elapsed_ms(run_started_at),
        )

    async def run_streaming(self, request: AgentRunRequest) -> AsyncGenerator[Any, None]:
        """流式运行 Agent，yield 每一个 StreamEvent。

        用法：
            async for event in service.run_streaming(request):
                handle(event)
        """

        run_config = request.to_run_config(self._policy)
        ctx = self._prepare_run(
            task=request.task,
            run_config=run_config,
            input_files=request.input_files,
            workspace_path=request.workspace_path,
        )

        state = self._initial_loop_state_from_config(
            task=ctx.task,
            run_config=run_config,
            pending_tool_calls=request.pending_tool_calls,
            approved_tool_call_ids=request.approved_tool_call_ids,
            denied_tool_call_ids=request.denied_tool_call_ids,
            messages=request.messages,
            memory_store=ctx.memory_store,
            file_manifest=ctx.manifest,
        )
        await self._persistent_memory_runtime.load(
            state,
            ctx.persistent_store,
            task=ctx.task,
        )
        uses_service_model_provider = self._model_turn_provider is None
        effective_tool_surface_request = (
            request.tool_surface_request if uses_service_model_provider else None
        )
        effective_tool_discovery_state = (
            request.tool_discovery_state if uses_service_model_provider else None
        )
        if (
            effective_tool_surface_request is None
            and uses_service_model_provider
            and not request.pending_tool_calls
        ):
            effective_tool_surface_request = _default_tool_surface_request()

        checkpoint_store = LangGraphCheckpointStore(
            self._checkpointer,
            run_config=run_config,
            compatibility_config=GoalCompatibilityConfig(goal_spec=request.goal_spec),
        )
        loop = self._build_loop(
            runtime_registry=ctx.runtime_registry,
            checkpoint_store=checkpoint_store,
            memory_store=ctx.memory_store,
            goal_spec=request.goal_spec,
            state=state,
            auto_activate_tools=ctx.file_tools_to_activate,
            scratch_dir=ctx.workspace.root / "scratch",
            tooling_registry=ctx.tooling_registry,
            tool_surface_request=effective_tool_surface_request,
            tool_discovery_state=effective_tool_discovery_state,
        )

        try:
            async for event in loop.run_streaming(state):
                yield event
        finally:
            if state["status"] in {"completed", "failed"}:
                RunRegistry.remove(run_config.run_id)
            if state["status"] == "completed":
                try:
                    await self._persistent_memory_runtime.extract(
                        state,
                        ctx.persistent_store,
                    )
                except Exception:
                    logger.warning("Persistent memory extraction failed", exc_info=True)

    def _initial_loop_state_from_config(
        self,
        *,
        task: str,
        run_config: AgentRunConfig,
        pending_tool_calls: list[ToolCallPlan] | None = None,
        approved_tool_call_ids: list[str] | None = None,
        denied_tool_call_ids: list[str] | None = None,
        messages: list[BaseMessage] | None = None,
        memory_store: WorkspaceMemoryStore | None = None,
        file_manifest: FileManifest | None = None,
    ) -> LoopState:
        RunRegistry.remove(run_config.run_id)
        handles = RunRegistry.get_or_create(run_config)
        if memory_store is not None:
            handles.memory_store = memory_store
        state = create_loop_state(
            task=task,
            run_config=run_config,
            pending_tool_calls=pending_tool_calls or (),
            messages=messages or (),
            runtime_diagnostics=self._runtime_diagnostics,
            file_manifest=file_manifest,
        )
        state["latency_profile"] = self._latency_profile
        state["approved_tool_call_ids"] = list(approved_tool_call_ids or ())
        state["denied_tool_call_ids"] = list(denied_tool_call_ids or ())
        return state

    def _build_loop(
        self,
        *,
        runtime_registry: ToolRegistry,
        checkpoint_store: LangGraphCheckpointStore,
        memory_store: WorkspaceMemoryStore | None,
        goal_spec: GoalSpec | None,
        state: LoopState | None = None,
        auto_activate_tools: set[str] | None = None,
        scratch_dir: Path | None = None,
        tooling_registry: NewToolRegistry | None = None,
        tool_surface_request: ToolSurfaceRequest | None = None,
        tool_discovery_state: ToolDiscoveryState | None = None,
    ) -> AgentLoop:
        # Create and bind DeferredToolStore BEFORE resolving provider
        # (provider needs store for tool filtering)
        store = DeferredToolStore(
            max_active=self._policy.max_active_deferred_tools,
        )
        deferred_store_var.set(store)
        if state is not None:
            store.sync_from_state(cast(dict[Any, Any], state))

        # Auto-activate file-related tools (skip tool_search + activate_tools)
        if auto_activate_tools:
            for tool_name in auto_activate_tools:
                try:
                    store.set_pending_candidates(
                        query="__file_manifest_auto__",
                        candidates=[
                            SearchCandidate(
                                name=tool_name,
                                description="Auto-activated for file processing",
                                reason="structured input files detected",
                            )
                        ],
                    )
                    store.activate(tool_name, iteration=0, source_query="__file_manifest_auto__")
                except (KeyError, RuntimeError):
                    pass  # Already active or not in allowed_tools
        provider = self._resolve_model_turn_provider(
            state,
            tool_registry=runtime_registry,
            tooling_registry=tooling_registry,
            tool_surface_request=tool_surface_request,
            tool_discovery_state=tool_discovery_state,
        )
        output_finalizer = self._resolve_output_finalizer(state)
        tool_runner = (
            ToolExecutorLoopAdapter(
                NewToolExecutor(
                    tooling_registry,
                    allow_write_tools=tool_surface_request.allow_write_tools,
                    allow_execute_tools=tool_surface_request.allow_execute_tools,
                )
            )
            if tooling_registry is not None and tool_surface_request is not None
            else ToolExecutionService(
                tool_registry=runtime_registry,
                record_writer=checkpoint_store,
                stream_sink=self._stream_sink,
            )
        )
        return AgentLoop(
            definition=self._policy,
            model_provider=provider,
            context_manager=LoopContextCompactor(store=memory_store),
            tool_runner=tool_runner,
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
            catalog=self._catalog,
            deferred_store=store,
            stream_sink=self._stream_sink,
            scratch_dir=scratch_dir,
        )

    def _resolve_model_turn_provider(
        self,
        state: LoopState | None,
        *,
        tool_registry: ToolRegistry | None = None,
        tooling_registry: NewToolRegistry | None = None,
        tool_surface_request: ToolSurfaceRequest | None = None,
        tool_discovery_state: ToolDiscoveryState | None = None,
    ) -> ModelTurnProvider:
        return ModelProviderResolver(
            model_turn_provider=self._model_turn_provider,
            model_registry=self._model_registry,
            policy=self._policy,
            base_tool_registry=self._base_tool_registry,
            catalog=self._catalog,
            strict_model_provider=self._strict_model_provider,
            stream_sink=self._stream_sink,
            skill_context_provider=(
                self._skill_runtime.render_prompt_context
                if self._skill_runtime is not None
                else None
            ),
        ).resolve(
            state,
            tool_registry=tool_registry,
            tooling_registry=tooling_registry,
            tool_surface_request=tool_surface_request,
            tool_discovery_state=tool_discovery_state,
        )

    def _resolve_output_finalizer(
        self,
        state: LoopState | None,
    ) -> StructuredOutputFinalizer | None:
        if self._output_finalizer is not None:
            return self._output_finalizer
        if self._policy.output_model is None or self._model_registry is None:
            return None
        try:
            return create_model_structured_output_finalizer(self._model_registry)
        except Exception as exc:
            if state is not None:
                append_loop_diagnostic(
                    state,
                    RuntimeDiagnostic.from_exception(
                        code=("structured_output_finalizer_initialization_failed"),
                        component="structured_output_finalizer",
                        error=exc,
                    ),
                )
            return None

    def _validate_allowed_tools(self, registry: ToolRegistry) -> None:
        registered = {tool.name for tool in registry.list_all()}
        missing = [name for name in self._policy.allowed_tools if name not in registered]
        if missing:
            raise ValueError(f"unregistered tools: {', '.join(dict.fromkeys(missing))}")

    @staticmethod
    def _validate_workspace_core_runners(
        registry: ToolRegistry,
        *,
        allowed_tool_names: Sequence[str] | None = None,
    ) -> None:
        """Verify workspace-dependent core tools have runners registered.

        These tools (search_text, apply_patch, run_command, etc.) depend on
        create_workspace_tools() for their runners.  If someone changes the
        workspace tool creation, this assertion catches the regression before
        a model call fails silently at runtime.
        """
        workspace_core_tools = frozenset({
            "list_files", "read_file", "write_file", "run_python",
            "search_text", "apply_patch", "run_command",
        })
        required_tools = workspace_core_tools
        if allowed_tool_names is not None:
            required_tools = workspace_core_tools & frozenset(allowed_tool_names)
        missing_runners = sorted(
            name for name in required_tools
            if not registry.has_runner(name)
        )
        if missing_runners:
            raise RuntimeError(
                f"Workspace core tools missing runners: {', '.join(missing_runners)}. "
                f"Ensure create_workspace_tools() includes all workspace-dependent core tools."
            )

    @staticmethod
    def _inject_manifest_into_task(task: str, manifest: FileManifest) -> str:
        """Inject file manifest + processing instructions before the user task.

        Only activates when input files are present.  Injects both the manifest
        and file processing instructions so the model knows: probes already ran,
        go straight to run_python, cite everything.
        """
        if not manifest.files:
            return task

        block = manifest.to_context_block()
        if not block:
            return task

        instructions = """\
Input files detected — you are in file processing mode:

- File manifest and structured_probe results are shown above.
  Do NOT call list_files or structured_probe again for these files.
- Use run_python with pandas for computation. Never guess column
  names, sheet names, or data values.
- Cite: file path, sheet/table name, columns, row count, method.
- For numerical answers, cross-validate (e.g. groupby-sum vs raw-sum).
- If the manifest shows ambiguity, report it before computing.
- For charts, use matplotlib; plt.savefig() to scratch/."""

        return f"{block}\n\n{instructions}\n\n── User Task ──\n{task}"

    def _runtime_tool_registry(
        self,
        run_config: AgentRunConfig,
        *,
        runners: Mapping[str, ToolRunner] | None = None,
        tools: list[Any] | None = None,  # list[BaseTool] instances
    ) -> ToolRegistry:
        """Clone base registry and inject per-request tool instances.

        Agent-as-tool adapters are request-scoped — each run_config gets fresh adapters
        to prevent concurrent request pollution of depth/budget/access_policy.
        """
        return RuntimeToolRegistryBuilder(
            base_tool_registry=self._base_tool_registry,
            policy=self._policy,
            catalog=self._catalog,
            model_registry=self._model_registry,
            model_turn_provider=self._model_turn_provider,
            retrieval_hint_provider=self._retrieval_hint_provider,
            task_delegated_runner_factory=(
                lambda runtime: _TaskChildRunner(
                    policy=self._policy,
                    tool_registry=runtime,
                    model_turn_provider=self._model_turn_provider,
                    retrieval_hint_provider=self._retrieval_hint_provider,
                )
            ),
            subagent_runner=self._subagent_runner,
            mcp_registry=self._mcp_registry,
        ).build(
            run_config,
            runners=runners,
            tools=tools,
        )

    def _register_discovery_tools(self) -> None:
        """Register tool_search and activate_tools as core tools.

        Runners read the per-run DeferredToolStore from a ContextVar,
        ensuring concurrent runs each get their own store.
        """
        if self._base_tool_registry.has_runner("tool_search"):
            return

        catalog = self._catalog
        definition = self._policy

        # ── tool_search ──
        search_spec = ToolSpec(
            name="tool_search",
            description=(
                "Discover available tools by natural language query. "
                "Returns candidate tools. Call activate_tools to load them. "
                "Do not use for simple questions, greetings, or literal reply "
                "requests; answer those directly."
            ),
            input_model=ToolSearchInput,
            output_model=ToolSearchOutput,
            error_model=ToolSearchOutput,
            permissions=ToolPermissions(),
            execution_category=ExecutionCategory.READ,
            timeout_seconds=5,
            idempotent=True,
            concurrency_safe=True,
        )

        def _search_runner(input_data: ToolSearchInput) -> ToolSearchOutput:
            store = deferred_store_var.get(None)
            if store is None:
                raise RuntimeError(
                    "DeferredToolStore is not bound — AgentLoop must set deferred_store_var before tool execution"
                )
            return execute_tool_search(
                query=input_data.query,
                catalog=catalog,
                store=store,
                max_results=input_data.max_results,
            )

        self._base_tool_registry.register(
            search_spec,
            runner=cast(ToolRunner, _search_runner),
        )

        # ── activate_tools ──
        activate_spec = ToolSpec(
            name="activate_tools",
            description=(
                "Activate tools found by tool_search. "
                "Only tools from the most recent tool_search can be activated. "
                "Activated tools become available on the next model turn."
            ),
            input_model=ActivateToolsInput,
            output_model=ActivateToolsOutput,
            error_model=ActivateToolsOutput,
            permissions=ToolPermissions(),
            execution_category=ExecutionCategory.TRANSFORM,
            timeout_seconds=5,
            idempotent=True,
            concurrency_safe=True,
        )

        def _activate_runner(input_data: ActivateToolsInput) -> ActivateToolsOutput:
            store = deferred_store_var.get(None)
            if store is None:
                raise RuntimeError(
                    "DeferredToolStore is not bound — AgentLoop must set deferred_store_var before tool execution"
                )
            return execute_activate_tools(
                names=input_data.names,
                catalog=catalog,
                store=store,
                allowed_tools=definition.allowed_tools,  # mutable list, extended by _inject_mcp_tools
                deny_tools=definition.tool_policy.deny_tools,
                iteration=iteration_var.get(0),
                group=input_data.group,
            )

        self._base_tool_registry.register(
            activate_spec,
            runner=cast(ToolRunner, _activate_runner),
        )

    def _register_task_tool(self) -> None:
        """Register the 'task' tool spec (runner injected per-request in _runtime_tool_registry)."""
        if self._base_tool_registry.has_runner("task"):
            return

        from rag.agent.tools.task_tool import task_tool_spec

        # Register spec only — runner is request-scoped (needs parent run_config)
        self._base_tool_registry.register(task_tool_spec)

    def _register_skill_tool(self) -> None:
        """Register the 'invoke_skill' tool spec with a catalog-bound runner."""
        from rag.agent.skills.invocation import INVOKE_SKILL_SPEC, make_invoke_skill_runner

        # Register spec and a permanent runner (skill catalog is service-scoped)
        _skill_runner = make_invoke_skill_runner(self._skill_catalog)
        self._base_tool_registry.register(INVOKE_SKILL_SPEC)
        self._base_tool_registry.register_contextual_runner("invoke_skill", _skill_runner)

        # Dynamically add skill tools to the policy's core tools so they
        # appears in the model's tool list ONLY when skills are available.
        skill_core_tools = ("invoke_skill", "materialize_skill_asset")
        missing = tuple(
            name for name in skill_core_tools
            if name not in self._policy.core_tool_names
        )
        if missing:
            self._policy = replace(
                self._policy,
                core_tool_names=self._policy.core_tool_names + missing,
            )

    async def resume(
        self,
        *,
        run_id: str,
        response: HumanInputResponse,
        workspace_path: str | None = None,
    ) -> AgentRunResult:
        """从中断点恢复。response 对应 HumanInputRequest 的用户响应。

        如果原始 run 使用了 PrimitiveOps（write_file / run_python 等），
        调用方必须传入 workspace_path 以恢复 request-scoped runners。
        """
        from rag.agent.workspace import open_workspace

        resume_started_at = time.perf_counter()
        prepare_started_at = resume_started_at
        lookup_config = self._checkpoint_lookup_config(run_id)
        checkpoint_store = LangGraphCheckpointStore(
            self._checkpointer,
            run_config=lookup_config,
        )
        restored = await checkpoint_store.load_for_resume()
        if restored is None:
            raise KeyError(f"No checkpoint found for run_id={run_id}")
        state = await checkpoint_store.apply_human_response(response)
        run_config = state["run_config"]
        await self._restore_runtime_handles_from_checkpoint(run_config)

        workspace_tools: list[Any] | None = None
        memory_store: WorkspaceMemoryStore | None = None
        if workspace_path:
            workspace = open_workspace(workspace_path)
            workspace_tools = create_workspace_tools(workspace)
            memory_store = WorkspaceMemoryStore(
                workspace=workspace,
                policy=run_config.memory_policy,
            )
            RunRegistry.get(run_config.run_id).memory_store = memory_store

        runtime_registry = self._runtime_tool_registry(
            run_config,
            tools=workspace_tools,
        )
        self._validate_allowed_tools(runtime_registry)
        self._validate_workspace_core_runners(
            runtime_registry,
            allowed_tool_names=self._policy.allowed_tools,
        )
        loop = self._build_loop(
            runtime_registry=runtime_registry,
            checkpoint_store=checkpoint_store,
            memory_store=memory_store,
            goal_spec=checkpoint_store.compatibility_config.goal_spec,
            state=state,
        )
        prepare_latency_ms = _elapsed_ms(prepare_started_at)
        result_state = await loop.run(state)
        if result_state["status"] in {"completed", "failed"}:
            RunRegistry.remove(run_config.run_id)
        self._record_result_latency_profile(
            result_state,
            run_started_at=resume_started_at,
            prepare_latency_ms=prepare_latency_ms,
            finalize_latency_ms=0.0,
        )
        return AgentRunResult.from_loop_result(
            result_state,
            definition=self._policy,
            workspace_path=workspace_path,
        )

    def pending_human_input_request(self, *, run_id: str) -> HumanInputRequest:
        state = LangGraphCheckpointStore(
            self._checkpointer,
            run_config=self._checkpoint_lookup_config(run_id),
        ).load_latest_sync()
        if state is None:
            raise KeyError(f"No checkpoint found for run_id={run_id}")
        request = state["approval_request"]
        if request is None and state["pause"] is not None:
            request = state["pause"].request
        if request is None:
            raise KeyError(f"No pending human input request for run_id={run_id}")
        return request

    async def apending_human_input_request(self, *, run_id: str) -> HumanInputRequest:
        state = await LangGraphCheckpointStore(
            self._checkpointer,
            run_config=self._checkpoint_lookup_config(run_id),
        ).load_for_resume()
        if state is None:
            raise KeyError(f"No checkpoint found for run_id={run_id}")
        request = state["approval_request"]
        if request is None and state["pause"] is not None:
            request = state["pause"].request
        if request is None:
            raise KeyError(f"No pending human input request for run_id={run_id}")
        return request

    async def _restore_runtime_handles_from_checkpoint(
        self,
        run_config: AgentRunConfig,
    ) -> AgentRunConfig:
        try:
            RunRegistry.get(run_config.run_id)
            return run_config
        except KeyError:
            RunRegistry.get_or_create(run_config)

        return run_config

    def _checkpoint_lookup_config(self, run_id: str) -> AgentRunConfig:
        return AgentRunConfig(
            run_id=run_id,
            thread_id=run_id,
            agent_type=self._policy.agent_type,
            max_depth=self._policy.max_depth,
            access_policy=(self._policy.access_policy_ceiling or AccessPolicy.default()),
            tool_policy=self._policy.tool_policy,
        )


def _derive_groundedness(tool_results: list[ToolResult]) -> bool:
    """Derive groundedness_flag from the last RAG generation ToolResult.output."""
    for result in reversed(tool_results):
        if result.status == "ok" and result.output is not None:
            if bool(getattr(result.output, "groundedness_flag", False)):
                return True
    return False


def _derive_insufficient_evidence(tool_results: list[ToolResult]) -> bool:
    """Derive insufficient_evidence_flag from the last RAG generation ToolResult.output."""
    for result in reversed(tool_results):
        if result.status == "ok" and result.output is not None:
            if bool(getattr(result.output, "insufficient_evidence", False)) or bool(
                getattr(result.output, "insufficient_evidence_flag", False)
            ):
                return True
    return False


def _restore_final_output(
    raw_output: ValidatedFinalOutput | dict[str, object] | None,
    *,
    definition: AgentRuntimePolicy | None,
) -> BaseModel | None:
    if raw_output is None or definition is None or definition.output_model is None:
        return None
    envelope = ValidatedFinalOutput.model_validate(raw_output)
    expected_path = output_model_path(definition.output_model)
    if envelope.model_path != expected_path:
        raise ValueError("Checkpoint final output model does not match AgentRuntimePolicy.output_model")
    return definition.output_model.model_validate(envelope.data)


__all__ = [
    "AgentRunRequest",
    "AgentRunResult",
    "AgentService",
]

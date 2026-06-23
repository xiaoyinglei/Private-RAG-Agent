from __future__ import annotations

import logging
from collections.abc import AsyncGenerator, Mapping, Sequence
from dataclasses import replace
from inspect import isawaitable
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast
from uuid import uuid4

from langchain_core.messages import BaseMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from pydantic import BaseModel, ConfigDict, Field

from rag.agent.capabilities.catalog import (
    CORE_TOOLS,
    DEFERRED_TOOLS,
    DeferredToolStore,
    SearchCandidate,
    ToolCatalog,
    ToolCatalogEntry,
    flatten_schema,
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
from rag.agent.compat.goal_contract import GoalCompatibilityConfig, GoalSpec
from rag.agent.core.checkpointing import (
    LangGraphCheckpointStore,
    create_agent_checkpointer,
)
from rag.agent.core.context import AgentRunConfig, RunRegistry, derive_child_config
from rag.agent.core.definition import AgentDefinition, AgentRuntimePolicy
from rag.agent.core.delegation import AgentDelegationRequest, DelegatedAgentRunner, ParentAgentContext
from rag.agent.core.finalization import FinishCandidateBuilder
from rag.agent.core.human_input import HumanInputRequest, HumanInputResponse
from rag.agent.core.llm_providers import create_loop_model_turn_provider
from rag.agent.core.llm_registry import ModelRegistry
from rag.agent.core.output_finalizer import (
    StructuredOutputFinalizer,
    create_model_structured_output_finalizer,
)
from rag.agent.core.output_models import (
    ValidatedFinalOutput,
    output_model_path,
)
from rag.agent.core.runtime_diagnostics import (
    RuntimeDiagnostic,
    merge_runtime_diagnostics,
)
from rag.agent.core.runtime_ports import RetrievalHintProvider
from rag.agent.core.tool_execution import ToolExecutionService
from rag.agent.core.turn_contracts import ToolCallPlan
from rag.agent.loop.runtime import AgentLoop, ModelTurnProvider
from rag.agent.loop.state import (
    LoopState,
    ModelTurnDraft,
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
from rag.agent.memory.store import WorkspaceMemoryStore
from rag.agent.tools.registry import ToolRegistry, ToolRunner
from rag.agent.tools.spec import ExecutionCategory, ToolPermissions, ToolResult, ToolSpec
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
    budget_total: int | None = Field(default=None, gt=0)
    max_context_tokens: int | None = Field(default=None, gt=0)
    max_depth: int | None = Field(default=None, ge=0)
    pending_tool_calls: list[ToolCallPlan] = Field(default_factory=list)
    approved_tool_call_ids: list[str] = Field(default_factory=list)
    denied_tool_call_ids: list[str] = Field(default_factory=list)
    messages: list[BaseMessage] = Field(default_factory=list)
    input_files: list[str] = Field(default_factory=list)
    workspace_path: str | None = None
    memory_policy: MemoryPolicy | None = None
    goal_spec: GoalSpec | None = None

    def to_run_config(self, definition: AgentDefinition) -> AgentRunConfig:
        run_id = self.run_id or f"run_{uuid4().hex[:12]}"
        return AgentRunConfig(
            run_id=run_id,
            thread_id=self.thread_id or run_id,
            budget_total=self.budget_total or definition.estimated_token_budget,
            work_budget_total=definition.estimated_work_budget,
            agent_type=definition.agent_type,
            max_context_tokens=self.max_context_tokens,
            max_depth=definition.max_depth if self.max_depth is None else self.max_depth,
            access_policy=definition.access_policy or AccessPolicy.default(),
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

    @classmethod
    def from_state(
        cls,
        state: LoopState,
        *,
        definition: AgentDefinition | None = None,
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
        definition: AgentDefinition | None = None,
        workspace_path: str | None = None,
    ) -> AgentRunResult:
        run_config = state["run_config"]
        is_terminal = state["status"] in {"completed", "failed"}
        terminal = state["terminal"]
        pause = state["pause"]
        pending = state["pending_tool_calls"]
        return cls(
            run_id=run_config.run_id,
            thread_id=run_config.thread_id,
            status=(
                "done"
                if state["status"] == "completed"
                else state["status"]
            ),
            final_answer=state["final_answer"],
            final_output=_restore_final_output(
                state["final_output"],
                definition=definition,
            ),
            output_validation_errors=list(
                state["output_validation_errors"]
            ),
            stop_reason=(
                None
                if terminal is None
                else terminal.stop_reason
            ),
            tool_results=list(state["tool_results"]),
            evidence=list(state["evidence"]),
            citations=list(state["citations"]),
            iteration=state["iteration"],
            groundedness_flag=state["groundedness_flag"],
            insufficient_evidence_flag=(
                state["insufficient_evidence_flag"]
            ),
            needs_user_input=(
                None
                if is_terminal or pause is None
                else pause.reason
            ),
            human_input_request=(
                None
                if is_terminal
                else state["approval_request"]
            ),
            pending_tool_calls_summary=[
                {
                    "tool_call_id": call.tool_call_id,
                    "tool_name": call.tool_name,
                }
                for call in pending
            ],
            workspace_path=workspace_path,
            runtime_diagnostics=list(state["runtime_diagnostics"]),
        )


class _ResultDrivenModelTurnProvider:
    async def next_turn(
        self,
        state: LoopState,
        *,
        definition: AgentDefinition,
        budget_remaining: int,
    ) -> ModelTurnDraft:
        del definition, budget_remaining
        if state["stop_hook_feedback"]:
            return ModelTurnDraft(
                action="pause",
                pause_reason=(
                    "No model turn provider is available to address "
                    "stop-hook feedback."
                ),
            )
        if state["answer_candidates"]:
            return ModelTurnDraft(
                action="finish",
                final_answer=state["answer_candidates"][-1].text,
            )
        if state["tool_results"]:
            latest = state["tool_results"][-1]
            if latest.error is not None:
                return ModelTurnDraft(action="finish")
            reason = (
                "Tool execution produced no answer candidate."
            )
            return ModelTurnDraft(
                action="pause",
                pause_reason=reason,
            )
        return ModelTurnDraft(
            action="pause",
            pause_reason="No model turn provider is configured.",
        )


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
            core_tool_names=tuple(
                tool
                for tool in self._policy.core_tool_names
                if tool != "task"
            ),
        )

    def _derive_child_config(
        self,
        parent_config: AgentRunConfig,
        child_policy: AgentRuntimePolicy,
        request: AgentDelegationRequest,
    ) -> AgentRunConfig:
        child_definition = AgentDefinition(
            agent_type="task_child",
            description="Generic task child",
            system_prompt=child_policy.system_instructions,
            allowed_tools=child_policy.allowed_tools,
            estimated_token_budget=(
                request.estimated_tokens or child_policy.token_budget
            ),
            estimated_work_budget=child_policy.work_budget,
            max_iterations=child_policy.max_iterations,
            max_depth=child_policy.max_depth,
            model_selection=child_policy.model_selection,
            tool_policy=child_policy.tool_policy,
        )
        child_config = derive_child_config(parent_config, child_definition)
        if request.estimated_tokens is not None:
            child_config = replace(
                child_config,
                budget_total=request.estimated_tokens,
            )
        return child_config


class AgentService:
    def __init__(
        self,
        *,
        definition: AgentDefinition | None = None,
        policy: AgentRuntimePolicy | None = None,
        tool_registry: ToolRegistry,
        model_turn_provider: ModelTurnProvider | None = None,
        retrieval_hint_provider: RetrievalHintProvider | None = None,
        subagent_runner: DelegatedAgentRunner | None = None,
        output_finalizer: StructuredOutputFinalizer | None = None,
        model_registry: ModelRegistry | None = None,
        checkpointer: BaseCheckpointSaver[str] | None = None,
        runtime_diagnostics: Sequence[RuntimeDiagnostic] = (),
        catalog: ToolCatalog | None = None,
        stream_sink: Any = None,  # StreamEventSink | None
    ) -> None:
        if definition is not None and policy is not None:
            raise ValueError("Provide either 'definition' or 'policy', not both")
        if policy is not None:
            # Convert AgentRuntimePolicy → AgentDefinition for internal compat.
            # Remove this when service internals migrate to AgentRuntimePolicy.
            core = [t for t in policy.core_tool_names]
            deferred = [t for t in policy.deferred_tool_names]
            definition = AgentDefinition(
                agent_type=policy.agent_type,
                description=policy.description,
                system_prompt=policy.system_instructions,
                allowed_tools=core + deferred,
                access_policy=policy.access_policy_ceiling,
                estimated_token_budget=policy.token_budget,
                estimated_work_budget=policy.work_budget,
                model_selection=policy.model_selection,
                output_model=policy.output_model,
                output_validation_max_retries=policy.output_validation_max_retries,
                max_stop_hook_blocks=policy.max_stop_hook_blocks,
                max_iterations=policy.max_iterations,
                max_depth=policy.max_depth,
                tool_policy=policy.tool_policy,
            )
        if definition is None:
            raise ValueError("Provide either 'definition' or 'policy'")
        self._definition = definition
        self._base_tool_registry = tool_registry
        self._catalog = catalog or self._build_catalog(tool_registry)
        self._model_turn_provider = model_turn_provider
        self._retrieval_hint_provider = retrieval_hint_provider
        self._subagent_runner = subagent_runner
        self._output_finalizer = output_finalizer
        self._model_registry = model_registry
        self._runtime_diagnostics = tuple(
            merge_runtime_diagnostics([], runtime_diagnostics)
        )
        self._checkpointer = checkpointer or create_agent_checkpointer(None)
        self._stream_sink = stream_sink
        # Register core tools: tool_search, activate_tools, task
        self._register_discovery_tools()
        self._register_task_tool()

    def initial_state(self, request: AgentRunRequest) -> LoopState:
        run_config = request.to_run_config(self._definition)
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
        state["approved_tool_call_ids"] = list(
            approved_tool_call_ids or ()
        )
        state["denied_tool_call_ids"] = list(
            denied_tool_call_ids or ()
        )
        return cast(
            LoopState,
            MessageCompactor(
                policy=run_config.memory_policy,
                store=memory_store,
            ).compact_initial_state(dict(state)),
        )

    async def run(self, request: AgentRunRequest) -> AgentRunResult:
        run_config = request.to_run_config(self._definition)
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
    ) -> AgentRunResult:
        from rag.agent.file_manifest import build_file_manifest
        from rag.agent.primitive_ops import PrimitiveOps
        from rag.agent.workspace import create_temp_workspace, import_files, open_workspace

        # 1. Create workspace
        if workspace_path:
            workspace = open_workspace(workspace_path, create=True)
        else:
            workspace = create_temp_workspace()

        # 2. Import input files
        if input_files:
            import_files(workspace, [Path(f) for f in input_files])

        # 3. Build file manifest (file-first processing)
        manifest = build_file_manifest(workspace)

        # 4. Create PrimitiveOps and inject runners
        ops = PrimitiveOps(workspace=workspace)
        memory_store = WorkspaceMemoryStore(
            workspace=workspace,
            policy=run_config.memory_policy,
        )

        # 5. Promote structured_probe to core if we have probeable files
        file_tools_to_activate: set[str] = (
            {"structured_probe"} if manifest.has_probeable_files else set()
        )
        runtime_registry = self._runtime_tool_registry(
            run_config,
            runners=ops.runners(),
        )
        self._validate_allowed_tools(runtime_registry)

        # 6. Inject manifest context into task
        enriched_task = self._inject_manifest_into_task(task, manifest)

        state = self._initial_loop_state_from_config(
            task=enriched_task,
            run_config=run_config,
            pending_tool_calls=pending_tool_calls,
            approved_tool_call_ids=approved_tool_call_ids,
            denied_tool_call_ids=denied_tool_call_ids,
            messages=messages,
            memory_store=memory_store,
            file_manifest=manifest,
        )
        await self._apply_retrieval_hint(state)

        # 7. Load persistent memories for cross-session context
        persistent_store = PersistentMemoryStore(workspace)
        await self._load_persistent_memories(state, persistent_store, task=enriched_task)
        checkpoint_store = LangGraphCheckpointStore(
            self._checkpointer,
            run_config=run_config,
            compatibility_config=GoalCompatibilityConfig(
                goal_spec=goal_spec
            ),
        )
        loop = self._build_loop(
            runtime_registry=runtime_registry,
            checkpoint_store=checkpoint_store,
            memory_store=memory_store,
            goal_spec=goal_spec,
            state=state,
            auto_activate_tools=file_tools_to_activate,
        )
        try:
            result_state = await loop.run(state)
        except Exception:
            RunRegistry.remove(run_config.run_id)
            raise
        if result_state["status"] in {"completed", "failed"}:
            RunRegistry.remove(run_config.run_id)

        # 8. Extract persistent memories from completed conversation
        if result_state["status"] == "completed":
            try:
                await self._extract_persistent_memories(result_state, persistent_store)
            except Exception:
                logger.warning("Persistent memory extraction failed", exc_info=True)

        return AgentRunResult.from_loop_result(
            result_state,
            definition=self._definition,
            workspace_path=str(workspace.root),
        )

    async def run_streaming(
        self, request: AgentRunRequest
    ) -> AsyncGenerator[Any, None]:
        """流式运行 Agent，yield 每一个 StreamEvent。

        用法：
            async for event in service.run_streaming(request):
                handle(event)
        """

        run_config = request.to_run_config(self._definition)

        # 复用 run_with_config 的 setup 逻辑
        from rag.agent.file_manifest import build_file_manifest
        from rag.agent.primitive_ops import PrimitiveOps
        from rag.agent.workspace import create_temp_workspace, import_files, open_workspace

        if request.workspace_path:
            workspace = open_workspace(request.workspace_path, create=True)
        else:
            workspace = create_temp_workspace()

        if request.input_files:
            import_files(workspace, [Path(f) for f in request.input_files])

        manifest = build_file_manifest(workspace)
        ops = PrimitiveOps(workspace=workspace)
        memory_store = WorkspaceMemoryStore(
            workspace=workspace,
            policy=run_config.memory_policy,
        )

        file_tools_to_activate: set[str] = (
            {"structured_probe"} if manifest.has_probeable_files else set()
        )
        runtime_registry = self._runtime_tool_registry(
            run_config,
            runners=ops.runners(),
        )
        self._validate_allowed_tools(runtime_registry)

        enriched_task = self._inject_manifest_into_task(request.task, manifest)

        state = self._initial_loop_state_from_config(
            task=enriched_task,
            run_config=run_config,
            pending_tool_calls=request.pending_tool_calls,
            approved_tool_call_ids=request.approved_tool_call_ids,
            denied_tool_call_ids=request.denied_tool_call_ids,
            messages=request.messages,
            memory_store=memory_store,
            file_manifest=manifest,
        )
        await self._apply_retrieval_hint(state)

        # Load persistent memories for cross-session context
        persistent_store = PersistentMemoryStore(workspace)
        await self._load_persistent_memories(state, persistent_store, task=enriched_task)

        checkpoint_store = LangGraphCheckpointStore(
            self._checkpointer,
            run_config=run_config,
            compatibility_config=GoalCompatibilityConfig(
                goal_spec=request.goal_spec
            ),
        )
        loop = self._build_loop(
            runtime_registry=runtime_registry,
            checkpoint_store=checkpoint_store,
            memory_store=memory_store,
            goal_spec=request.goal_spec,
            state=state,
            auto_activate_tools=file_tools_to_activate,
        )

        try:
            async for event in loop.run_streaming(state):
                yield event
        finally:
            if state["status"] in {"completed", "failed"}:
                RunRegistry.remove(run_config.run_id)
            # Extract persistent memories from completed conversation
            if state["status"] == "completed":
                try:
                    await self._extract_persistent_memories(state, persistent_store)
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
        state["approved_tool_call_ids"] = list(
            approved_tool_call_ids or ()
        )
        state["denied_tool_call_ids"] = list(
            denied_tool_call_ids or ()
        )
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
    ) -> AgentLoop:
        # Create and bind DeferredToolStore BEFORE resolving provider
        # (provider needs store for tool filtering)
        store = DeferredToolStore(
            max_active=self._definition.runtime_policy.max_active_deferred_tools,
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
        provider = self._resolve_model_turn_provider(state)
        output_finalizer = self._resolve_output_finalizer(state)
        return AgentLoop(
            definition=self._definition,
            model_provider=provider,
            context_manager=LoopContextCompactor(store=memory_store),
            tool_runner=ToolExecutionService(
                tool_registry=runtime_registry,
                record_writer=checkpoint_store,
                stream_sink=self._stream_sink,
            ),
            checkpoint_store=checkpoint_store,
            stop_hook_runner=StopHookRunner(
                hooks=build_stop_hooks(
                    definition=self._definition,
                    output_finalizer=output_finalizer,
                    goal_spec=goal_spec,
                ),
                max_blocks=self._definition.max_stop_hook_blocks,
            ),
            finish_candidate_builder=FinishCandidateBuilder(),
            catalog=self._catalog,
            deferred_store=store,
            stream_sink=self._stream_sink,
        )

    def _resolve_model_turn_provider(
        self,
        state: LoopState | None,
    ) -> ModelTurnProvider:
        if self._model_turn_provider is not None:
            return self._model_turn_provider
        if self._model_registry is not None:
            try:
                store = deferred_store_var.get(None)
                if store is None:
                    raise RuntimeError(
                        "DeferredToolStore is not bound — "
                        "AgentLoop must set deferred_store_var before creating provider"
                    )
                return create_loop_model_turn_provider(
                    self._model_registry,
                    self._definition.model_selection,
                    tool_registry=self._base_tool_registry,
                    definition=self._definition,
                    catalog=self._catalog,
                    deferred_store=store,
                    stream_sink=self._stream_sink,
                )
            except Exception as exc:
                if state is not None:
                    append_loop_diagnostic(
                        state,
                        RuntimeDiagnostic.from_exception(
                            code="default_providers_initialization_failed",
                            component="model_providers",
                            error=exc,
                        ),
                    )
        return _ResultDrivenModelTurnProvider()

    def _resolve_output_finalizer(
        self,
        state: LoopState | None,
    ) -> StructuredOutputFinalizer | None:
        if self._output_finalizer is not None:
            return self._output_finalizer
        if (
            self._definition.output_model is None
            or self._model_registry is None
        ):
            return None
        try:
            return create_model_structured_output_finalizer(
                self._model_registry
            )
        except Exception as exc:
            if state is not None:
                append_loop_diagnostic(
                    state,
                    RuntimeDiagnostic.from_exception(
                        code=(
                            "structured_output_finalizer_"
                            "initialization_failed"
                        ),
                        component="structured_output_finalizer",
                        error=exc,
                    ),
                )
            return None

    async def _apply_retrieval_hint(self, state: LoopState) -> None:
        provider = self._retrieval_hint_provider
        if provider is None:
            return
        try:
            update = provider.hint(state)
            if isawaitable(update):
                update = await update
        except Exception as exc:
            append_loop_diagnostic(
                state,
                RuntimeDiagnostic.from_exception(
                    code="retrieval_hint_failed",
                    component="retrieval_hint",
                    error=exc,
                ),
            )
            return
        retrieval_signals = update.get("retrieval_signals")
        if retrieval_signals is not None:
            state["retrieval_signals"] = retrieval_signals
        state["retrieval_signals_debug"] = update.get(
            "retrieval_signals_debug"
        )

    def _validate_allowed_tools(self, registry: ToolRegistry) -> None:
        registered = {tool.name for tool in registry.list_all()}
        missing = [
            name
            for name in self._definition.allowed_tools
            if name not in registered
        ]
        if missing:
            raise ValueError(
                f"unregistered tools: {', '.join(dict.fromkeys(missing))}"
            )

    @staticmethod
    def _inject_manifest_into_task(task: str, manifest: FileManifest) -> str:
        """Prepend file manifest context to the task string.

        The task becomes the first HumanMessage. By prepending the manifest,
        the model sees file info before the user's question, enabling
        file-first processing on turn 1.
        """
        if not manifest.files:
            return task

        block = manifest.to_context_block()
        if not block:
            return task

        return f"{block}\n\n── User Task ──\n{task}"

    # ── Persistent memory helpers ──

    async def _load_persistent_memories(
        self,
        state: LoopState,
        store: PersistentMemoryStore,
        *,
        task: str,
    ) -> None:
        """Load persistent memories into the loop state for context injection."""
        if not store.is_available:
            return

        try:
            from rag.agent.memory.persistent import MemorySelector

            index_content = store.read_index()
            state["memory_index"] = index_content

            if not index_content.strip():
                return

            # Create a memory gateway if available
            memory_gateway = self._create_memory_gateway("memory_select")
            if memory_gateway is None:
                # No gateway — fall back to rule-only selection
                selector = MemorySelector(max_selected=5, max_tokens=4000)
            else:
                selector = MemorySelector(
                    llm_gateway=memory_gateway,
                    max_selected=5,
                    max_tokens=4000,
                )

            selected = await selector.select(
                task=task,
                index_content=index_content,
                store=store,
            )
            state["persistent_memories"] = [m.to_markdown() for m in selected]
        except Exception:
            logger.warning("Failed to load persistent memories", exc_info=True)

    async def _extract_persistent_memories(
        self,
        state: LoopState,
        store: PersistentMemoryStore,
    ) -> None:
        """Extract new memories from the completed conversation (background task)."""
        if not store.is_available:
            return

        try:
            from rag.agent.memory.persistent import MemoryExtractor

            extract_gateway = self._create_memory_gateway("memory_extract")
            if extract_gateway is None:
                return

            extractor = MemoryExtractor(llm_gateway=extract_gateway)
            written = await extractor.extract(state=state, store=store)
            if written:
                logger.info("Extracted persistent memories: %s", written)

            # Optionally consolidate
            from rag.agent.memory.persistent import MemoryConsolidator

            consolidate_gateway = self._create_memory_gateway("memory_consolidate")
            consolidator = MemoryConsolidator(
                llm_gateway=consolidate_gateway or extract_gateway
            )
            result = await consolidator.consolidate(store)
            if result.action == "consolidated":
                logger.info(
                    "Consolidated memories: %d -> %d",
                    result.before_count,
                    result.after_count,
                )
        except Exception:
            logger.warning("Failed to extract persistent memories", exc_info=True)

    def _create_memory_gateway(
        self,
        stage: str = "memory_select",
    ) -> Any | None:
        """Create an LLM gateway for memory operations.

        Resolves the model for the given memory stage from the generation
        config in models.yaml. Falls back to the default model if no
        stage-specific config is found.

        Returns None if no model registry is available.
        """
        if self._model_registry is None:
            return None
        try:
            model_alias = self._resolve_memory_model_alias(stage)
            resolved = self._model_registry.resolve_or_fallback(model_alias)
            if resolved.gateway is not None:
                return resolved.gateway
            if resolved.token_accounting is None:
                return None
            from rag.providers.llm_gateway import LLMGateway

            return LLMGateway(
                generator=resolved.generator,
                token_accounting=resolved.token_accounting,
                model_context_tokens=resolved.context_window_tokens,
            )
        except Exception:
            logger.debug("Failed to create memory gateway for stage %s", stage, exc_info=True)
            return None

    def _resolve_memory_model_alias(self, stage: str) -> str:
        """Resolve model alias for a memory stage from the runtime config.

        Uses the GenerationConfig parsed from models.yaml by the catalog.
        Falls back to the default model if no stage-specific config is found.
        """
        if self._model_registry is None:
            return ""
        try:
            task_config = getattr(self._model_registry.generation_config, stage, None)
            if task_config is not None and task_config.model:
                return str(task_config.model)
        except Exception:
            logger.debug("Failed to resolve memory model from runtime config", exc_info=True)
        return str(self._model_registry.default_model)

    def _runtime_tool_registry(
        self,
        run_config: AgentRunConfig,
        *,
        runners: Mapping[str, ToolRunner] | None = None,
    ) -> ToolRegistry:
        """Clone base registry and inject AgentAsToolAdapter runners for this request.

        Agent-as-tool adapters are request-scoped — each run_config gets fresh adapters
        to prevent concurrent request pollution of depth/budget/access_policy.
        """
        runtime = self._base_tool_registry.clone()

        self._inject_model_llm_tool_runners(runtime)

        from rag.agent.core.agent_as_tool import AgentAsToolAdapter

        # Inject runtime runners (e.g. PrimitiveOps) — override existing runners
        if runners:
            for extra_name, extra_runner in runners.items():
                try:
                    runtime.register_runner(extra_name, extra_runner)
                except KeyError:
                    pass

        # Wire generic 'task' tool with request-scoped parent_config
        if runtime.has_runner("task") or "task" in {s.name for s in self._base_tool_registry.list_all()}:
            from rag.agent.tools.task_tool import TaskOutput, TaskToolRunner

            task_runner = TaskToolRunner(
                policy=self._definition.to_runtime_policy(),
                tool_registry=runtime,
                model_turn_provider=self._model_turn_provider,
                retrieval_hint_provider=self._retrieval_hint_provider,
                delegated_runner=_TaskChildRunner(
                    policy=self._definition.to_runtime_policy(),
                    tool_registry=runtime,
                    model_turn_provider=self._model_turn_provider,
                    retrieval_hint_provider=self._retrieval_hint_provider,
                ),
            )

            async def _task_runner(payload: BaseModel) -> TaskOutput:
                from rag.agent.tools.task_tool import TaskInput
                input_data = TaskInput.model_validate(payload)
                return await task_runner.run(input_data, parent_config=run_config)

            runtime.register_runner("task", _task_runner)

        if self._subagent_runner is None:
            return runtime

        runner = self._subagent_runner

        # Wire adapters for all agent-tool specs registered in the base registry
        for spec in self._base_tool_registry.list_all():
            if not spec.name.startswith("agent_"):
                continue
            agent_type = spec.name[len("agent_"):]
            adapter = AgentAsToolAdapter(
                runner=runner,
                agent_type=agent_type,
                run_config=run_config,
            )
            runtime.register_runner(spec.name, adapter)

        return runtime

    def _inject_model_llm_tool_runners(self, registry: ToolRegistry) -> None:
        if self._model_registry is None:
            return
        from rag.agent.core.llm_tool_runners import create_model_llm_tool_runners

        for tool_name, runner in create_model_llm_tool_runners(self._model_registry).items():
            if registry.has_runner(tool_name):
                continue
            try:
                registry.register_contextual_runner(tool_name, runner)
            except KeyError:
                pass

    def _build_catalog(self, registry: ToolRegistry) -> ToolCatalog:
        """Build a ToolCatalog from all tools in the registry.

        Categorizes each tool and builds search_text by flattening the
        input schema.  Only deferred tools are indexed for search.
        """
        filt = self._definition.runtime_policy.tool_catalog_filter
        catalog = ToolCatalog()
        for spec in registry.list_all():
            if spec.name in filt.deny:
                continue
            category: Literal["core", "deferred", "internal"]
            if spec.name in filt.promote_to_core or spec.name in CORE_TOOLS:
                category = "core"
            elif spec.name in DEFERRED_TOOLS:
                category = "deferred"
            else:
                category = "internal"
            # Build search_text for BM25 indexing
            schema_text = ""
            if category == "deferred" and hasattr(spec.input_model, "model_json_schema"):
                schema_text = flatten_schema(spec.input_model.model_json_schema())
            search_text = " ".join(filter(None, [
                spec.name,
                spec.name.replace("_", " "),
                spec.description,
                schema_text,
            ]))
            catalog.register(
                ToolCatalogEntry(
                    name=spec.name,
                    description=spec.description,
                    category=category,
                    search_text=search_text,
                    schema_text=schema_text,
                ),
            )
        return catalog

    def _register_discovery_tools(self) -> None:
        """Register tool_search and activate_tools as core tools.

        Runners read the per-run DeferredToolStore from a ContextVar,
        ensuring concurrent runs each get their own store.
        """
        if self._base_tool_registry.has_runner("tool_search"):
            return

        catalog = self._catalog
        definition = self._definition

        # ── tool_search ──
        search_spec = ToolSpec(
            name="tool_search",
            description=(
                "Discover available tools by natural language query. "
                "Returns candidate tools. Call activate_tools to load them."
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
                    "DeferredToolStore is not bound — "
                    "AgentLoop must set deferred_store_var before tool execution"
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
                    "DeferredToolStore is not bound — "
                    "AgentLoop must set deferred_store_var before tool execution"
                )
            return execute_activate_tools(
                names=input_data.names,
                catalog=catalog,
                store=store,
                allowed_tools=list(definition.allowed_tools),
                deny_tools=definition.tool_policy.deny_tools,
                iteration=iteration_var.get(0),
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
        from rag.agent.primitive_ops import PrimitiveOps
        from rag.agent.workspace import open_workspace

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

        runners: Mapping[str, ToolRunner] | None = None
        memory_store: WorkspaceMemoryStore | None = None
        if workspace_path:
            workspace = open_workspace(workspace_path)
            ops = PrimitiveOps(workspace=workspace)
            runners = ops.runners()
            memory_store = WorkspaceMemoryStore(
                workspace=workspace,
                policy=run_config.memory_policy,
            )
            RunRegistry.get(run_config.run_id).memory_store = memory_store

        runtime_registry = self._runtime_tool_registry(
            run_config,
            runners=runners,
        )
        self._validate_allowed_tools(runtime_registry)
        loop = self._build_loop(
            runtime_registry=runtime_registry,
            checkpoint_store=checkpoint_store,
            memory_store=memory_store,
            goal_spec=checkpoint_store.compatibility_config.goal_spec,
            state=state,
        )
        result_state = await loop.run(state)
        if result_state["status"] in {"completed", "failed"}:
            RunRegistry.remove(run_config.run_id)
        return AgentRunResult.from_loop_result(
            result_state,
            definition=self._definition,
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
            handles = RunRegistry.get_or_create(run_config)

        committed = max(0, run_config.budget_committed)
        if committed > 0:
            lease_id = f"checkpoint_restore:{run_config.run_id}"
            reserved = await handles.budget_ledger.reserve(
                lease_id,
                min(committed, run_config.budget_total),
            )
            if reserved:
                await handles.budget_ledger.commit(lease_id, committed)
        return run_config

    def _checkpoint_lookup_config(self, run_id: str) -> AgentRunConfig:
        return AgentRunConfig(
            run_id=run_id,
            thread_id=run_id,
            budget_total=self._definition.estimated_token_budget,
            work_budget_total=self._definition.estimated_work_budget,
            agent_type=self._definition.agent_type,
            max_depth=self._definition.max_depth,
            access_policy=(
                self._definition.access_policy
                or AccessPolicy.default()
            ),
            tool_policy=self._definition.tool_policy,
        )


def _restore_final_output(
    raw_output: ValidatedFinalOutput | dict[str, object] | None,
    *,
    definition: AgentDefinition | None,
) -> BaseModel | None:
    if raw_output is None or definition is None or definition.output_model is None:
        return None
    envelope = ValidatedFinalOutput.model_validate(raw_output)
    expected_path = output_model_path(definition.output_model)
    if envelope.model_path != expected_path:
        raise ValueError(
            "Checkpoint final output model does not match AgentDefinition.output_model"
        )
    return definition.output_model.model_validate(envelope.data)


__all__ = [
    "AgentRunRequest",
    "AgentRunResult",
    "AgentService",
]

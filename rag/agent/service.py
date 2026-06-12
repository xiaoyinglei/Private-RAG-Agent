from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

from langchain_core.messages import BaseMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from pydantic import BaseModel, ConfigDict, Field

from rag.agent.core.checkpointing import create_agent_checkpointer
from rag.agent.core.compiler import GraphCompiler
from rag.agent.core.context import AgentRunConfig, RunRegistry
from rag.agent.core.definition import AgentDefinition
from rag.agent.core.delegation import DelegatedAgentRunner
from rag.agent.core.human_input import HumanInputRequest, HumanInputResponse
from rag.agent.core.llm_registry import ModelRegistry
from rag.agent.core.output_finalizer import StructuredOutputFinalizer
from rag.agent.core.output_models import (
    ValidatedFinalOutput,
    output_model_path,
)
from rag.agent.core.runtime_diagnostics import (
    RuntimeDiagnostic,
    merge_runtime_diagnostics,
)
from rag.agent.goal_runtime import GoalSpec
from rag.agent.graphs.nodes.goal_runtime import GoalContractProvider
from rag.agent.graphs.nodes.llm_decide import ToolDecisionProvider
from rag.agent.graphs.nodes.retrieval_hint import RetrievalHintProvider
from rag.agent.graphs.nodes.synthesize import SynthesisRunner
from rag.agent.memory.compactor import MessageCompactor
from rag.agent.memory.models import MemoryPolicy
from rag.agent.memory.store import WorkspaceMemoryStore
from rag.agent.state import AgentState, ToolCallPlan, create_agent_state
from rag.agent.tools.registry import ToolRegistry, ToolRunner
from rag.agent.tools.spec import ToolResult
from rag.schema.query import AnswerCitation, EvidenceItem
from rag.schema.runtime import AccessPolicy


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
        state: AgentState,
        *,
        definition: AgentDefinition | None = None,
        workspace_path: str | None = None,
    ) -> AgentRunResult:
        run_config = state["run_config"]
        is_terminal = state["status"] in {"done", "failed"}
        human_request = None if is_terminal else state.get("human_input_request")
        pending = state.get("pending_tool_calls", [])
        return cls(
            run_id=run_config.run_id,
            thread_id=run_config.thread_id,
            status=state["status"],
            final_answer=state.get("final_answer"),
            final_output=_restore_final_output(
                state.get("final_output"),
                definition=definition,
            ),
            output_validation_errors=list(
                state.get("output_validation_errors", [])
            ),
            stop_reason=state.get("stop_reason"),
            tool_results=list(state.get("tool_results", [])),
            evidence=list(state.get("evidence", [])),
            citations=list(state.get("citations", [])),
            iteration=state.get("iteration", 0),
            groundedness_flag=state.get("groundedness_flag", False),
            insufficient_evidence_flag=state.get("insufficient_evidence_flag", False),
            needs_user_input=None if is_terminal else state.get("needs_user_input"),
            human_input_request=human_request,
            pending_tool_calls_summary=[
                {"tool_call_id": tc.tool_call_id, "tool_name": tc.tool_name}
                for tc in pending
            ],
            workspace_path=workspace_path,
            runtime_diagnostics=list(state.get("runtime_diagnostics", [])),
        )


class AgentService:
    def __init__(
        self,
        *,
        definition: AgentDefinition,
        tool_registry: ToolRegistry,
        tool_decision_provider: ToolDecisionProvider | None = None,
        goal_contract_provider: GoalContractProvider | None = None,
        retrieval_hint_provider: RetrievalHintProvider | None = None,
        subagent_runner: DelegatedAgentRunner | None = None,
        synthesis_runner: SynthesisRunner | None = None,
        output_finalizer: StructuredOutputFinalizer | None = None,
        model_registry: ModelRegistry | None = None,
        checkpointer: BaseCheckpointSaver[str] | None = None,
        runtime_diagnostics: Sequence[RuntimeDiagnostic] = (),
    ) -> None:
        self._definition = definition
        self._base_tool_registry = tool_registry
        self._tool_decision_provider = tool_decision_provider
        self._goal_contract_provider = goal_contract_provider
        self._retrieval_hint_provider = retrieval_hint_provider
        self._subagent_runner = subagent_runner
        self._synthesis_runner = synthesis_runner
        self._output_finalizer = output_finalizer
        self._model_registry = model_registry
        self._runtime_diagnostics = tuple(
            merge_runtime_diagnostics([], runtime_diagnostics)
        )
        self._checkpointer = checkpointer or create_agent_checkpointer(None)
        self._compiler = GraphCompiler(
            tool_registry=tool_registry,
            tool_decision_provider=tool_decision_provider,
            goal_contract_provider=goal_contract_provider,
            retrieval_hint_provider=retrieval_hint_provider,
            synthesis_runner=synthesis_runner,
            output_finalizer=output_finalizer,
            model_registry=model_registry,
            checkpointer=self._checkpointer,
        )

    def initial_state(self, request: AgentRunRequest) -> AgentState:
        run_config = request.to_run_config(self._definition)
        return self.initial_state_from_config(
            task=request.task,
            run_config=run_config,
            pending_tool_calls=request.pending_tool_calls,
            approved_tool_call_ids=request.approved_tool_call_ids,
            denied_tool_call_ids=request.denied_tool_call_ids,
            messages=request.messages,
            goal_spec=request.goal_spec,
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
        goal_spec: GoalSpec | None = None,
        memory_store: WorkspaceMemoryStore | None = None,
    ) -> AgentState:
        RunRegistry.remove(run_config.run_id)
        handles = RunRegistry.get_or_create(run_config)
        if memory_store is not None:
            handles.memory_store = memory_store
        state = create_agent_state(
            task=task,
            run_config=run_config,
            pending_tool_calls=pending_tool_calls,
            approved_tool_call_ids=approved_tool_call_ids,
            denied_tool_call_ids=denied_tool_call_ids,
            messages=messages,
            goal_spec=goal_spec,
            runtime_diagnostics=self._runtime_diagnostics,
        )
        return cast(
            AgentState,
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

        # 3. Create PrimitiveOps and inject runners
        ops = PrimitiveOps(workspace=workspace)
        memory_store = WorkspaceMemoryStore(
            workspace=workspace,
            policy=run_config.memory_policy,
        )
        runtime_registry = self._runtime_tool_registry(run_config, runners=ops.runners())
        compiler = GraphCompiler(
            tool_registry=runtime_registry,
            tool_decision_provider=self._tool_decision_provider,
            goal_contract_provider=self._goal_contract_provider,
            retrieval_hint_provider=self._retrieval_hint_provider,
            synthesis_runner=self._synthesis_runner,
            output_finalizer=self._output_finalizer,
            model_registry=self._model_registry,
            checkpointer=self._checkpointer,
        )
        graph = cast(Any, compiler.compile(self._definition))
        state = self.initial_state_from_config(
            task=task,
            run_config=run_config,
            pending_tool_calls=pending_tool_calls,
            approved_tool_call_ids=approved_tool_call_ids,
            denied_tool_call_ids=denied_tool_call_ids,
            messages=messages,
            goal_spec=goal_spec,
            memory_store=memory_store,
        )
        try:
            result_state = await graph.ainvoke(
                state,
                config={"configurable": {"thread_id": run_config.thread_id}},
            )
        except Exception:
            RunRegistry.remove(run_config.run_id)
            raise
        if result_state.get("status") in {"done", "failed"}:
            RunRegistry.remove(run_config.run_id)
        return AgentRunResult.from_state(
            result_state,
            definition=self._definition,
            workspace_path=str(workspace.root),
        )

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
        from langgraph.types import Command

        from rag.agent.primitive_ops import PrimitiveOps
        from rag.agent.workspace import open_workspace

        # 1. Restore PrimitiveOps runners if workspace_path provided
        runtime_registry = self._base_tool_registry.clone()
        self._inject_model_llm_tool_runners(runtime_registry)
        memory_store: WorkspaceMemoryStore | None = None
        if workspace_path:
            workspace = open_workspace(workspace_path)
            ops = PrimitiveOps(workspace=workspace)
            memory_store = WorkspaceMemoryStore(workspace=workspace)
            for extra_name, extra_runner in ops.runners().items():
                try:
                    runtime_registry.register_runner(extra_name, extra_runner)
                except KeyError:
                    pass

        compiler = GraphCompiler(
            tool_registry=runtime_registry,
            tool_decision_provider=self._tool_decision_provider,
            goal_contract_provider=self._goal_contract_provider,
            retrieval_hint_provider=self._retrieval_hint_provider,
            synthesis_runner=self._synthesis_runner,
            output_finalizer=self._output_finalizer,
            model_registry=self._model_registry,
            checkpointer=self._checkpointer,
        )
        graph = cast(Any, compiler.compile(self._definition))
        run_config = await self._restore_runtime_handles_from_checkpoint(
            graph,
            thread_id=run_id,
        )
        if memory_store is not None:
            RunRegistry.get(run_config.run_id).memory_store = WorkspaceMemoryStore(
                workspace=workspace,
                policy=run_config.memory_policy,
            )
        result_state = await graph.ainvoke(
            Command(resume=response.model_dump(mode="json")),
            config={"configurable": {"thread_id": run_config.thread_id}},
        )
        if result_state.get("status") in {"done", "failed"}:
            RunRegistry.remove(run_config.run_id)
        return AgentRunResult.from_state(
            result_state,
            definition=self._definition,
            workspace_path=workspace_path,
        )

    def pending_human_input_request(self, *, run_id: str) -> HumanInputRequest:
        graph = self._compiler.compile(self._definition)
        state = self._checkpoint_state(graph, thread_id=run_id)
        request = state.get("human_input_request")
        if request is None:
            raise KeyError(f"No pending human input request for run_id={run_id}")
        return request

    async def apending_human_input_request(self, *, run_id: str) -> HumanInputRequest:
        graph = self._compiler.compile(self._definition)
        state = await self._acheckpoint_state(graph, thread_id=run_id)
        request = state.get("human_input_request")
        if request is None:
            raise KeyError(f"No pending human input request for run_id={run_id}")
        return request

    async def _restore_runtime_handles_from_checkpoint(
        self,
        graph: object,
        *,
        thread_id: str,
    ) -> AgentRunConfig:
        state = await self._acheckpoint_state(graph, thread_id=thread_id)
        run_config = state["run_config"]
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

    @staticmethod
    def _checkpoint_state(graph: object, *, thread_id: str) -> AgentState:
        snapshot = graph.get_state(  # type: ignore[attr-defined]
            {"configurable": {"thread_id": thread_id}}
        )
        if not snapshot.values:
            raise KeyError(f"No checkpoint found for run_id={thread_id}")
        return cast(AgentState, snapshot.values)

    @staticmethod
    async def _acheckpoint_state(graph: object, *, thread_id: str) -> AgentState:
        snapshot = await graph.aget_state(  # type: ignore[attr-defined]
            {"configurable": {"thread_id": thread_id}}
        )
        if not snapshot.values:
            raise KeyError(f"No checkpoint found for run_id={thread_id}")
        return cast(AgentState, snapshot.values)


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

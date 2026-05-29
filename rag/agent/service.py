from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

from langchain_core.messages import BaseMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from pydantic import BaseModel, ConfigDict, Field

from rag.agent.core.compiler import AgentGraphCompiler
from rag.agent.core.context import AgentRunConfig, RuntimeRegistry
from rag.agent.core.definition import AgentDefinition
from rag.agent.core.delegation import DelegatedAgentRunner
from rag.agent.core.human_input import HumanInputRequest, HumanInputResponse
from rag.agent.core.llm_registry import ModelRegistry
from rag.agent.graphs.nodes.llm_decide import ToolDecisionProvider
from rag.agent.graphs.nodes.retrieval_hint import RetrievalHintProvider
from rag.agent.graphs.nodes.synthesize import SynthesisRunner
from rag.agent.state import AgentState, ToolCallPlan
from rag.agent.tools.registry import ToolRegistry, ToolRunner
from rag.agent.tools.spec import ToolResult
from rag.schema.query import AnswerCitation, EvidenceItem, RetrievalSignals
from rag.schema.runtime import AccessPolicy


class AgentRunRequest(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    task: str = Field(min_length=1)
    run_id: str | None = None
    thread_id: str | None = None
    budget_total: int | None = Field(default=None, gt=0)
    max_depth: int | None = Field(default=None, ge=0)
    pending_tool_calls: list[ToolCallPlan] = Field(default_factory=list)
    approved_tool_call_ids: list[str] = Field(default_factory=list)
    denied_tool_call_ids: list[str] = Field(default_factory=list)
    messages: list[BaseMessage] = Field(default_factory=list)
    input_files: list[str] = Field(default_factory=list)
    workspace_path: str | None = None

    def to_run_config(self, definition: AgentDefinition) -> AgentRunConfig:
        run_id = self.run_id or f"run_{uuid4().hex[:12]}"
        return AgentRunConfig(
            run_id=run_id,
            thread_id=self.thread_id or run_id,
            budget_total=self.budget_total or definition.estimated_token_budget,
            max_depth=definition.max_depth if self.max_depth is None else self.max_depth,
            access_policy=definition.access_policy or AccessPolicy.default(),
            tool_policy=definition.tool_policy,
        )


class AgentRunResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    run_id: str
    thread_id: str
    status: str
    final_answer: str | None = None
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

    @classmethod
    def from_state(cls, state: AgentState, *, workspace_path: str | None = None) -> AgentRunResult:
        run_config = state["run_config"]
        human_request = state.get("human_input_request")
        pending = state.get("pending_tool_calls", [])
        return cls(
            run_id=run_config.run_id,
            thread_id=run_config.thread_id,
            status=state["status"],
            final_answer=state.get("final_answer"),
            stop_reason=state.get("stop_reason"),
            tool_results=list(state.get("tool_results", [])),
            evidence=list(state.get("evidence", [])),
            citations=list(state.get("citations", [])),
            iteration=state.get("iteration", 0),
            groundedness_flag=state.get("groundedness_flag", False),
            insufficient_evidence_flag=state.get("insufficient_evidence_flag", False),
            needs_user_input=state.get("needs_user_input"),
            human_input_request=human_request,
            pending_tool_calls_summary=[
                {"tool_call_id": tc.tool_call_id, "tool_name": tc.tool_name}
                for tc in pending
            ],
            workspace_path=workspace_path,
        )


class AgentService:
    def __init__(
        self,
        *,
        definition: AgentDefinition,
        tool_registry: ToolRegistry,
        tool_decision_provider: ToolDecisionProvider | None = None,
        retrieval_hint_provider: RetrievalHintProvider | None = None,
        subagent_runner: DelegatedAgentRunner | None = None,
        synthesis_runner: SynthesisRunner | None = None,
        model_registry: ModelRegistry | None = None,
        checkpointer: BaseCheckpointSaver[str] | None = None,
    ) -> None:
        self._definition = definition
        self._base_tool_registry = tool_registry
        self._tool_decision_provider = tool_decision_provider
        self._retrieval_hint_provider = retrieval_hint_provider
        self._subagent_runner = subagent_runner
        self._synthesis_runner = synthesis_runner
        self._model_registry = model_registry
        self._checkpointer = checkpointer
        self._compiler = AgentGraphCompiler(
            tool_registry=tool_registry,
            tool_decision_provider=tool_decision_provider,
            retrieval_hint_provider=retrieval_hint_provider,
            synthesis_runner=synthesis_runner,
            model_registry=model_registry,
            checkpointer=checkpointer,
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
    ) -> AgentState:
        RuntimeRegistry.remove(run_config.run_id)
        RuntimeRegistry.get_or_create(run_config)
        return {
            "messages": list(messages or []),
            "evidence": [],
            "citations": [],
            "tool_results": [],
            "task": task,
            "retrieval_signals": RetrievalSignals(),
            "retrieval_signals_debug": None,
            "run_config": run_config,
            "iteration": 0,
            "status": "running",
            "decision_reason": None,
            "stop_reason": None,
            "needs_user_input": None,
            "pending_tool_calls": list(pending_tool_calls or []),
            "approved_tool_call_ids": list(approved_tool_call_ids or []),
            "denied_tool_call_ids": list(denied_tool_call_ids or []),
            "user_decision": None,
            "user_message": None,
            "human_input_request": None,
            "human_input_response": None,
            "working_summary": None,
            "extracted_facts": [],
            "context_budget": None,
            "final_answer": None,
            "groundedness_flag": False,
            "insufficient_evidence_flag": False,
            "goal_spec": None,
            "goal_requirements": [],
            "satisfied_requirements": [],
            "open_gaps": [],
            "evidence_refs": [],
            "answer_candidates": [],
            "computation_results": [],
            "structured_observations": [],
            "context_units": [],
            "context_bindings": [],
            "locators": [],
            "asset_refs": [],
            "conflicts": [],
            "no_progress_count": 0,
            "satisfaction_report": None,
            "controller_next": None,
        }

    async def run(self, request: AgentRunRequest) -> AgentRunResult:
        run_config = request.to_run_config(self._definition)
        return await self.run_with_config(
            task=request.task,
            run_config=run_config,
            pending_tool_calls=request.pending_tool_calls,
            approved_tool_call_ids=request.approved_tool_call_ids,
            denied_tool_call_ids=request.denied_tool_call_ids,
            messages=request.messages,
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
        runtime_registry = self._runtime_tool_registry(run_config, runners=ops.runners())
        compiler = AgentGraphCompiler(
            tool_registry=runtime_registry,
            tool_decision_provider=self._tool_decision_provider,
            retrieval_hint_provider=self._retrieval_hint_provider,
            synthesis_runner=self._synthesis_runner,
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
        )
        try:
            result_state = await graph.ainvoke(
                state,
                config={"configurable": {"thread_id": run_config.thread_id}},
            )
        except Exception:
            RuntimeRegistry.remove(run_config.run_id)
            raise
        if result_state.get("status") in {"done", "failed"}:
            RuntimeRegistry.remove(run_config.run_id)
        return AgentRunResult.from_state(result_state, workspace_path=str(workspace.root))

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

        from rag.agent.core.agent_as_tool import AgentAsToolAdapter

        # Inject runtime runners (e.g. PrimitiveOps)
        if runners:
            for name, runner in runners.items():
                if runtime.has_runner(name):
                    continue
                runtime.register_runner(name, runner)

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


    async def resume(
        self,
        *,
        run_id: str,
        response: HumanInputResponse,
    ) -> AgentRunResult:
        """从中断点恢复。response 对应 HumanInputRequest 的用户响应。"""
        from langgraph.types import Command

        graph = cast(Any, self._compiler.compile(self._definition))
        run_config = await self._restore_runtime_handles_from_checkpoint(
            graph,
            thread_id=run_id,
        )
        result_state = await graph.ainvoke(
            Command(resume=response.model_dump(mode="json")),
            config={"configurable": {"thread_id": run_config.thread_id}},
        )
        if result_state.get("status") in {"done", "failed"}:
            RuntimeRegistry.remove(run_config.run_id)
        return AgentRunResult.from_state(result_state)

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
            RuntimeRegistry.get(run_config.run_id)
            return run_config
        except KeyError:
            handles = RuntimeRegistry.get_or_create(run_config)

        committed = sum(
            max(0, getattr(result, "token_used", 0))
            for result in state.get("tool_results", [])
        )
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


__all__ = [
    "AgentRunRequest",
    "AgentRunResult",
    "AgentService",
]

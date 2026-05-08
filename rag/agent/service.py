from __future__ import annotations

from uuid import uuid4

from langgraph.graph.message import BaseMessage
from pydantic import BaseModel, ConfigDict, Field

from rag.agent.core.compiler import AgentGraphCompiler
from rag.agent.core.context import AgentRunConfig, RuntimeRegistry
from rag.agent.core.definition import AgentDefinition
from rag.agent.graphs.nodes.evaluate import EvaluateDecisionProvider
from rag.agent.state import AgentState, ToolCallPlan
from rag.agent.tools.registry import ToolRegistry
from rag.agent.tools.spec import ToolResult
from rag.schema.runtime import AccessPolicy, ExecutionLocationPreference


class AgentRunRequest(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    task: str = Field(min_length=1)
    run_id: str | None = None
    thread_id: str | None = None
    budget_total: int | None = Field(default=None, gt=0)
    max_depth: int | None = Field(default=None, ge=0)
    pending_tool_calls: list[ToolCallPlan] = Field(default_factory=list)
    confirmed_tool_call_ids: set[str] = Field(default_factory=set)
    messages: list[BaseMessage] = Field(default_factory=list)

    def to_run_config(self, definition: AgentDefinition) -> AgentRunConfig:
        run_id = self.run_id or f"run_{uuid4().hex[:12]}"
        return AgentRunConfig(
            run_id=run_id,
            thread_id=self.thread_id or run_id,
            budget_total=self.budget_total or definition.estimated_token_budget,
            max_depth=definition.max_depth if self.max_depth is None else self.max_depth,
            access_policy=definition.access_policy or AccessPolicy.default(),
            execution_location_preference=ExecutionLocationPreference.LOCAL_FIRST,
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
    iteration: int = 0
    groundedness_flag: bool = False
    insufficient_evidence_flag: bool = False

    @classmethod
    def from_state(cls, state: AgentState) -> AgentRunResult:
        run_config = state["run_config"]
        return cls(
            run_id=run_config.run_id,
            thread_id=run_config.thread_id,
            status=state["status"],
            final_answer=state.get("final_answer"),
            stop_reason=state.get("stop_reason"),
            tool_results=list(state.get("tool_results", [])),
            iteration=state.get("iteration", 0),
            groundedness_flag=state.get("groundedness_flag", False),
            insufficient_evidence_flag=state.get("insufficient_evidence_flag", False),
        )


class AgentService:
    def __init__(
        self,
        *,
        definition: AgentDefinition,
        tool_registry: ToolRegistry,
        query_understanding_service: object | None = None,
        evaluate_decision_provider: EvaluateDecisionProvider | None = None,
    ) -> None:
        self._definition = definition
        self._compiler = AgentGraphCompiler(
            tool_registry=tool_registry,
            query_understanding_service=query_understanding_service,
            evaluate_decision_provider=evaluate_decision_provider,
        )

    def initial_state(self, request: AgentRunRequest) -> AgentState:
        run_config = request.to_run_config(self._definition)
        RuntimeRegistry.remove(run_config.run_id)
        RuntimeRegistry.get_or_create(run_config)
        return {
            "messages": list(request.messages),
            "evidence": [],
            "citations": [],
            "tool_results": [],
            "task": request.task,
            "run_config": run_config,
            "plan": None,
            "iteration": 0,
            "status": "running",
            "route_reason": None,
            "stop_reason": None,
            "needs_user_input": None,
            "pending_tool_calls": list(request.pending_tool_calls),
            "confirmed_tool_call_ids": set(request.confirmed_tool_call_ids),
            "user_decision": None,
            "next_subtasks": None,
            "working_summary": None,
            "extracted_facts": [],
            "context_budget": None,
            "subtask_results": {},
            "terminal_subtasks": set(),
            "successful_subtasks": set(),
            "final_answer": None,
            "groundedness_flag": False,
            "insufficient_evidence_flag": False,
        }

    async def run(self, request: AgentRunRequest) -> AgentRunResult:
        graph = self._compiler.compile(self._definition)
        state = self.initial_state(request)
        run_config = state["run_config"]
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
        return AgentRunResult.from_state(result_state)


__all__ = [
    "AgentRunRequest",
    "AgentRunResult",
    "AgentService",
]

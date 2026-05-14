from __future__ import annotations

from uuid import uuid4

from langgraph.graph.message import BaseMessage
from pydantic import BaseModel, ConfigDict, Field

from rag.agent.core.compiler import AgentGraphCompiler
from rag.agent.core.context import AgentRunConfig, RuntimeRegistry
from rag.agent.core.definition import AgentDefinition
from rag.agent.core.human_input import HumanInputResponse
from rag.agent.core.llm_registry import ModelRegistry
from rag.agent.graphs.nodes.evaluate import EvaluateDecisionProvider
from rag.agent.graphs.nodes.execute_subagent import SubAgentRunner
from rag.agent.graphs.nodes.plan import PlanProvider
from rag.agent.graphs.nodes.route import RouteProvider
from rag.agent.graphs.nodes.synthesize import SynthesisRunner
from rag.agent.state import AgentState, ToolCallPlan
from rag.agent.tools.registry import ToolRegistry
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

    @classmethod
    def from_state(cls, state: AgentState) -> AgentRunResult:
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
        )


class AgentService:
    def __init__(
        self,
        *,
        definition: AgentDefinition,
        tool_registry: ToolRegistry,
        evaluate_decision_provider: EvaluateDecisionProvider | None = None,
        plan_provider: PlanProvider | None = None,
        route_provider: RouteProvider | None = None,
        subagent_runner: SubAgentRunner | None = None,
        synthesis_runner: SynthesisRunner | None = None,
        model_registry: ModelRegistry | None = None,
    ) -> None:
        self._definition = definition
        self._compiler = AgentGraphCompiler(
            tool_registry=tool_registry,
            evaluate_decision_provider=evaluate_decision_provider,
            plan_provider=plan_provider,
            route_provider=route_provider,
            subagent_runner=subagent_runner,
            synthesis_runner=synthesis_runner,
            model_registry=model_registry,
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
            "plan": None,
            "iteration": 0,
            "status": "running",
            "route_reason": None,
            "stop_reason": None,
            "needs_user_input": None,
            "pending_tool_calls": list(pending_tool_calls or []),
            "approved_tool_call_ids": list(approved_tool_call_ids or []),
            "denied_tool_call_ids": list(denied_tool_call_ids or []),
            "user_decision": None,
            "user_message": None,
            "human_input_request": None,
            "human_input_response": None,
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
        run_config = request.to_run_config(self._definition)
        return await self.run_with_config(
            task=request.task,
            run_config=run_config,
            pending_tool_calls=request.pending_tool_calls,
            approved_tool_call_ids=request.approved_tool_call_ids,
            denied_tool_call_ids=request.denied_tool_call_ids,
            messages=request.messages,
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
    ) -> AgentRunResult:
        graph = self._compiler.compile(self._definition)
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
        return AgentRunResult.from_state(result_state)


    async def resume(
        self,
        *,
        run_id: str,
        response: HumanInputResponse,
    ) -> AgentRunResult:
        """从中断点恢复。response 对应 HumanInputRequest 的用户响应。"""
        from langgraph.types import Command

        graph = self._compiler.compile(self._definition)
        result_state = await graph.ainvoke(
            Command(resume=response.model_dump(mode="json")),
            config={"configurable": {"thread_id": run_id}},
        )
        if result_state.get("status") in {"done", "failed"}:
            RuntimeRegistry.remove(run_id)
        return AgentRunResult.from_state(result_state)


__all__ = [
    "AgentRunRequest",
    "AgentRunResult",
    "AgentService",
]

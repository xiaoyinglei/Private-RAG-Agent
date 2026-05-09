from __future__ import annotations

from typing import Protocol

from rag.agent.state import AgentState
from rag.schema.query import QueryUnderstanding, TaskType


class QueryUnderstandingLike(Protocol):
    def analyze(
        self,
        query: str,
        *,
        access_policy: object | None = None,
        execution_location_preference: object | None = None,
    ) -> QueryUnderstanding: ...


def route_node(state: AgentState, *, query_understanding_service: QueryUnderstandingLike) -> dict:
    if state.get("pending_tool_calls"):
        return {"status": "direct", "route_reason": "pending_tool_calls"}

    understanding = query_understanding_service.analyze(
        state.get("task", ""),
        access_policy=state["run_config"].access_policy,
        execution_location_preference=state["run_config"].execution_location_preference,
    )
    task_type = understanding.task_type
    if task_type in {TaskType.LOOKUP, TaskType.SINGLE_DOC_QA}:
        return {"status": "fast_path", "route_reason": "simple_lookup"}
    if task_type in {TaskType.COMPARISON, TaskType.SYNTHESIS, TaskType.TIMELINE}:
        return {"status": "decompose", "route_reason": "multi_hop_or_compare"}
    return {"status": "direct", "route_reason": "single_agent_research"}


def route_after_route(state: AgentState) -> str:
    if state.get("status") == "fast_path":
        return "synthesize"
    if state.get("status") == "failed":
        return "synthesize"
    if state.get("status") == "decompose":
        return "plan"
    return "execute"

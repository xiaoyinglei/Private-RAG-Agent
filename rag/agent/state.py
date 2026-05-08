from __future__ import annotations

from typing import Annotated, Literal

from langgraph.graph import add_messages
from langgraph.graph.message import BaseMessage
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from rag.agent.core.context import AgentRunConfig
from rag.agent.core.task import SubTaskResult, TaskDAG
from rag.agent.memory.models import ContextBudgetSnapshot, ExtractedFact, WorkingSummary


class ToolCallPlan(BaseModel):
    tool_call_id: str
    tool_name: str
    arguments: dict[str, object]

    @classmethod
    def create(cls, tool_name: str, arguments: dict[str, object]) -> ToolCallPlan:
        from uuid import uuid4

        return cls(
            tool_call_id=f"tc_{uuid4().hex[:12]}",
            tool_name=tool_name,
            arguments=arguments,
        )


class ThinkOutput(BaseModel):
    action: Literal["execute", "synthesize", "pause"]
    tool_calls: list[ToolCallPlan] = Field(default_factory=list)
    thought: str
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    stop_reason: str | None = None
    needs_user_input: str | None = None


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    evidence: Annotated[list, _merge_evidence]
    citations: Annotated[list, _merge_citations]
    tool_results: Annotated[list, _merge_tool_results]
    task: str
    run_config: AgentRunConfig
    plan: TaskDAG | None
    iteration: int
    status: str
    route_reason: str | None
    stop_reason: str | None
    needs_user_input: str | None
    pending_tool_calls: list[ToolCallPlan]
    confirmed_tool_call_ids: set[str]
    user_decision: str | None
    next_subtasks: list[object] | None
    working_summary: WorkingSummary | None
    extracted_facts: list[ExtractedFact]
    context_budget: ContextBudgetSnapshot | None
    subtask_results: Annotated[dict[str, SubTaskResult], _merge_subtask_results]
    terminal_subtasks: Annotated[set[str], _merge_sets]
    successful_subtasks: Annotated[set[str], _merge_sets]
    final_answer: str | None
    groundedness_flag: bool
    insufficient_evidence_flag: bool


def _merge_evidence(left: list, right: list) -> list:
    from rag.schema.query import EvidenceItem

    merged: dict[str, EvidenceItem] = {}
    for item in left + right:
        existing = merged.get(item.evidence_id)
        if existing is None:
            merged[item.evidence_id] = item
        elif _texts_contradict(existing.text, item.text):
            merged[item.evidence_id] = existing.model_copy(
                update={"retrieval_channels": [*existing.retrieval_channels, "conflict"]}
            )
            conflict_id = f"{item.evidence_id}__conflict"
            merged[conflict_id] = item.model_copy(
                update={
                    "evidence_id": conflict_id,
                    "retrieval_channels": [*item.retrieval_channels, "conflict"],
                }
            )
        elif item.score > existing.score:
            merged[item.evidence_id] = item
    return sorted(merged.values(), key=lambda evidence: evidence.score, reverse=True)


def _texts_contradict(a: str, b: str) -> bool:
    negation_markers = (" not ", " no ", " does not ", " cannot ", " without ", "未", "不")
    a_lower = f" {a.lower().strip()} "
    b_lower = f" {b.lower().strip()} "
    a_negated = any(marker in a_lower for marker in negation_markers)
    b_negated = any(marker in b_lower for marker in negation_markers)
    return a_negated != b_negated


def _merge_citations(left: list, right: list) -> list:
    return list({citation.citation_id: citation for citation in left + right}.values())


def _merge_tool_results(left: list, right: list) -> list:
    return list({result.tool_call_id: result for result in left + right}.values())


def _merge_subtask_results(left: dict, right: dict) -> dict:
    return {**left, **right}


def _merge_sets(left: set, right: set) -> set:
    return left | right


__all__ = [
    "AgentState",
    "ContextBudgetSnapshot",
    "ExtractedFact",
    "ThinkOutput",
    "ToolCallPlan",
    "WorkingSummary",
    "_merge_citations",
    "_merge_evidence",
    "_merge_sets",
    "_merge_subtask_results",
    "_merge_tool_results",
]

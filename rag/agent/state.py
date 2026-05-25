from __future__ import annotations

from typing import Annotated, Any, Literal

from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from rag.agent.core.context import AgentRunConfig
from rag.agent.core.human_input import HumanInputRequest, HumanInputResponse
from rag.agent.memory.models import ContextBudgetSnapshot, ExtractedFact, WorkingSummary
from rag.schema.query import RetrievalSignals


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
    evidence: Annotated[list[Any], _merge_evidence]
    citations: Annotated[list[Any], _merge_citations]
    tool_results: Annotated[list[Any], _merge_tool_results]
    task: str
    retrieval_signals: RetrievalSignals
    retrieval_signals_debug: dict[str, object] | None
    run_config: AgentRunConfig
    iteration: int
    status: str
    decision_reason: str | None
    stop_reason: str | None
    needs_user_input: str | None
    pending_tool_calls: list[ToolCallPlan]
    approved_tool_call_ids: list[str]
    denied_tool_call_ids: list[str]
    user_decision: str | None
    user_message: str | None
    human_input_request: HumanInputRequest | None
    human_input_response: HumanInputResponse | None
    working_summary: WorkingSummary | None
    extracted_facts: list[ExtractedFact]
    context_budget: ContextBudgetSnapshot | None
    final_answer: str | None
    groundedness_flag: bool
    insufficient_evidence_flag: bool
    goal_spec: Any | None
    goal_requirements: list[str]
    satisfied_requirements: list[str]
    open_gaps: list[Any]
    evidence_refs: Annotated[list[Any], _merge_keyed_items]
    answer_candidates: Annotated[list[Any], _merge_keyed_items]
    computation_results: Annotated[list[Any], _merge_keyed_items]
    structured_observations: Annotated[list[Any], _merge_keyed_items]
    context_units: Annotated[list[Any], _merge_keyed_items]
    context_bindings: Annotated[list[Any], _merge_keyed_items]
    locators: Annotated[list[Any], _merge_keyed_items]
    asset_refs: Annotated[list[int], _merge_ints]
    conflicts: list[Any]
    no_progress_count: int
    satisfaction_report: Any | None
    controller_next: str | None


def _merge_evidence(left: list[Any], right: list[Any]) -> list[Any]:
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


def _merge_citations(left: list[Any], right: list[Any]) -> list[Any]:
    return list({citation.citation_id: citation for citation in left + right}.values())


def _merge_tool_results(left: list[Any], right: list[Any]) -> list[Any]:
    return list({result.tool_call_id: result for result in left + right}.values())


def _merge_keyed_items(left: list[Any], right: list[Any]) -> list[Any]:
    merged: dict[str, Any] = {}
    for index, item in enumerate(left + right):
        key = _item_key(item, fallback=str(index))
        merged[key] = item
    return list(merged.values())


def _item_key(item: Any, *, fallback: str) -> str:
    key = getattr(item, "key", None)
    if isinstance(key, str) and key:
        return key
    for attr in ("tool_call_id", "source_tool_call_id", "evidence_id", "citation_id"):
        value = getattr(item, attr, None)
        if value:
            return f"{attr}:{value}"
    if isinstance(item, dict):
        for attr in ("tool_call_id", "source_tool_call_id", "evidence_id", "citation_id"):
            value = item.get(attr)
            if value:
                return f"{attr}:{value}"
        return str(sorted(item.items()))
    return fallback


def _merge_ints(left: list[int], right: list[int]) -> list[int]:
    return list(dict.fromkeys([*left, *right]))


__all__ = [
    "AgentState",
    "ContextBudgetSnapshot",
    "ExtractedFact",
    "ThinkOutput",
    "ToolCallPlan",
    "WorkingSummary",
    "_merge_citations",
    "_merge_evidence",
    "_merge_ints",
    "_merge_keyed_items",
    "_merge_tool_results",
]

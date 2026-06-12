from __future__ import annotations

from typing import Annotated, Any, Literal, cast

from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from rag.agent.core.context import AgentRunConfig
from rag.agent.core.human_input import HumanInputRequest, HumanInputResponse
from rag.agent.core.output_models import ValidatedFinalOutput
from rag.agent.memory.models import (
    ContextBudgetSnapshot,
    ExtractedFact,
    MemoryBudgetSnapshot,
    MemoryRef,
    StateChannelReplacement,
    WorkingSummary,
)
from rag.agent.planning import MAX_PLAN_EVENTS, AgentPlan, PlanEvent, PlanUpdate
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
    plan_update: PlanUpdate | None = None


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], _merge_messages]
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
    final_output: ValidatedFinalOutput | None
    output_validation_errors: list[dict[str, object]]
    groundedness_flag: bool
    insufficient_evidence_flag: bool
    goal_spec: Any | None
    goal_contract_hint: Any | None
    goal_contract_debug: dict[str, object] | None
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
    agent_plan: AgentPlan | None
    plan_events: Annotated[list[PlanEvent], _merge_plan_events]
    memory_refs: Annotated[list[MemoryRef], _merge_memory_refs]
    memory_budget: MemoryBudgetSnapshot | None
    memory_warnings: Annotated[list[str], _merge_strings]


def create_agent_state(
    *,
    task: str,
    run_config: AgentRunConfig,
    messages: list[BaseMessage] | None = None,
    pending_tool_calls: list[ToolCallPlan] | None = None,
    approved_tool_call_ids: list[str] | None = None,
    denied_tool_call_ids: list[str] | None = None,
    goal_spec: Any | None = None,
) -> AgentState:
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
        "final_output": None,
        "output_validation_errors": [],
        "groundedness_flag": False,
        "insufficient_evidence_flag": False,
        "goal_spec": goal_spec,
        "goal_contract_hint": None,
        "goal_contract_debug": None,
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
        "agent_plan": None,
        "plan_events": [],
        "memory_refs": [],
        "memory_budget": None,
        "memory_warnings": [],
    }


def _merge_evidence(left: list[Any], right: list[Any]) -> list[Any]:
    from rag.schema.query import EvidenceItem

    replacement = _replacement_items(right)
    if replacement is not None:
        return replacement

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


def _merge_messages(left: list[BaseMessage], right: list[Any]) -> list[BaseMessage]:
    replacement = _replacement_items(right)
    if replacement is not None:
        return cast(list[BaseMessage], replacement)
    return cast(list[BaseMessage], add_messages(cast(Any, left), cast(Any, right)))


def _texts_contradict(a: str, b: str) -> bool:
    negation_markers = (" not ", " no ", " does not ", " cannot ", " without ", "未", "不")
    a_lower = f" {a.lower().strip()} "
    b_lower = f" {b.lower().strip()} "
    a_negated = any(marker in a_lower for marker in negation_markers)
    b_negated = any(marker in b_lower for marker in negation_markers)
    return a_negated != b_negated


def _merge_citations(left: list[Any], right: list[Any]) -> list[Any]:
    replacement = _replacement_items(right)
    if replacement is not None:
        return replacement
    return list({citation.citation_id: citation for citation in left + right}.values())


def _merge_tool_results(left: list[Any], right: list[Any]) -> list[Any]:
    replacement = _replacement_items(right)
    if replacement is not None:
        return replacement
    return list({result.tool_call_id: result for result in left + right}.values())


def _merge_keyed_items(left: list[Any], right: list[Any]) -> list[Any]:
    replacement = _replacement_items(right)
    if replacement is not None:
        return replacement
    merged: dict[str, Any] = {}
    for index, item in enumerate(left + right):
        key = _item_key(item, fallback=str(index))
        merged[key] = item
    return list(merged.values())


def _merge_plan_events(left: list[PlanEvent], right: list[PlanEvent]) -> list[PlanEvent]:
    merged = _merge_keyed_items(left, right)
    return cast(list[PlanEvent], merged[-MAX_PLAN_EVENTS:])


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
    replacement = _replacement_items(right)
    if replacement is not None:
        return cast(list[int], replacement)
    return list(dict.fromkeys([*left, *right]))


def _merge_memory_refs(left: list[MemoryRef], right: list[MemoryRef]) -> list[MemoryRef]:
    replacement = _replacement_items(cast(list[Any], right))
    if replacement is not None:
        return cast(list[MemoryRef], replacement)
    return cast(list[MemoryRef], _merge_keyed_items(list(left), list(right)))


def _merge_strings(left: list[str], right: list[str]) -> list[str]:
    replacement = _replacement_items(cast(list[Any], right))
    if replacement is not None:
        return [str(item) for item in replacement]
    return list(dict.fromkeys([*left, *right]))


def _replacement_items(right: list[Any]) -> list[Any] | None:
    if len(right) == 1 and isinstance(right[0], StateChannelReplacement):
        return list(right[0].items)
    return None


__all__ = [
    "AgentState",
    "AgentPlan",
    "ContextBudgetSnapshot",
    "ExtractedFact",
    "MemoryBudgetSnapshot",
    "MemoryRef",
    "PlanEvent",
    "PlanUpdate",
    "ThinkOutput",
    "ToolCallPlan",
    "WorkingSummary",
    "create_agent_state",
    "_merge_messages",
    "_merge_citations",
    "_merge_evidence",
    "_merge_ints",
    "_merge_keyed_items",
    "_merge_memory_refs",
    "_merge_plan_events",
    "_merge_strings",
    "_merge_tool_results",
]

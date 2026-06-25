from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Literal, Self

from langchain_core.messages import BaseMessage
from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing_extensions import TypedDict

from rag.agent.core.context import AgentRunConfig
from rag.agent.core.human_input import HumanInputRequest, HumanInputResponse
from rag.agent.core.output_models import ValidatedFinalOutput
from rag.agent.core.runtime_diagnostics import (
    RuntimeDiagnostic,
    merge_runtime_diagnostics,
)
from rag.agent.core.tool_execution import ToolExecutionRecord
from rag.agent.core.turn_contracts import ToolCallPlan
from rag.agent.memory.models import (
    ContextBudgetSnapshot,
    ExtractedFact,
    MemoryBudgetSnapshot,
    MemoryRef,
    WorkingSummary,
)
from rag.agent.planning import AgentPlan, PlanEvent
from rag.agent.tools.spec import ToolResult

if TYPE_CHECKING:
    from rag.agent.file_manifest import FileManifest
    from rag.agent.loop.substate import (
        DeferredToolState,
        FinishState,
        MemoryState,
        PlanState,
    )

MAX_STOP_HOOK_FEEDBACK = 10
MAX_LOOP_MEMORY_WARNINGS = 20

LoopStatus = Literal["running", "paused", "completed", "failed"]
LoopTransitionReason = Literal[
    "next_turn",
    "tool_execution",
    "approval_required",
    "stop_hook_blocked",
    "retry",
    "fallback",
    "compaction",
    "paused",
    "finished",
    "failed",
    "max_iterations",
]


class ModelTurnDraft(BaseModel):
    """Provider output before strict kernel validation."""

    model_config = ConfigDict(frozen=True)

    action: Literal["execute", "finish", "pause"]
    tool_calls: tuple[ToolCallPlan, ...] = ()
    final_answer: str | None = None
    pause_reason: str | None = None


class ModelTurn(BaseModel):
    """A complete, unambiguous model outcome accepted by the loop kernel."""

    model_config = ConfigDict(frozen=True)

    action: Literal["execute", "finish", "pause"]
    tool_calls: tuple[ToolCallPlan, ...] = ()
    final_answer: str | None = None
    pause_reason: str | None = None

    @model_validator(mode="after")
    def validate_outcome(self) -> Self:
        if self.action == "execute" and not self.tool_calls:
            raise ValueError("execute requires at least one tool call")
        if self.action != "execute" and self.tool_calls:
            raise ValueError("tool calls require execute action")
        if self.action == "finish" and not _nonempty(self.final_answer):
            raise ValueError("finish requires a non-empty final answer")
        if self.action == "pause" and not _nonempty(self.pause_reason):
            raise ValueError("pause requires a non-empty reason")
        return self


class LoopTransition(BaseModel):
    """Latest bounded transition marker persisted with loop state."""

    model_config = ConfigDict(frozen=True)

    reason: LoopTransitionReason
    iteration: int = Field(ge=0)
    detail: dict[str, object] = Field(default_factory=dict)


class LoopPause(BaseModel):
    model_config = ConfigDict(frozen=True)

    reason: str = Field(min_length=1, max_length=1000)
    request: HumanInputRequest | None = None


class LoopTerminal(BaseModel):
    model_config = ConfigDict(frozen=True)

    status: Literal["completed", "failed"]
    stop_reason: str = Field(min_length=1, max_length=200)
    final_answer: str | None = None
    final_output: ValidatedFinalOutput | None = None
    error: str | None = Field(default=None, max_length=2000)


class StopHookFeedback(BaseModel):
    model_config = ConfigDict(frozen=True)

    code: str = Field(min_length=1, max_length=120)
    message: str = Field(min_length=1, max_length=1000)
    occurrences: int = Field(default=1, ge=1)


class PendingToolCall(BaseModel):
    """Single canonical pending tool call. Replaces ToolCallPlan-as-pending + old PendingToolCall."""

    plan: ToolCallPlan
    status: Literal["pending", "approved", "denied", "running", "completed", "failed"]
    approval_request_id: str | None = None
    operation_id: str | None = None
    summary: str | None = None

    @property
    def tool_call_id(self) -> str:
        return self.plan.tool_call_id

    @property
    def tool_name(self) -> str:
        return self.plan.tool_name


class ToolCallLedgerEntry(BaseModel):
    """Transcript source for one tool call — no runtime state, just plan + position."""

    plan: ToolCallPlan
    turn: int
    sequence: int


class ToolCallLedger(BaseModel):
    """Bounded ledger of all tool calls for native transcript rebuild.
    Only cleaned when entries are no longer needed for transcript reconstruction.
    """

    entries: list[ToolCallLedgerEntry] = Field(default_factory=list)
    max_entries: int = 128

    def append_plans(self, plans: Iterable[ToolCallPlan], *, turn: int) -> None:
        """Record model-requested calls idempotently; do not store pending state."""
        existing = {entry.plan.tool_call_id for entry in self.entries}
        for plan in plans:
            if plan.tool_call_id in existing:
                continue
            self.entries.append(
                ToolCallLedgerEntry(
                    plan=plan,
                    turn=turn,
                    sequence=len(self.entries),
                )
            )
            existing.add(plan.tool_call_id)

    def trim(self, *, active_tool_call_ids: set[str]) -> None:
        """Remove oldest non-active entries when over max_entries."""
        while len(self.entries) > self.max_entries:
            for index, entry in enumerate(self.entries):
                if entry.plan.tool_call_id not in active_tool_call_ids:
                    self.entries.pop(index)
                    break
            else:
                break


class LoopState(TypedDict):
    task: str
    messages: list[BaseMessage]
    run_config: AgentRunConfig
    iteration: int
    status: LoopStatus
    pending_tool_calls: list[PendingToolCall]  # single-track
    tool_call_ledger: ToolCallLedger  # bounded transcript source
    tool_execution_records: dict[str, ToolExecutionRecord]
    approval_request: HumanInputRequest | None
    approval_response: HumanInputResponse | None
    approved_tool_call_ids: list[str]
    denied_tool_call_ids: list[str]
    tool_results: list[ToolResult]
    working_summary: WorkingSummary | None
    extracted_facts: list[ExtractedFact]
    context_budget: ContextBudgetSnapshot | None
    memory_refs: list[MemoryRef]
    memory_budget: MemoryBudgetSnapshot | None
    memory_warnings: list[str]
    reactive_compact_used: bool
    agent_plan: AgentPlan | None
    plan_events: list[PlanEvent]
    stop_hook_feedback: list[StopHookFeedback]
    stop_hook_warnings: list[StopHookFeedback]
    runtime_diagnostics: list[RuntimeDiagnostic]
    last_model_turn: ModelTurn | None
    final_answer: str | None
    final_output: ValidatedFinalOutput | None
    output_validation_errors: list[dict[str, object]]
    pause: LoopPause | None
    terminal: LoopTerminal | None
    latest_transition: LoopTransition | None
    # ── PR1-like: typed sub-state convergence (dual-write, no deletions) ──
    plan_state: PlanState
    memory_state: MemoryState
    deferred_tool_state: DeferredToolState
    finish_state: FinishState
    # ── PR1: tool discovery state ──
    discovery_active_tools: list[str]
    discovery_active_tool_iterations: dict[str, int]
    discovery_last_candidates: list[dict[str, object]]
    discovery_last_search_query: str
    discovery_search_history: list[dict[str, object]]
    discovery_pinned_tools: list[str]
    active_deferred_tools: list[str]  # backward compat alias
    capability_diagnostics: list[RuntimeDiagnostic]
    # ── File manifest (file-first processing) ──
    file_manifest: FileManifest | None
    # ── Persistent cross-session memory ──
    persistent_memories: list[str]  # selected memory texts for current run
    memory_index: str  # MEMORY.md content (cheap, always loaded)


def create_loop_state(
    *,
    task: str,
    run_config: AgentRunConfig,
    messages: Iterable[BaseMessage] = (),
    pending_tool_calls: Iterable[ToolCallPlan] = (),
    memory_warnings: Iterable[str] = (),
    runtime_diagnostics: Iterable[RuntimeDiagnostic] = (),
    file_manifest: FileManifest | None = None,
) -> LoopState:
    # ── Function-level imports to avoid circular import with substate.py ──
    from rag.agent.loop.substate import (
        DeferredToolState,
        FinishState,
        MemoryState,
        PersistentMemorySnapshot,
        PlanState,
    )

    return {
        "task": task,
        "messages": list(messages),
        "run_config": run_config,
        "iteration": 0,
        "status": "running",
        "pending_tool_calls": [PendingToolCall(plan=call, status="pending") for call in pending_tool_calls],
        "tool_call_ledger": ToolCallLedger() if not pending_tool_calls
        else ToolCallLedger(entries=[
            ToolCallLedgerEntry(plan=call, turn=0, sequence=i)
            for i, call in enumerate(pending_tool_calls)
        ]),
        "tool_execution_records": {},
        "approval_request": None,
        "approval_response": None,
        "approved_tool_call_ids": [],
        "denied_tool_call_ids": [],
        "tool_results": [],
        "working_summary": None,
        "extracted_facts": [],
        "context_budget": None,
        "memory_refs": [],
        "memory_budget": None,
        "memory_warnings": _bounded_unique_strings(
            memory_warnings,
            limit=MAX_LOOP_MEMORY_WARNINGS,
        ),
        "reactive_compact_used": False,
        "agent_plan": None,
        "plan_events": [],
        "stop_hook_feedback": [],
        "stop_hook_warnings": [],
        "runtime_diagnostics": merge_runtime_diagnostics([], runtime_diagnostics),
        "last_model_turn": None,
        "final_answer": None,
        "final_output": None,
        "output_validation_errors": [],
        "pause": None,
        "terminal": None,
        "latest_transition": None,
        # ── PR1: tool discovery state ──
        "discovery_active_tools": [],
        "discovery_active_tool_iterations": {},
        "discovery_last_candidates": [],
        "discovery_last_search_query": "",
        "discovery_search_history": [],
        "discovery_pinned_tools": [],
        "active_deferred_tools": [],
        "capability_diagnostics": [],
        # ── File manifest (file-first processing) ──
        "file_manifest": file_manifest,
        # ── Persistent cross-session memory ──
        "persistent_memories": [],
        "memory_index": "",
        # ── PR1: typed sub-state convergence (dual-write alongside flat fields) ──
        "plan_state": PlanState(
            agent_plan=None,
            plan_events=[],
        ),
        "memory_state": MemoryState(
            working_summary=None,
            extracted_facts=[],
            context_budget=None,
            memory_refs=[],
            memory_budget=None,
            memory_warnings=_bounded_unique_strings(
                memory_warnings,
                limit=MAX_LOOP_MEMORY_WARNINGS,
            ),
            reactive_compact_used=False,
            persistent=PersistentMemorySnapshot(),
        ),
        "deferred_tool_state": DeferredToolState(),
        "finish_state": FinishState(),
    }


def materialize_model_turn(
    draft: ModelTurnDraft,
) -> ModelTurn:
    """Apply tool-call precedence before the strict kernel contract."""

    if draft.tool_calls:
        return ModelTurn(action="execute", tool_calls=draft.tool_calls)
    if draft.action == "finish":
        return ModelTurn(action="finish", final_answer=draft.final_answer)
    return ModelTurn(
        action=draft.action,
        final_answer=draft.final_answer,
        pause_reason=draft.pause_reason,
    )


def replace_latest_transition(
    state: LoopState,
    transition: LoopTransition,
) -> None:
    state["latest_transition"] = transition


def append_stop_hook_feedback(
    state: LoopState,
    feedback: StopHookFeedback,
) -> StopHookFeedback:
    existing = next(
        (
            item
            for item in state["stop_hook_feedback"]
            if item.code == feedback.code and item.message == feedback.message
        ),
        None,
    )
    updated = feedback.model_copy(
        update={"occurrences": (feedback.occurrences if existing is None else existing.occurrences + 1)}
    )
    items = [item for item in state["stop_hook_feedback"] if item is not existing]
    state["stop_hook_feedback"] = [*items, updated][-MAX_STOP_HOOK_FEEDBACK:]
    # Dual-write to typed finish_state sub-state
    fs = state.get("finish_state")
    if fs is not None and hasattr(fs, "model_copy"):
        try:
            state["finish_state"] = fs.model_copy(update={"feedback": list(state["stop_hook_feedback"])})
        except Exception:
            pass
    return updated


def append_stop_hook_warning(
    state: LoopState,
    warning: StopHookFeedback,
) -> StopHookFeedback:
    existing = next(
        (item for item in state["stop_hook_warnings"] if item.code == warning.code and item.message == warning.message),
        None,
    )
    updated = warning.model_copy(
        update={"occurrences": (warning.occurrences if existing is None else existing.occurrences + 1)}
    )
    items = [item for item in state["stop_hook_warnings"] if item is not existing]
    state["stop_hook_warnings"] = [*items, updated][-MAX_STOP_HOOK_FEEDBACK:]
    # Dual-write to typed finish_state sub-state
    fs = state.get("finish_state")
    if fs is not None and hasattr(fs, "model_copy"):
        try:
            state["finish_state"] = fs.model_copy(update={"warnings": list(state["stop_hook_warnings"])})
        except Exception:
            pass
    return updated


def append_memory_warning(state: LoopState, warning: str) -> None:
    state["memory_warnings"] = _bounded_unique_strings(
        [*state["memory_warnings"], warning],
        limit=MAX_LOOP_MEMORY_WARNINGS,
    )
    # Dual-write to typed memory_state sub-state
    ms = state.get("memory_state")
    if ms is not None and hasattr(ms, "model_copy"):
        try:
            state["memory_state"] = ms.model_copy(update={"memory_warnings": list(state["memory_warnings"])})
        except Exception:
            pass  # non-critical sync, don't crash


def append_loop_diagnostic(
    state: LoopState,
    diagnostic: RuntimeDiagnostic,
) -> None:
    state["runtime_diagnostics"] = merge_runtime_diagnostics(
        state["runtime_diagnostics"],
        [diagnostic],
    )


def _bounded_unique_strings(values: Iterable[str], *, limit: int) -> list[str]:
    merged: dict[str, None] = {}
    for value in values:
        normalized = value.strip()
        if not normalized:
            continue
        merged.pop(normalized, None)
        merged[normalized] = None
    return list(merged)[-limit:]


def _nonempty(value: str | None) -> bool:
    return bool(value and value.strip())


__all__ = [
    "MAX_LOOP_MEMORY_WARNINGS",
    "MAX_STOP_HOOK_FEEDBACK",
    "LoopPause",
    "LoopState",
    "LoopStatus",
    "LoopTerminal",
    "LoopTransition",
    "LoopTransitionReason",
    "ModelTurn",
    "ModelTurnDraft",
    "PendingToolCall",
    "StopHookFeedback",
    "ToolCallLedger",
    "ToolCallLedgerEntry",
    "append_loop_diagnostic",
    "append_memory_warning",
    "append_stop_hook_feedback",
    "append_stop_hook_warning",
    "create_loop_state",
    "materialize_model_turn",
    "replace_latest_transition",
]

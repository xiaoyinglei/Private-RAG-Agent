from __future__ import annotations

import asyncio
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Any, Protocol, cast

import aiosqlite
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    ChannelVersions,
    CheckpointMetadata,
    CheckpointTuple,
    empty_checkpoint,
)
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.serde.base import SerializerProtocol
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from rag.agent.compat.goal_contract import GoalCompatibilityConfig
from rag.agent.core.context import AgentRunConfig
from rag.agent.core.human_input import (
    HumanInputRequest,
    HumanInputRequestIdMismatchError,
    HumanInputResponse,
)
from rag.agent.core.runtime_diagnostics import RuntimeDiagnostic
from rag.agent.core.tool_execution import (
    ExecutionRecordWriter,
    ToolExecutionRecord,
    apply_tool_reconciliation,
)
from rag.agent.loop.state import (
    LoopPause,
    LoopState,
    LoopTransition,
)
from rag.agent.loop.substate import (
    DeferredToolState,
    DiscoveryCandidate,
    DiscoveryEvent,
    FinishState,
    MemoryState,
    PersistentMemorySnapshot,
    PlanState,
)

LOOP_CHECKPOINT_NAMESPACE = "agent_loop"
LOOP_COMPATIBILITY_CHANNEL = "loop_compatibility"
LOOP_STATE_CHANNEL = "loop_state"

AGENT_CHECKPOINT_MSGPACK_ALLOWLIST: tuple[tuple[str, ...], ...] = (
    ("rag.agent.core.messages", "ModelMessage"),
    ("rag.agent.core.messages", "ToolCall"),
    ("rag.agent.core.messages", "StopReason"),
    ("rag.agent.core.messages", "ToolUseResult"),
    ("rag.agent.core.context", "AgentRunConfig"),
    ("rag.agent.core.definition", "ToolPolicy"),
    ("rag.agent.core.human_input", "HumanInputRequest"),
    ("rag.agent.core.human_input", "HumanInputResponse"),
    ("rag.agent.core.human_input", "ToolCallSummary"),
    ("rag.agent.core.output_models", "ValidatedFinalOutput"),
    ("rag.agent.core.runtime_diagnostics", "RuntimeDiagnostic"),
    ("rag.agent.core.tool_execution", "ToolBatchRequest"),
    ("rag.agent.core.tool_execution", "ToolBatchResult"),
    ("rag.agent.core.tool_execution", "ToolExecutionRecord"),
    ("rag.agent.core.tool_execution", "ToolExecutionSummary"),
    ("rag.agent.core.turn_contracts", "ToolCallPlan"),
    ("rag.agent.compat.goal_contract", "GoalConstraint"),
    ("rag.agent.compat.goal_contract", "GoalCompatibilityConfig"),
    ("rag.agent.compat.goal_contract", "GoalContractEvaluation"),
    ("rag.agent.compat.goal_contract", "GoalContractIssue"),
    ("rag.agent.compat.goal_contract", "GoalDeliverable"),
    ("rag.agent.compat.goal_contract", "GoalSpec"),
    ("rag.agent.memory.models", "ContextBudgetSnapshot"),
    ("rag.agent.memory.models", "EvictedStateItem"),
    ("rag.agent.memory.models", "ExtractedFact"),
    ("rag.agent.memory.models", "ExternalizedToolOutput"),
    ("rag.agent.memory.models", "MessageBatchPayload"),
    ("rag.agent.memory.models", "MemoryBudgetSnapshot"),
    ("rag.agent.memory.models", "MemoryPolicy"),
    ("rag.agent.memory.models", "MemoryRecord"),
    ("rag.agent.memory.models", "MemoryRef"),
    ("rag.agent.memory.models", "StateChannelReplacement"),
    ("rag.agent.memory.models", "ToolErrorDetailPayload"),
    ("rag.agent.memory.models", "WorkingSummary"),
    ("rag.agent.loop.state", "LoopPause"),
    ("rag.agent.loop.state", "LoopTerminal"),
    ("rag.agent.loop.state", "LoopTransition"),
    ("rag.agent.loop.state", "ModelTurn"),
    ("rag.agent.loop.state", "ModelTurnDraft"),
    # File manifest (file-first processing)
    ("rag.agent.file_manifest", "FileManifest"),
    ("rag.agent.file_manifest", "FileManifestEntry"),
    ("rag.agent.file_manifest", "SheetPreview"),
    ("rag.agent.file_manifest", "ColumnPreview"),
    ("rag.agent.primitive_ops", "StructuredProbeOutput"),
    ("rag.agent.primitive_ops", "StructuredTableProbe"),
    ("rag.agent.primitive_ops", "CandidateHeaderRow"),
    ("rag.agent.loop.state", "StopHookFeedback"),
    # ── PR3: PendingToolCall v2 + ToolCallLedger ──
    ("rag.agent.loop.state", "PendingToolCall"),
    ("rag.agent.loop.state", "ToolCallLedger"),
    ("rag.agent.loop.state", "ToolCallLedgerEntry"),
    # ── PR1: typed sub-state convergence ──
    ("rag.agent.loop.substate", "DeferredToolState"),
    ("rag.agent.loop.substate", "FinishState"),
    ("rag.agent.loop.substate", "MemoryState"),
    ("rag.agent.loop.substate", "PersistentMemorySnapshot"),
    ("rag.agent.loop.substate", "PlanState"),
    ("rag.agent.loop.stop_hooks", "StopHookOutcome"),
    ("rag.agent.loop.stop_hooks", "StopVerdict"),
    ("rag.agent.planning", "AgentPlan"),
    ("rag.agent.planning", "PlanEvent"),
    ("rag.agent.planning", "PlanStep"),
    ("rag.agent.planning", "PlanStepPatch"),
    ("rag.agent.planning", "PlanUpdate"),
    ("rag.agent.primitive_ops", "CandidateHeaderRow"),
    ("rag.agent.primitive_ops", "StructuredProbeOutput"),
    ("rag.agent.primitive_ops", "StructuredTableProbe"),
    ("rag.agent.tools.spec", "ToolError"),
    ("rag.agent.tools.spec", "ToolResult"),
    ("rag.schema.query", "AnswerCitation"),
    ("rag.schema.query", "EvidenceItem"),
    ("rag.schema.query", "RetrievalSignals"),
    ("rag.schema.runtime", "AccessPolicy"),
    ("rag.schema.runtime", "RuntimeMode"),
)


__all__ = [
    # Checkpoint store API
    "CheckpointPersistenceError",
    "CheckpointStore",
    "LangGraphCheckpointStore",
    "agent_checkpoint_serde",
    "create_agent_checkpointer",
    "aclose_agent_checkpointer",
    # Migration helpers (used by Tasks 6, 8)
    "_migrate_legacy_state",
    "_digest_text",
    "_migrate_discovery_candidates",
    "_migrate_discovery_events",
    "_string_list",
]


def agent_checkpoint_serde() -> SerializerProtocol:
    return JsonPlusSerializer(
        allowed_msgpack_modules=AGENT_CHECKPOINT_MSGPACK_ALLOWLIST,
    )


class CheckpointPersistenceError(RuntimeError):
    """A loop transition could not be durably persisted."""


class CheckpointStore(ExecutionRecordWriter, Protocol):
    async def load_latest(self) -> LoopState | None: ...

    async def load_for_resume(self) -> LoopState | None: ...

    async def save_snapshot(
        self,
        state: LoopState,
        *,
        reason: str,
    ) -> None: ...

    async def apply_human_response(
        self,
        response: HumanInputResponse,
    ) -> LoopState: ...


class LangGraphCheckpointStore:
    """Persist loop snapshots as one coarse channel in a dedicated namespace."""

    durable = True

    def __init__(
        self,
        checkpointer: BaseCheckpointSaver[str],
        *,
        run_config: AgentRunConfig,
        compatibility_config: GoalCompatibilityConfig | None = None,
    ) -> None:
        self._checkpointer = checkpointer
        self._run_config = run_config
        self._compatibility_config = compatibility_config or GoalCompatibilityConfig()
        self._state: LoopState | None = None
        self._lock = asyncio.Lock()

    @property
    def compatibility_config(self) -> GoalCompatibilityConfig:
        return self._compatibility_config.model_copy(deep=True)

    def load_latest_sync(self) -> LoopState | None:
        try:
            checkpoint_tuple = self._checkpointer.get_tuple(self._base_config())
        except Exception as exc:
            raise CheckpointPersistenceError(f"failed to load loop checkpoint: {exc}") from exc
        if checkpoint_tuple is None:
            self._state = None
            return None
        self._state = self._restore_checkpoint_tuple(checkpoint_tuple)
        return deepcopy(self._state)

    async def load_latest(self) -> LoopState | None:
        try:
            checkpoint_tuple = await self._checkpointer.aget_tuple(self._base_config())
        except Exception as exc:
            raise CheckpointPersistenceError(f"failed to load loop checkpoint: {exc}") from exc
        if checkpoint_tuple is None:
            self._state = None
            return None
        self._state = self._restore_checkpoint_tuple(checkpoint_tuple)
        return deepcopy(self._state)

    async def load_for_resume(self) -> LoopState | None:
        state = await self.load_latest()
        if state is None:
            return None

        ambiguous = [
            record
            for record in state["tool_execution_records"].values()
            if not record.idempotent and record.status in {"started", "unknown"}
        ]
        if not ambiguous:
            return state

        for record in ambiguous:
            state["tool_execution_records"][record.tool_call_id] = record.model_copy(update={"status": "unknown"})
        request = self.reconciliation_request(ambiguous[0])
        state["status"] = "paused"
        state["approval_request"] = request
        state["pause"] = LoopPause(
            reason="A non-idempotent tool outcome is ambiguous.",
            request=request,
        )
        state["latest_transition"] = LoopTransition(
            reason="approval_required",
            iteration=state["iteration"],
            detail={
                "tool_call_id": ambiguous[0].tool_call_id,
                "execution_status": "unknown",
            },
        )
        await self.save_snapshot(state, reason="recovery_reconciliation")
        return deepcopy(state)

    async def save_snapshot(
        self,
        state: LoopState,
        *,
        reason: str,
    ) -> None:
        async with self._lock:
            await self._save_snapshot_unlocked(state, reason=reason)

    async def write_execution_record(
        self,
        record: ToolExecutionRecord,
    ) -> None:
        async with self._lock:
            state = deepcopy(self._state) if self._state is not None else await self._load_latest_unlocked()
            if state is None:
                raise CheckpointPersistenceError("cannot persist an execution record before loop state exists")
            state["tool_execution_records"][record.tool_call_id] = record
            state["latest_transition"] = LoopTransition(
                reason="tool_execution",
                iteration=state["iteration"],
                detail={
                    "tool_call_id": record.tool_call_id,
                    "execution_status": record.status,
                },
            )
            await self._save_snapshot_unlocked(
                state,
                reason=f"tool_{record.status}",
            )

    async def apply_human_response(
        self,
        response: HumanInputResponse,
    ) -> LoopState:
        async with self._lock:
            state = await self._load_latest_unlocked()
            if state is None or state["pause"] is None:
                raise CheckpointPersistenceError("no paused loop checkpoint is available")
            request = state["pause"].request
            if request is None:
                raise CheckpointPersistenceError("paused loop checkpoint has no typed human request")
            if response.request_id != request.request_id:
                raise HumanInputRequestIdMismatchError(
                    f"Response request_id={response.request_id!r} does not match "
                    f"current request_id={request.request_id!r}"
                )

            state["approval_response"] = response
            if request.kind == "tool_reconciliation":
                tool_call_id = request.context.get("tool_call_id")
                if not isinstance(tool_call_id, str):
                    raise CheckpointPersistenceError("tool reconciliation request is missing tool_call_id")
                record = state["tool_execution_records"].get(tool_call_id)
                if record is None:
                    raise CheckpointPersistenceError(f"execution record not found for {tool_call_id}")
                state["tool_execution_records"][tool_call_id] = apply_tool_reconciliation(record, response)
            elif request.kind == "tool_approval":
                state["approved_tool_call_ids"] = list(
                    dict.fromkeys(
                        [
                            *state["approved_tool_call_ids"],
                            *response.approved_tool_call_ids,
                        ]
                    )
                )
                state["denied_tool_call_ids"] = list(
                    dict.fromkeys(
                        [
                            *state["denied_tool_call_ids"],
                            *response.denied_tool_call_ids,
                        ]
                    )
                )

            state["status"] = "running"
            state["approval_request"] = None
            state["pause"] = None
            state["latest_transition"] = LoopTransition(
                reason="next_turn",
                iteration=state["iteration"],
                detail={"human_decision": response.decision},
            )
            await self._save_snapshot_unlocked(
                state,
                reason="human_response",
            )
            return deepcopy(state)

    def reconciliation_request(
        self,
        record: ToolExecutionRecord,
    ) -> HumanInputRequest:
        return HumanInputRequest(
            request_id=f"hir_{_uuid_suffix()}",
            kind="tool_reconciliation",
            question=(f"工具 {record.tool_name} 的外部副作用状态不明确，请选择恢复方式。"),
            context={
                "tool_call_id": record.tool_call_id,
                "tool_name": record.tool_name,
                "operation_id": record.operation_id,
                "execution_status": record.status,
            },
            options=[
                "mark_completed",
                "mark_failed",
                "retry_new_operation",
            ],
        )

    async def _load_latest_unlocked(self) -> LoopState | None:
        try:
            checkpoint_tuple = await self._checkpointer.aget_tuple(self._base_config())
        except Exception as exc:
            raise CheckpointPersistenceError(f"failed to load loop checkpoint: {exc}") from exc
        if checkpoint_tuple is None:
            self._state = None
            return None
        self._state = self._restore_checkpoint_tuple(checkpoint_tuple)
        return deepcopy(self._state)

    async def _save_snapshot_unlocked(
        self,
        state: LoopState,
        *,
        reason: str,
    ) -> None:
        snapshot = deepcopy(state)
        try:
            previous = await self._checkpointer.aget_tuple(self._base_config())
            current_version = cast(
                str | None,
                (None if previous is None else previous.checkpoint["channel_versions"].get(LOOP_STATE_CHANNEL)),
            )
            version = self._checkpointer.get_next_version(
                current_version,
                None,
            )
            compatibility = self._compatibility_config
            if previous is not None and compatibility.goal_spec is None:
                raw_compatibility = previous.checkpoint["channel_values"].get(LOOP_COMPATIBILITY_CHANNEL)
                if raw_compatibility is not None:
                    compatibility = _normalize_compatibility_config(raw_compatibility)
                    self._compatibility_config = compatibility
            checkpoint = empty_checkpoint()
            checkpoint["channel_values"] = {
                LOOP_STATE_CHANNEL: snapshot,
            }
            checkpoint["channel_versions"] = {
                LOOP_STATE_CHANNEL: version,
            }
            updated_channels = [LOOP_STATE_CHANNEL]
            checkpoint["updated_channels"] = updated_channels
            new_versions: ChannelVersions = {LOOP_STATE_CHANNEL: version}
            if compatibility.goal_spec is not None:
                compatibility_version = self._checkpointer.get_next_version(
                    cast(
                        str | None,
                        (
                            None
                            if previous is None
                            else previous.checkpoint["channel_versions"].get(LOOP_COMPATIBILITY_CHANNEL)
                        ),
                    ),
                    None,
                )
                checkpoint["channel_values"][LOOP_COMPATIBILITY_CHANNEL] = compatibility
                checkpoint["channel_versions"][LOOP_COMPATIBILITY_CHANNEL] = compatibility_version
                updated_channels.append(LOOP_COMPATIBILITY_CHANNEL)
                new_versions[LOOP_COMPATIBILITY_CHANNEL] = compatibility_version
            config = previous.config if previous is not None else self._base_config()
            metadata = cast(
                CheckpointMetadata,
                {
                    "source": "loop",
                    "step": snapshot["iteration"],
                    "parents": {},
                    "reason": reason,
                },
            )
            await self._checkpointer.aput(
                config,
                checkpoint,
                metadata,
                new_versions,
            )
        except Exception as exc:
            raise CheckpointPersistenceError(f"failed to persist loop checkpoint: {exc}") from exc
        self._state = snapshot

    def _restore_checkpoint_tuple(
        self,
        checkpoint_tuple: CheckpointTuple,
    ) -> LoopState:
        channel_values = checkpoint_tuple.checkpoint["channel_values"]
        raw_state = channel_values.get(LOOP_STATE_CHANNEL)
        if not isinstance(raw_state, dict):
            raise CheckpointPersistenceError("loop checkpoint is missing the loop_state channel")
        self._compatibility_config = _normalize_compatibility_config(channel_values.get(LOOP_COMPATIBILITY_CHANNEL))
        return _normalize_loaded_state(cast(LoopState, deepcopy(raw_state)))

    def _base_config(self) -> RunnableConfig:
        return cast(
            RunnableConfig,
            {
                "configurable": {
                    "thread_id": self._run_config.thread_id,
                    "checkpoint_ns": LOOP_CHECKPOINT_NAMESPACE,
                }
            },
        )


def _uuid_suffix() -> str:
    from uuid import uuid4

    return uuid4().hex[:12]


def _normalize_loaded_state(state: LoopState) -> LoopState:
    run_config = state["run_config"]
    if not isinstance(run_config.source_scope, tuple):
        state["run_config"] = replace(
            run_config,
            source_scope=tuple(run_config.source_scope),
        )
    # Backfill PR0/PR1 fields missing from older checkpoints
    state.setdefault("discovery_active_tools", [])
    state.setdefault("discovery_active_tool_iterations", {})
    state.setdefault("discovery_last_candidates", [])
    state.setdefault("discovery_last_search_query", "")
    state.setdefault("discovery_search_history", [])
    state.setdefault("discovery_pinned_tools", [])
    state.setdefault("active_deferred_tools", [])
    state.setdefault("capability_diagnostics", [])
    # ── PR1: migrate legacy flat fields into typed sub-states ──
    state = _migrate_legacy_state(cast(dict[str, Any], state))
    return state


_DEPRECATED_STATE_FIELDS = frozenset({
    "retrieval_signals", "retrieval_signals_debug",
    "evidence", "citations", "evidence_refs",
    "answer_candidates", "computation_results",
    "structured_observations", "context_units",
    "context_bindings", "locators", "asset_refs",
})


def _migrate_legacy_state(raw: dict[str, Any]) -> LoopState:
    """Populate new sub-state models from legacy flat fields.

    Reads old flat fields and writes them into the corresponding
    sub-state container.  Leaves the old flat fields intact so that
    existing callers continue to work (dual-read safe).

    This is called from ``_normalize_loaded_state`` for every
    checkpoint load, including old checkpoints that lack the new
    sub-state keys.
    """
    state = dict(raw)

    # ── PR3: drop legacy loop_messages and tool_result_store ──
    if state.get("loop_messages"):
        state.setdefault("runtime_diagnostics", []).append(
            RuntimeDiagnostic(
                code="legacy_loop_messages_dropped",
                component="checkpoint_migration",
                message="Old loop_messages were dropped; transcript is rebuilt from tool_call_ledger and tool_results.",
                severity="warning",
            )
        )
    state.pop("loop_messages", None)
    state.pop("tool_result_store", None)

    # ── PlanState ──
    state.setdefault(
        "plan_state",
        PlanState(
            agent_plan=state.get("agent_plan"),
            plan_events=list(state.get("plan_events", [])),
        ),
    )

    # ── MemoryState ──
    state.setdefault(
        "memory_state",
        MemoryState(
            working_summary=state.get("working_summary"),
            extracted_facts=list(state.get("extracted_facts", [])),
            context_budget=state.get("context_budget"),
            memory_refs=list(state.get("memory_refs", [])),
            memory_budget=state.get("memory_budget"),
            memory_warnings=list(state.get("memory_warnings", [])),
            reactive_compact_used=bool(state.get("reactive_compact_used", False)),
            persistent=PersistentMemorySnapshot(
                index_digest=_digest_text(state.get("memory_index", "")),
                selected_count=len(state.get("persistent_memories", [])),
            ),
        ),
    )

    # ── DeferredToolState ──
    state.setdefault(
        "deferred_tool_state",
        DeferredToolState(
            active_tools=list(state.get("discovery_active_tools", [])),
            active_tool_iterations=dict(state.get("discovery_active_tool_iterations", {})),
            last_candidates=_migrate_discovery_candidates(state.get("discovery_last_candidates", [])),
            last_search_query=str(state.get("discovery_last_search_query", "")),
            search_history=_migrate_discovery_events(state.get("discovery_search_history", [])),
            pinned_tools=list(state.get("discovery_pinned_tools", [])),
            capability_diagnostics=list(state.get("capability_diagnostics", [])),
        ),
    )

    # ── FinishState ──
    state.setdefault(
        "finish_state",
        FinishState(
            feedback=list(state.get("stop_hook_feedback", [])),
            warnings=list(state.get("stop_hook_warnings", [])),
        ),
    )

    # ── PR3: migrate legacy pending_loop_tool_calls into pending_tool_calls ──
    if "pending_loop_tool_calls" in raw:
        from rag.agent.core.turn_contracts import ToolCallPlan
        from rag.agent.loop.state import PendingToolCall as NewPendingToolCall

        legacy_pending = raw.get("pending_loop_tool_calls", [])
        existing_pending = list(state.get("pending_tool_calls", []))
        migrated: list[NewPendingToolCall] = []
        for call in legacy_pending:
            if isinstance(call, dict):
                tc_id = call.get("tool_call_id", "")
                tc_name = call.get("tool_name", "")
                args = call.get("arguments", {})
                status = call.get("status", "pending")
                summary = call.get("summary")
            else:
                tc_id = getattr(call, "tool_call_id", "")
                tc_name = getattr(call, "tool_name", "")
                args = getattr(call, "arguments", {})
                status = getattr(call, "status", "pending")
                summary = getattr(call, "summary", None)
            if tc_id:
                migrated.append(
                    NewPendingToolCall(
                        plan=ToolCallPlan(
                            tool_call_id=tc_id,
                            tool_name=tc_name,
                            arguments=args,
                        ),
                        status=status,
                        summary=summary,
                    )
                )
        if migrated:
            state["pending_tool_calls"] = existing_pending + migrated
        del state["pending_loop_tool_calls"]

    # ── PR3: normalize pending_tool_calls to PendingToolCall v2 ──
    raw_pending = state.get("pending_tool_calls", [])
    if raw_pending:
        from rag.agent.core.turn_contracts import ToolCallPlan as NewToolCallPlan
        from rag.agent.loop.state import PendingToolCall as NewPendingToolCall

        normalized: list[NewPendingToolCall] = []
        for item in raw_pending:
            if isinstance(item, NewPendingToolCall):
                normalized.append(item)
            elif isinstance(item, NewToolCallPlan):
                # Legacy bare ToolCallPlan → wrap
                normalized.append(NewPendingToolCall(plan=item, status="pending"))
            elif isinstance(item, dict) and "plan" in item:
                # Possibly serialized PendingToolCall
                try:
                    normalized.append(NewPendingToolCall.model_validate(item))
                except Exception:
                    pass
        state["pending_tool_calls"] = normalized

    # ── PR3: backfill tool_call_ledger from pending_tool_calls ──
    from rag.agent.loop.state import ToolCallLedger as NewToolCallLedger

    state.setdefault("tool_call_ledger", NewToolCallLedger())
    ledger = state["tool_call_ledger"]
    if isinstance(ledger, NewToolCallLedger) and not ledger.entries:
        pending = state.get("pending_tool_calls", [])
        if pending:
            ledger.append_plans(
                [p.plan for p in pending if isinstance(p, NewPendingToolCall)],
                turn=state.get("iteration", 0),
            )

    # ── PR3: drop deprecated state fields after sub-states are populated ──
    for key in _DEPRECATED_STATE_FIELDS:
        state.pop(key, None)

    return cast(LoopState, state)


def _digest_text(text: str, *, max_chars: int = 500) -> str:
    """Truncate text to a bounded digest for PersistentMemorySnapshot."""
    stripped = text.strip()
    if len(stripped) <= max_chars:
        return stripped
    return stripped[:max_chars].rstrip() + "..."


def _migrate_discovery_candidates(
    raw: list[dict[str, object]],
) -> list[DiscoveryCandidate]:
    """Convert legacy dict-based candidates to typed DiscoveryCandidate."""
    candidates: list[DiscoveryCandidate] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        try:
            candidates.append(
                DiscoveryCandidate(
                    name=str(item.get("name", "")),
                    description=str(item.get("description", "")),
                    reason=str(item.get("reason", "")),
                    metadata={k: v for k, v in item.items() if k not in {"name", "description", "reason"}},
                )
            )
        except Exception:
            continue  # non-critical migration, skip uncoercible item
    return candidates


def _migrate_discovery_events(
    raw: list[dict[str, object]],
) -> list[DiscoveryEvent]:
    """Convert legacy dict-based search events to typed DiscoveryEvent."""
    events: list[DiscoveryEvent] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        try:
            events.append(
                DiscoveryEvent(
                    query=str(item.get("query", "")),
                    candidates=_string_list(item.get("candidates")),
                    activated=_string_list(item.get("activated")),
                )
            )
        except Exception:
            continue  # non-critical migration, skip uncoercible item
    return events


def _string_list(value: object) -> list[str]:
    """Coerce a value to a list of strings safely."""
    if isinstance(value, list):
        return [str(v) for v in value if v is not None]
    return []


def _normalize_compatibility_config(
    value: object,
) -> GoalCompatibilityConfig:
    if value is None:
        return GoalCompatibilityConfig()
    if isinstance(value, GoalCompatibilityConfig):
        return value.model_copy(deep=True)
    try:
        return GoalCompatibilityConfig.model_validate(value)
    except Exception as exc:
        raise CheckpointPersistenceError("loop checkpoint has invalid compatibility metadata") from exc


def create_agent_checkpointer(checkpoint_db: Path | str | None) -> BaseCheckpointSaver[str]:
    if checkpoint_db is None:
        return MemorySaver(serde=agent_checkpoint_serde())

    path = Path(checkpoint_db)
    path.parent.mkdir(parents=True, exist_ok=True)
    return AsyncSqliteSaver(
        aiosqlite.connect(str(path)),
        serde=agent_checkpoint_serde(),
    )


async def aclose_agent_checkpointer(checkpointer: BaseCheckpointSaver[str]) -> None:
    connection = getattr(checkpointer, "conn", None)
    if connection is not None and hasattr(connection, "close"):
        await connection.close()

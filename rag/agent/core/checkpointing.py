from __future__ import annotations

import asyncio
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Protocol, cast

import aiosqlite
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    CheckpointMetadata,
    empty_checkpoint,
)
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.serde.base import SerializerProtocol
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from rag.agent.core.context import AgentRunConfig
from rag.agent.core.human_input import (
    HumanInputRequest,
    HumanInputRequestIdMismatchError,
    HumanInputResponse,
)
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

LOOP_CHECKPOINT_NAMESPACE = "agent_loop"
LOOP_STATE_CHANNEL = "loop_state"

AGENT_CHECKPOINT_MSGPACK_ALLOWLIST: tuple[tuple[str, ...], ...] = (
    ("rag.agent.core.context", "AgentRunConfig"),
    ("rag.agent.core.definition", "ToolPolicy"),
    ("rag.agent.core.human_input", "HumanInputRequest"),
    ("rag.agent.core.human_input", "HumanInputResponse"),
    ("rag.agent.core.human_input", "ToolCallSummary"),
    ("rag.agent.core.output_models", "ValidatedFinalOutput"),
    ("rag.agent.core.observations", "AnswerCandidate"),
    ("rag.agent.core.observations", "ComputationResult"),
    ("rag.agent.core.observations", "ContextBinding"),
    ("rag.agent.core.observations", "ContextUnit"),
    ("rag.agent.core.observations", "EvidenceRef"),
    ("rag.agent.core.observations", "ObservationBatch"),
    ("rag.agent.core.observations", "ObservationError"),
    ("rag.agent.core.observations", "StructuredObservation"),
    ("rag.agent.core.tool_execution", "ToolBatchRequest"),
    ("rag.agent.core.tool_execution", "ToolBatchResult"),
    ("rag.agent.core.tool_execution", "ToolExecutionRecord"),
    ("rag.agent.core.tool_execution", "ToolExecutionSummary"),
    ("rag.agent.core.finalization", "FinalizationEvent"),
    ("rag.agent.goal_runtime", "AnswerCandidate"),
    ("rag.agent.goal_runtime", "ComputationResult"),
    ("rag.agent.goal_runtime", "ContextBinding"),
    ("rag.agent.goal_runtime", "ContextUnit"),
    ("rag.agent.goal_runtime", "EvidenceRef"),
    ("rag.agent.goal_runtime", "GoalConflict"),
    ("rag.agent.goal_runtime", "GoalContractConstraintHint"),
    ("rag.agent.goal_runtime", "GoalContractHint"),
    ("rag.agent.goal_runtime", "GoalConstraint"),
    ("rag.agent.goal_runtime", "GoalDeliverable"),
    ("rag.agent.goal_runtime", "GoalGap"),
    ("rag.agent.goal_runtime", "GoalInitializationHints"),
    ("rag.agent.goal_runtime", "GoalSpec"),
    ("rag.agent.goal_runtime", "SatisfactionReport"),
    ("rag.agent.goal_runtime", "StructuredObservation"),
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
    ("rag.agent.loop.state", "StopHookFeedback"),
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
    ("rag.agent.state", "ThinkOutput"),
    ("rag.agent.state", "ToolCallPlan"),
    ("rag.agent.tools.spec", "ToolError"),
    ("rag.agent.tools.spec", "ToolResult"),
    ("rag.schema.query", "AnswerCitation"),
    ("rag.schema.query", "EvidenceItem"),
    ("rag.schema.query", "RetrievalSignals"),
    ("rag.schema.runtime", "AccessPolicy"),
    ("rag.schema.runtime", "RuntimeMode"),
)


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
    ) -> None:
        self._checkpointer = checkpointer
        self._run_config = run_config
        self._state: LoopState | None = None
        self._lock = asyncio.Lock()

    async def load_latest(self) -> LoopState | None:
        try:
            checkpoint_tuple = await self._checkpointer.aget_tuple(
                self._base_config()
            )
        except Exception as exc:
            raise CheckpointPersistenceError(
                f"failed to load loop checkpoint: {exc}"
            ) from exc
        if checkpoint_tuple is None:
            self._state = None
            return None
        raw_state = checkpoint_tuple.checkpoint["channel_values"].get(
            LOOP_STATE_CHANNEL
        )
        if not isinstance(raw_state, dict):
            raise CheckpointPersistenceError(
                "loop checkpoint is missing the loop_state channel"
            )
        self._state = _normalize_loaded_state(
            cast(LoopState, deepcopy(raw_state))
        )
        return deepcopy(self._state)

    async def load_for_resume(self) -> LoopState | None:
        state = await self.load_latest()
        if state is None:
            return None

        ambiguous = [
            record
            for record in state["tool_execution_records"].values()
            if not record.idempotent
            and record.status in {"started", "unknown"}
        ]
        if not ambiguous:
            return state

        for record in ambiguous:
            state["tool_execution_records"][record.tool_call_id] = (
                record.model_copy(update={"status": "unknown"})
            )
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
            state = (
                deepcopy(self._state)
                if self._state is not None
                else await self._load_latest_unlocked()
            )
            if state is None:
                raise CheckpointPersistenceError(
                    "cannot persist an execution record before loop state exists"
                )
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
                raise CheckpointPersistenceError(
                    "no paused loop checkpoint is available"
                )
            request = state["pause"].request
            if request is None:
                raise CheckpointPersistenceError(
                    "paused loop checkpoint has no typed human request"
                )
            if response.request_id != request.request_id:
                raise HumanInputRequestIdMismatchError(
                    f"Response request_id={response.request_id!r} does not match "
                    f"current request_id={request.request_id!r}"
                )

            state["approval_response"] = response
            if request.kind == "tool_reconciliation":
                tool_call_id = request.context.get("tool_call_id")
                if not isinstance(tool_call_id, str):
                    raise CheckpointPersistenceError(
                        "tool reconciliation request is missing tool_call_id"
                    )
                record = state["tool_execution_records"].get(tool_call_id)
                if record is None:
                    raise CheckpointPersistenceError(
                        f"execution record not found for {tool_call_id}"
                    )
                state["tool_execution_records"][tool_call_id] = (
                    apply_tool_reconciliation(record, response)
                )
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
            question=(
                f"工具 {record.tool_name} 的外部副作用状态不明确，"
                "请选择恢复方式。"
            ),
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
            checkpoint_tuple = await self._checkpointer.aget_tuple(
                self._base_config()
            )
        except Exception as exc:
            raise CheckpointPersistenceError(
                f"failed to load loop checkpoint: {exc}"
            ) from exc
        if checkpoint_tuple is None:
            self._state = None
            return None
        raw_state = checkpoint_tuple.checkpoint["channel_values"].get(
            LOOP_STATE_CHANNEL
        )
        if not isinstance(raw_state, dict):
            raise CheckpointPersistenceError(
                "loop checkpoint is missing the loop_state channel"
            )
        self._state = _normalize_loaded_state(
            cast(LoopState, deepcopy(raw_state))
        )
        return deepcopy(self._state)

    async def _save_snapshot_unlocked(
        self,
        state: LoopState,
        *,
        reason: str,
    ) -> None:
        snapshot = deepcopy(state)
        try:
            previous = await self._checkpointer.aget_tuple(
                self._base_config()
            )
            current_version = cast(
                str | None,
                (
                    None
                    if previous is None
                    else previous.checkpoint["channel_versions"].get(
                        LOOP_STATE_CHANNEL
                    )
                ),
            )
            version = self._checkpointer.get_next_version(
                current_version,
                None,
            )
            checkpoint = empty_checkpoint()
            checkpoint["channel_values"] = {
                LOOP_STATE_CHANNEL: snapshot,
            }
            checkpoint["channel_versions"] = {
                LOOP_STATE_CHANNEL: version,
            }
            checkpoint["updated_channels"] = [LOOP_STATE_CHANNEL]
            config = (
                previous.config
                if previous is not None
                else self._base_config()
            )
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
                {LOOP_STATE_CHANNEL: version},
            )
        except Exception as exc:
            raise CheckpointPersistenceError(
                f"failed to persist loop checkpoint: {exc}"
            ) from exc
        self._state = snapshot

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
    return state


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

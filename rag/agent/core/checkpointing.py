from __future__ import annotations

import asyncio
import sqlite3
from collections.abc import AsyncIterator, Iterator, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Protocol, cast

import aiosqlite
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    empty_checkpoint,
)
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.serde.base import SerializerProtocol
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from rag.agent.core.context import AgentRunConfig
from rag.agent.core.goal_contract import GoalCompatibilityConfig
from rag.agent.core.human_input import (
    HumanInputRequest,
    HumanInputRequestIdMismatchError,
    HumanInputResponse,
)
from rag.agent.core.messages import (
    ModelMessage,
    canonical_json_text,
    model_message_payload,
    snapshot_model_message,
)
from rag.agent.core.messages import (
    ToolCall as ModelToolCall,
)
from rag.agent.core.model_request import (
    ModelCallRecord,
    model_call_record_payload,
)
from rag.agent.core.runtime_diagnostics import RuntimeDiagnostic
from rag.agent.core.tool_execution import (
    ExecutionRecordWriter,
    ToolExecutionRecord,
    apply_tool_reconciliation,
)
from rag.agent.core.turn_contracts import (
    ToolManifest,
    ToolManifestDriftDecision,
    ToolManifestDriftStatus,
    ToolManifestEntry,
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
from rag.agent.tools.tool import (
    JsonValue,
    ToolCallOrigin,
)
from rag.agent.tools.tool import (
    ToolCall as CanonicalToolCall,
)
from rag.schema.llm import LLMUsage

LOOP_CHECKPOINT_NAMESPACE = "agent_loop"
LOOP_COMPATIBILITY_CHANNEL = "loop_compatibility"
LOOP_STATE_CHANNEL = "loop_state"
TOOL_CHECKPOINT_FORMAT_VERSION = 2


@dataclass(frozen=True, slots=True)
class CanonicalToolCheckpoint:
    context_revision: str
    prompt_revision: str
    transcript: tuple[ModelMessage, ...]
    manifest: ToolManifest
    tool_calls: tuple[CanonicalToolCall, ...] = ()
    pending_tool_calls: tuple[CanonicalToolCall, ...] = ()
    paused_tool_calls: tuple[CanonicalToolCall, ...] = ()
    model_call_records: tuple[ModelCallRecord, ...] = ()
    legacy_migrated: bool = False
    format_version: int = field(
        default=TOOL_CHECKPOINT_FORMAT_VERSION,
        init=False,
    )

    def __post_init__(self) -> None:
        _checkpoint_string(self.context_revision, field_name="context_revision")
        _checkpoint_string(self.prompt_revision, field_name="prompt_revision")
        if not isinstance(self.manifest, ToolManifest):
            raise TypeError("manifest must be a ToolManifest")
        transcript = tuple(
            snapshot_model_message(message)
            for message in self.transcript
        )
        pending = _checkpoint_tool_calls(
            self.pending_tool_calls,
            field_name="pending_tool_calls",
        )
        paused = _checkpoint_tool_calls(
            self.paused_tool_calls,
            field_name="paused_tool_calls",
        )
        calls = _checkpoint_tool_calls(
            self.tool_calls or (*pending, *paused),
            field_name="tool_calls",
        )
        pending_ids = {call.tool_call_id for call in pending}
        paused_ids = {call.tool_call_id for call in paused}
        if pending_ids & paused_ids:
            raise ValueError("a tool call cannot be both pending and paused")
        calls_by_id = {call.tool_call_id: call for call in calls}
        dependent_calls = (*pending, *paused)
        if any(call.tool_call_id not in calls_by_id for call in dependent_calls):
            raise ValueError("pending and paused tool calls must exist in tool_calls")
        if any(
            calls_by_id[call.tool_call_id] != call
            for call in dependent_calls
        ):
            raise ValueError("pending and paused tool calls must exactly match tool_calls")
        records: list[ModelCallRecord] = []
        for record in self.model_call_records:
            if not isinstance(record, ModelCallRecord):
                raise TypeError("model_call_records must contain ModelCallRecord values")
            records.append(
                ModelCallRecord(
                    request_id=record.request_id,
                    prompt_revision=record.prompt_revision,
                    toolset_revision=record.toolset_revision,
                    provider_wire_hash=record.provider_wire_hash,
                    usage=record.usage,
                )
            )
        if type(self.legacy_migrated) is not bool:
            raise TypeError("legacy_migrated must be a bool")
        object.__setattr__(self, "transcript", transcript)
        object.__setattr__(self, "tool_calls", calls)
        object.__setattr__(self, "pending_tool_calls", pending)
        object.__setattr__(self, "paused_tool_calls", paused)
        object.__setattr__(self, "model_call_records", tuple(records))

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
    ("rag.agent.core.runtime_diagnostics", "ToolCallMetrics"),
    ("rag.agent.core.runtime_diagnostics", "AgentLatencyProfile"),
    ("rag.agent.core.tool_execution", "ToolBatchRequest"),
    ("rag.agent.core.tool_execution", "ToolBatchResult"),
    ("rag.agent.core.tool_execution", "ToolExecutionRecord"),
    ("rag.agent.core.tool_execution", "ToolExecutionSummary"),
    ("rag.agent.core.turn_contracts", "ToolCallPlan"),
    ("rag.agent.core.goal_contract", "GoalConstraint"),
    ("rag.agent.core.goal_contract", "GoalCompatibilityConfig"),
    ("rag.agent.core.goal_contract", "GoalContractEvaluation"),
    ("rag.agent.core.goal_contract", "GoalContractIssue"),
    ("rag.agent.core.goal_contract", "GoalDeliverable"),
    ("rag.agent.core.goal_contract", "GoalSpec"),
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
    ("rag.agent.skills.models", "LoadedSkill"),
    ("rag.agent.skills.models", "LoadedSkillRef"),
    ("rag.agent.skills.models", "SkillInvocation"),
    ("rag.agent.skills.models", "SkillManifest"),
    ("rag.agent.skills.models", "SkillSource"),
    ("rag.agent.skills.models", "SkillState"),
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
    "CanonicalToolCheckpoint",
    "CheckpointPersistenceError",
    "CheckpointStore",
    "LangGraphCheckpointStore",
    "agent_checkpoint_serde",
    "create_agent_checkpointer",
    "aclose_agent_checkpointer",
    "decode_legacy_tool_state_v1",
    "decode_tool_checkpoint",
    "encode_tool_checkpoint",
    "reconcile_tool_manifest",
    "TOOL_CHECKPOINT_FORMAT_VERSION",
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


class LazyAsyncSqliteSaver(BaseCheckpointSaver[str]):
    """SQLite checkpointer that creates async resources inside the active loop."""

    def __init__(
        self,
        path: Path,
        *,
        serde: SerializerProtocol,
    ) -> None:
        super().__init__(serde=serde)
        self._path = path
        self._async_conn: aiosqlite.Connection | None = None
        self._async_saver: AsyncSqliteSaver | None = None

    async def _aget_saver(self) -> AsyncSqliteSaver:
        if self._async_saver is None:
            self._async_conn = await aiosqlite.connect(str(self._path))
            self._async_saver = AsyncSqliteSaver(
                self._async_conn,
                serde=self.serde,
            )
        return self._async_saver

    def _sync_saver(self) -> SqliteSaver:
        conn = sqlite3.connect(str(self._path), check_same_thread=False)
        return SqliteSaver(conn, serde=self.serde)

    def get_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        saver = self._sync_saver()
        try:
            return saver.get_tuple(config)
        finally:
            saver.conn.close()

    async def aget_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        saver = await self._aget_saver()
        return await saver.aget_tuple(config)

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        saver = self._sync_saver()
        try:
            return saver.put(config, checkpoint, metadata, new_versions)
        finally:
            saver.conn.close()

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        saver = await self._aget_saver()
        return await saver.aput(config, checkpoint, metadata, new_versions)

    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        saver = self._sync_saver()
        try:
            saver.put_writes(config, writes, task_id, task_path)
        finally:
            saver.conn.close()

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        saver = await self._aget_saver()
        await saver.aput_writes(config, writes, task_id, task_path)

    def list(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> Iterator[CheckpointTuple]:
        saver = self._sync_saver()
        try:
            yield from saver.list(config, filter=filter, before=before, limit=limit)
        finally:
            saver.conn.close()

    async def alist(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> AsyncIterator[CheckpointTuple]:
        saver = await self._aget_saver()
        async for item in saver.alist(
            config,
            filter=filter,
            before=before,
            limit=limit,
        ):
            yield item

    async def aclose(self) -> None:
        if self._async_conn is not None:
            await self._async_conn.close()
            self._async_conn = None
            self._async_saver = None


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


def encode_tool_checkpoint(
    checkpoint: CanonicalToolCheckpoint,
) -> dict[str, object]:
    """Encode the dormant v2 tool checkpoint as plain JSON data."""

    if not isinstance(checkpoint, CanonicalToolCheckpoint):
        raise TypeError("checkpoint must be a CanonicalToolCheckpoint")
    return {
        "format_version": TOOL_CHECKPOINT_FORMAT_VERSION,
        "context_revision": checkpoint.context_revision,
        "prompt_revision": checkpoint.prompt_revision,
        "transcript": [
            _plain_checkpoint_json(model_message_payload(message))
            for message in checkpoint.transcript
        ],
        "manifest": checkpoint.manifest.model_dump(mode="json"),
        "tool_calls": [
            _encode_checkpoint_tool_call(call)
            for call in checkpoint.tool_calls
        ],
        "pending_tool_calls": [
            _encode_checkpoint_tool_call(call)
            for call in checkpoint.pending_tool_calls
        ],
        "paused_tool_calls": [
            _encode_checkpoint_tool_call(call)
            for call in checkpoint.paused_tool_calls
        ],
        "model_call_records": [
            model_call_record_payload(record)
            for record in checkpoint.model_call_records
        ],
        "legacy_migrated": checkpoint.legacy_migrated,
    }


def decode_tool_checkpoint(raw: object) -> CanonicalToolCheckpoint:
    """Decode only the canonical v2 value; legacy migration stays explicit."""

    payload = _checkpoint_mapping(raw, field_name="checkpoint")
    version = payload.get("format_version")
    if (
        not isinstance(version, int)
        or isinstance(version, bool)
        or version != TOOL_CHECKPOINT_FORMAT_VERSION
    ):
        raise ValueError(
            f"unsupported tool checkpoint format_version: {version!r}"
        )
    return CanonicalToolCheckpoint(
        context_revision=_checkpoint_string(
            payload.get("context_revision"),
            field_name="context_revision",
        ),
        prompt_revision=_checkpoint_string(
            payload.get("prompt_revision"),
            field_name="prompt_revision",
        ),
        transcript=tuple(
            _decode_checkpoint_message(item)
            for item in _checkpoint_sequence(
                payload.get("transcript"),
                field_name="transcript",
            )
        ),
        manifest=ToolManifest.model_validate(
            _checkpoint_mapping(
                payload.get("manifest"),
                field_name="manifest",
            )
        ),
        tool_calls=tuple(
            _decode_checkpoint_tool_call(item)
            for item in _checkpoint_sequence(
                payload.get("tool_calls", ()),
                field_name="tool_calls",
            )
        ),
        pending_tool_calls=tuple(
            _decode_checkpoint_tool_call(item)
            for item in _checkpoint_sequence(
                payload.get("pending_tool_calls", ()),
                field_name="pending_tool_calls",
            )
        ),
        paused_tool_calls=tuple(
            _decode_checkpoint_tool_call(item)
            for item in _checkpoint_sequence(
                payload.get("paused_tool_calls", ()),
                field_name="paused_tool_calls",
            )
        ),
        model_call_records=tuple(
            _decode_model_call_record(item)
            for item in _checkpoint_sequence(
                payload.get("model_call_records", ()),
                field_name="model_call_records",
            )
        ),
        legacy_migrated=_checkpoint_bool(
            payload.get("legacy_migrated", False),
            field_name="legacy_migrated",
        ),
    )


def decode_legacy_tool_state_v1(raw: object) -> CanonicalToolCheckpoint:
    """Pure one-time migration from the committed legacy tool-state shape."""

    payload = _checkpoint_mapping(raw, field_name="legacy checkpoint")
    legacy_version = payload.get("format_version")
    if legacy_version is not None and (
        not isinstance(legacy_version, int)
        or isinstance(legacy_version, bool)
        or legacy_version != 1
    ):
        raise ValueError("legacy tool checkpoint format_version must be absent or 1")
    trace_value = payload.get("tooling_model_request_trace", {})
    trace = _checkpoint_mapping(trace_value, field_name="tooling_model_request_trace")
    exposed_names = _checkpoint_names(
        payload.get("tooling_sent_schema_names", ()),
        field_name="tooling_sent_schema_names",
    )
    active_names = _checkpoint_names(
        payload.get("discovery_active_tools", ()),
        field_name="discovery_active_tools",
    )
    if not set(active_names) <= set(exposed_names):
        raise ValueError("legacy active tools must be present in sent schema names")
    resident_names = tuple(
        name for name in exposed_names if name not in set(active_names)
    )
    ordered_manifest_names = (*resident_names, *active_names)
    legacy_hash = "legacy_unverified_v1"
    toolset_revision = _checkpoint_string_or_default(
        trace.get("toolset_revision"),
        default="legacy_tools_unverified_v1",
        field_name="toolset_revision",
    )
    manifest = ToolManifest(
        entries=tuple(
            ToolManifestEntry(
                name=name,
                description_hash=legacy_hash,
                input_schema_hash=legacy_hash,
                static_effects_hash=legacy_hash,
                execution_contract_hash=legacy_hash,
            )
            for name in ordered_manifest_names
        ),
        resident_tool_names=resident_names,
        explicit_tool_names=(),
        active_tool_names=active_names,
        toolset_revision=toolset_revision,
        provider_serializer_revision=_checkpoint_string_or_default(
            trace.get("provider_serializer_revision"),
            default="legacy_provider_serializer_v1",
            field_name="provider_serializer_revision",
        ),
    )
    origin = ToolCallOrigin(
        request_id=_checkpoint_string_or_default(
            trace.get("request_id"),
            default="legacy_request_v1",
            field_name="request_id",
        ),
        toolset_revision=toolset_revision,
        exposed_tool_names=exposed_names,
    )
    result_by_id = _legacy_tool_results(payload.get("tool_results", ()))
    pending_statuses = _legacy_pending_statuses(
        payload.get(
            "pending_loop_tool_calls",
            payload.get("pending_tool_calls", ()),
        )
    )
    ledger_value = _checkpoint_mapping(
        payload.get("tool_call_ledger", {}),
        field_name="tool_call_ledger",
    )
    ledger_entries = list(
        _checkpoint_sequence(
            ledger_value.get("entries", ()),
            field_name="tool_call_ledger.entries",
        )
    )
    ledger_entries.sort(key=_legacy_ledger_order)

    initial_task = _checkpoint_string_or_default(
        payload.get("initial_user_task"),
        default="Legacy checkpoint task unavailable.",
        field_name="initial_user_task",
    )
    transcript: list[ModelMessage] = [
        ModelMessage(role="user", content=initial_task)
    ]
    calls: list[CanonicalToolCall] = []
    pending: list[CanonicalToolCall] = []
    paused: list[CanonicalToolCall] = []
    seen_call_ids: set[str] = set()
    for raw_entry in ledger_entries:
        entry = _checkpoint_mapping(
            raw_entry,
            field_name="tool_call_ledger entry",
        )
        plan_value = entry.get("plan", entry)
        plan = _checkpoint_mapping(plan_value, field_name="tool call plan")
        call_id = _checkpoint_string(
            plan.get("tool_call_id"),
            field_name="tool_call_id",
        )
        if call_id in seen_call_ids:
            raise ValueError(f"duplicate legacy tool_call_id: {call_id}")
        seen_call_ids.add(call_id)
        tool_name = _checkpoint_string(
            plan.get("tool_name"),
            field_name="tool_name",
        )
        arguments = _checkpoint_mapping(
            plan.get("arguments", {}),
            field_name="tool call arguments",
        )
        plain_arguments = _plain_checkpoint_json(arguments)
        if not isinstance(plain_arguments, dict):
            raise TypeError("legacy tool arguments must serialize as an object")
        transcript.append(
            ModelMessage(
                role="assistant",
                content="",
                tool_calls=(
                    ModelToolCall(
                        id=call_id,
                        name=tool_name,
                        input=plain_arguments,
                    ),
                ),
            )
        )
        call = CanonicalToolCall(
            tool_call_id=call_id,
            tool_name=tool_name,
            arguments=cast(Mapping[str, JsonValue], arguments),
            origin=origin,
        )
        calls.append(call)
        result = result_by_id.get(call_id)
        if result is not None:
            visible_result: dict[str, object] = {
                "status": result.get("status"),
                "output": result.get("output"),
                "error": result.get("error"),
            }
            transcript.append(
                ModelMessage(
                    role="tool",
                    content=canonical_json_text(
                        cast(JsonValue, visible_result)
                    ),
                    tool_call_id=call_id,
                )
            )
        status = pending_statuses.get(call_id)
        if status == "paused":
            paused.append(call)
        elif status is not None or result is None:
            pending.append(call)

    unknown_pending = set(pending_statuses) - seen_call_ids
    unknown_results = set(result_by_id) - seen_call_ids
    if unknown_pending:
        raise ValueError("legacy pending calls are missing from the tool-call ledger")
    if unknown_results:
        raise ValueError("legacy tool results are missing from the tool-call ledger")
    return CanonicalToolCheckpoint(
        context_revision=_checkpoint_string_or_default(
            trace.get("context_revision"),
            default="legacy_context_v1",
            field_name="context_revision",
        ),
        prompt_revision=_checkpoint_string_or_default(
            trace.get("prompt_revision"),
            default="legacy_prompt_unverified_v1",
            field_name="prompt_revision",
        ),
        transcript=tuple(transcript),
        manifest=manifest,
        tool_calls=tuple(calls),
        pending_tool_calls=tuple(pending),
        paused_tool_calls=tuple(paused),
        model_call_records=(),
        legacy_migrated=True,
    )


def reconcile_tool_manifest(
    *,
    persisted: ToolManifest,
    rebuilt: ToolManifest,
    pending_tool_calls: Sequence[CanonicalToolCall],
    paused_tool_calls: Sequence[CanonicalToolCall],
) -> ToolManifestDriftDecision:
    """Compare persisted evidence with rebuilt tools without mutating runtime state."""

    if not isinstance(persisted, ToolManifest):
        raise TypeError("persisted must be a ToolManifest")
    if not isinstance(rebuilt, ToolManifest):
        raise TypeError("rebuilt must be a ToolManifest")
    pending = _checkpoint_tool_calls(
        pending_tool_calls,
        field_name="pending_tool_calls",
    )
    paused = _checkpoint_tool_calls(
        paused_tool_calls,
        field_name="paused_tool_calls",
    )
    persisted_entries = {entry.name: entry for entry in persisted.entries}
    rebuilt_entries = {entry.name: entry for entry in rebuilt.entries}
    missing = tuple(
        entry.name
        for entry in persisted.entries
        if entry.name not in rebuilt_entries
    )
    changed_existing = tuple(
        entry.name
        for entry in persisted.entries
        if entry.name in rebuilt_entries
        and entry != rebuilt_entries[entry.name]
    )
    added = tuple(
        entry.name
        for entry in rebuilt.entries
        if entry.name not in persisted_entries
    )
    changed = (*changed_existing, *added)
    affected_existing = set(missing) | set(changed_existing)
    dependents = tuple(
        call
        for call in (*pending, *paused)
        if call.tool_name in affected_existing
    )

    if persisted == rebuilt:
        return ToolManifestDriftDecision(
            status=ToolManifestDriftStatus.MATCH,
            reason="manifest_match",
            toolset_revision=persisted.toolset_revision,
            active_tool_names=persisted.active_tool_names,
            provider_wire_hash_guaranteed=True,
        )
    if dependents:
        return ToolManifestDriftDecision(
            status=ToolManifestDriftStatus.RECONCILIATION_REQUIRED,
            reason="tool_definition_changed",
            toolset_revision=persisted.toolset_revision,
            active_tool_names=persisted.active_tool_names,
            missing_tool_names=missing,
            changed_tool_names=changed,
            dependent_tool_calls=dependents,
            provider_wire_hash_guaranteed=False,
        )

    serializer_changed = (
        persisted.provider_serializer_revision
        != rebuilt.provider_serializer_revision
    )
    available_names = set(rebuilt_entries)
    active_names = tuple(
        name
        for name in persisted.active_tool_names
        if name in available_names
    )
    return ToolManifestDriftDecision(
        status=ToolManifestDriftStatus.NEW_REVISION_REQUIRED,
        reason=(
            "provider_serializer_changed"
            if serializer_changed
            and not missing
            and not changed
            and persisted.toolset_revision == rebuilt.toolset_revision
            else "tool_manifest_changed"
        ),
        toolset_revision=rebuilt.toolset_revision,
        active_tool_names=active_names,
        missing_tool_names=missing,
        changed_tool_names=changed,
        provider_wire_hash_guaranteed=False,
    )


def _encode_checkpoint_tool_call(
    call: CanonicalToolCall,
) -> dict[str, object]:
    if not isinstance(call, CanonicalToolCall):
        raise TypeError("tool call must be a canonical ToolCall")
    return {
        "tool_call_id": call.tool_call_id,
        "tool_name": call.tool_name,
        "arguments": _plain_checkpoint_json(call.arguments),
        "origin": {
            "request_id": call.origin.request_id,
            "toolset_revision": call.origin.toolset_revision,
            "exposed_tool_names": list(call.origin.exposed_tool_names),
        },
    }


def _decode_checkpoint_tool_call(raw: object) -> CanonicalToolCall:
    payload = _checkpoint_mapping(raw, field_name="tool call")
    origin_payload = _checkpoint_mapping(
        payload.get("origin"),
        field_name="tool call origin",
    )
    arguments = _checkpoint_mapping(
        payload.get("arguments", {}),
        field_name="tool call arguments",
    )
    return CanonicalToolCall(
        tool_call_id=_checkpoint_string(
            payload.get("tool_call_id"),
            field_name="tool_call_id",
        ),
        tool_name=_checkpoint_string(
            payload.get("tool_name"),
            field_name="tool_name",
        ),
        arguments=cast(Mapping[str, JsonValue], arguments),
        origin=ToolCallOrigin(
            request_id=_checkpoint_string(
                origin_payload.get("request_id"),
                field_name="origin request_id",
            ),
            toolset_revision=_checkpoint_string(
                origin_payload.get("toolset_revision"),
                field_name="origin toolset_revision",
            ),
            exposed_tool_names=_checkpoint_names(
                origin_payload.get("exposed_tool_names", ()),
                field_name="origin exposed_tool_names",
            ),
        ),
    )


def _decode_checkpoint_message(raw: object) -> ModelMessage:
    payload = _checkpoint_mapping(raw, field_name="model message")
    role = _checkpoint_string(payload.get("role"), field_name="message role")
    content = payload.get("content")
    if not isinstance(content, str):
        raise TypeError("message content must be a string")
    tool_calls: list[ModelToolCall] = []
    for raw_call in _checkpoint_sequence(
        payload.get("tool_calls", ()),
        field_name="message tool_calls",
    ):
        call = _checkpoint_mapping(raw_call, field_name="message tool call")
        arguments = _plain_checkpoint_json(
            _checkpoint_mapping(
                call.get("arguments", {}),
                field_name="message tool-call arguments",
            )
        )
        if not isinstance(arguments, dict):
            raise TypeError("message tool-call arguments must be an object")
        tool_calls.append(
            ModelToolCall(
                id=_checkpoint_string(
                    call.get("id"),
                    field_name="message tool-call id",
                ),
                name=_checkpoint_string(
                    call.get("name"),
                    field_name="message tool-call name",
                ),
                input=arguments,
            )
        )
    tool_call_id = payload.get("tool_call_id")
    if tool_call_id is not None and not isinstance(tool_call_id, str):
        raise TypeError("message tool_call_id must be a string or None")
    return snapshot_model_message(
        ModelMessage(
            role=cast(Any, role),
            content=content,
            tool_calls=tuple(tool_calls),
            tool_call_id=tool_call_id,
        )
    )


def _decode_model_call_record(raw: object) -> ModelCallRecord:
    payload = _checkpoint_mapping(raw, field_name="model call record")
    usage_payload = _checkpoint_mapping(
        payload.get("usage"),
        field_name="model call usage",
    )
    usage_source = _checkpoint_string(
        usage_payload.get("usage_source"),
        field_name="usage_source",
    )
    if usage_source not in {"provider", "tokenizer_estimate"}:
        raise ValueError(f"unsupported usage_source: {usage_source}")
    logical_input = _checkpoint_optional_int(
        usage_payload.get("logical_input_tokens"),
        field_name="logical_input_tokens",
    )
    uncached_input = _checkpoint_optional_int(
        usage_payload.get("uncached_input_tokens"),
        field_name="uncached_input_tokens",
    )
    cache_read = _checkpoint_optional_int(
        usage_payload.get("cache_read_input_tokens"),
        field_name="cache_read_input_tokens",
    )
    cache_write = _checkpoint_optional_int(
        usage_payload.get("cache_write_input_tokens"),
        field_name="cache_write_input_tokens",
    )
    output_tokens = _checkpoint_int(
        usage_payload.get("output_tokens"),
        field_name="output_tokens",
    )
    raw_usage = usage_payload.get("raw_provider_usage")
    if raw_usage is not None and not isinstance(raw_usage, Mapping):
        raise TypeError("raw_provider_usage must be an object or None")
    if (
        logical_input is not None
        and uncached_input is not None
        and cache_read is not None
        and logical_input
        != uncached_input + cache_read + (cache_write or 0)
    ):
        raise ValueError("checkpoint contains inconsistent normalized usage")
    if usage_source == "tokenizer_estimate" and (
        cache_read is not None
        or cache_write is not None
        or raw_usage is not None
    ):
        raise ValueError("tokenizer usage cannot contain provider cache evidence")
    usage = LLMUsage(
        input_tokens=(
            logical_input
            if logical_input is not None
            else (uncached_input or 0)
        ),
        output_tokens=output_tokens,
        cached_input_tokens=cache_read or 0,
        source=cast(Any, usage_source),
        logical_input_tokens=logical_input,
        uncached_input_tokens=uncached_input,
        cache_read_input_tokens=cache_read,
        cache_write_input_tokens=cache_write,
        usage_source=cast(Any, usage_source),
        raw_provider_usage=cast(Any, raw_usage),
    )
    return ModelCallRecord(
        request_id=_checkpoint_string(
            payload.get("request_id"),
            field_name="request_id",
        ),
        prompt_revision=_checkpoint_string(
            payload.get("prompt_revision"),
            field_name="prompt_revision",
        ),
        toolset_revision=_checkpoint_string(
            payload.get("toolset_revision"),
            field_name="toolset_revision",
        ),
        provider_wire_hash=_checkpoint_string(
            payload.get("provider_wire_hash"),
            field_name="provider_wire_hash",
        ),
        usage=usage,
    )


def _legacy_tool_results(raw: object) -> dict[str, Mapping[str, object]]:
    results: dict[str, Mapping[str, object]] = {}
    for item in _checkpoint_sequence(raw, field_name="tool_results"):
        result = _checkpoint_mapping(item, field_name="legacy tool result")
        call_id = _checkpoint_string(
            result.get("tool_call_id"),
            field_name="legacy result tool_call_id",
        )
        if call_id in results:
            raise ValueError(f"duplicate legacy tool result: {call_id}")
        results[call_id] = result
    return results


def _legacy_pending_statuses(raw: object) -> dict[str, str]:
    statuses: dict[str, str] = {}
    for item in _checkpoint_sequence(raw, field_name="pending tool calls"):
        pending = _checkpoint_mapping(item, field_name="legacy pending tool call")
        call_id = _checkpoint_string(
            pending.get("tool_call_id"),
            field_name="legacy pending tool_call_id",
        )
        status = pending.get("status", "pending")
        if not isinstance(status, str) or not status:
            raise ValueError("legacy pending status must be a non-empty string")
        statuses[call_id] = status
    return statuses


def _legacy_ledger_order(raw: object) -> tuple[int, int]:
    entry = _checkpoint_mapping(raw, field_name="tool-call ledger entry")
    return (
        _checkpoint_int(entry.get("turn", 0), field_name="ledger turn"),
        _checkpoint_int(
            entry.get("sequence", 0),
            field_name="ledger sequence",
        ),
    )


def _checkpoint_tool_calls(
    calls: Sequence[CanonicalToolCall],
    *,
    field_name: str,
) -> tuple[CanonicalToolCall, ...]:
    if isinstance(calls, (str, bytes)) or not isinstance(calls, Sequence):
        raise TypeError(f"{field_name} must be a sequence of canonical ToolCall values")
    result = tuple(calls)
    if any(not isinstance(call, CanonicalToolCall) for call in result):
        raise TypeError(f"{field_name} must contain canonical ToolCall values")
    ids = tuple(call.tool_call_id for call in result)
    if len(set(ids)) != len(ids):
        raise ValueError(f"{field_name} must contain unique tool_call_id values")
    return result


def _checkpoint_mapping(
    value: object,
    *,
    field_name: str,
) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be an object")
    for key in value:
        if not isinstance(key, str):
            raise TypeError(f"{field_name} keys must be strings")
    return cast(Mapping[str, object], value)


def _checkpoint_sequence(
    value: object,
    *,
    field_name: str,
) -> Sequence[object]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{field_name} must be an array")
    return cast(Sequence[object], value)


def _checkpoint_names(value: object, *, field_name: str) -> tuple[str, ...]:
    names = tuple(
        _checkpoint_string(item, field_name=field_name)
        for item in _checkpoint_sequence(value, field_name=field_name)
    )
    if len(set(names)) != len(names):
        raise ValueError(f"{field_name} must contain unique names")
    return names


def _checkpoint_string(value: object, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    if not value:
        raise ValueError(f"{field_name} must not be empty")
    return value


def _checkpoint_string_or_default(
    value: object,
    *,
    default: str,
    field_name: str,
) -> str:
    if value is None:
        return default
    return _checkpoint_string(value, field_name=field_name)


def _checkpoint_int(value: object, *, field_name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{field_name} must be an integer")
    if value < 0:
        raise ValueError(f"{field_name} must be non-negative")
    return value


def _checkpoint_optional_int(value: object, *, field_name: str) -> int | None:
    if value is None:
        return None
    return _checkpoint_int(value, field_name=field_name)


def _checkpoint_bool(value: object, *, field_name: str) -> bool:
    if type(value) is not bool:
        raise TypeError(f"{field_name} must be a bool")
    return value


def _plain_checkpoint_json(value: object) -> object:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        result: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("checkpoint JSON keys must be strings")
            result[key] = _plain_checkpoint_json(item)
        return result
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_plain_checkpoint_json(item) for item in value]
    raise TypeError("checkpoint value must contain only JSON-compatible values")


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
    # Backfill typed sub-state for old checkpoints (flat discovery fields deprecated)
    from rag.agent.loop.substate import DeferredToolState

    state.setdefault("deferred_tool_state", DeferredToolState())
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

    # ── SkillState ──
    from rag.agent.skills.models import SkillState

    raw_skill_state = state.get("skill_state")
    if raw_skill_state is None:
        state["skill_state"] = SkillState()
    elif not isinstance(raw_skill_state, SkillState):
        try:
            state["skill_state"] = SkillState.model_validate(raw_skill_state)
        except Exception:
            state.setdefault("runtime_diagnostics", []).append(
                RuntimeDiagnostic(
                    code="invalid_skill_state_dropped",
                    component="checkpoint_migration",
                    message="Invalid skill_state was dropped during checkpoint migration.",
                    severity="warning",
                )
            )
            state["skill_state"] = SkillState()

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
    return LazyAsyncSqliteSaver(path, serde=agent_checkpoint_serde())


async def aclose_agent_checkpointer(checkpointer: BaseCheckpointSaver[str]) -> None:
    close_method = getattr(checkpointer, "aclose", None)
    if callable(close_method):
        await close_method()
        return
    connection = getattr(checkpointer, "conn", None)
    if connection is not None and hasattr(connection, "close"):
        await connection.close()

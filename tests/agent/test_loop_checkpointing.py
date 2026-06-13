from __future__ import annotations

from pathlib import Path

import pytest
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel

from rag.agent.compat.goal_contract import GoalCompatibilityConfig, GoalSpec
from rag.agent.core.checkpointing import (
    CheckpointPersistenceError,
    LangGraphCheckpointStore,
    aclose_agent_checkpointer,
    agent_checkpoint_serde,
    create_agent_checkpointer,
)
from rag.agent.core.context import AgentRunConfig, RunRegistry
from rag.agent.core.human_input import (
    HumanInputRequest,
    HumanInputRequestIdMismatchError,
    HumanInputResponse,
)
from rag.agent.core.tool_execution import (
    ToolBatchRequest,
    ToolExecutionRecord,
    ToolExecutionService,
    tool_arguments_digest,
)
from rag.agent.loop.state import (
    LoopPause,
    LoopTransition,
    create_loop_state,
)
from rag.agent.state import ToolCallPlan
from rag.agent.tools.registry import ToolRegistry
from rag.agent.tools.spec import ToolError, ToolPermissions, ToolSpec
from rag.schema.runtime import AccessPolicy


class _Input(BaseModel):
    value: str


class _Output(BaseModel):
    value: str


def _config(run_id: str) -> AgentRunConfig:
    config = AgentRunConfig(
        run_id=run_id,
        thread_id=run_id,
        budget_total=100,
        max_depth=2,
        access_policy=AccessPolicy.default(),
    )
    RunRegistry.remove(run_id)
    RunRegistry.get_or_create(config)
    return config


def _record(
    call: ToolCallPlan,
    *,
    status: str,
    idempotent: bool,
    operation_id: str = "op-stable",
) -> ToolExecutionRecord:
    return ToolExecutionRecord(
        tool_call_id=call.tool_call_id,
        tool_name=call.tool_name,
        operation_id=operation_id,
        arguments_digest=tool_arguments_digest(call.arguments),
        idempotent=idempotent,
        status=status,
        attempt_count=1 if status != "prepared" else 0,
    )


@pytest.mark.anyio
async def test_memory_store_round_trips_loop_snapshot_in_dedicated_namespace() -> None:
    config = _config("loop-checkpoint-memory")
    saver = MemorySaver(serde=agent_checkpoint_serde())
    store = LangGraphCheckpointStore(saver, run_config=config)
    state = create_loop_state(task="Checkpoint loop", run_config=config)
    state["iteration"] = 2
    state["latest_transition"] = LoopTransition(
        reason="compaction",
        iteration=2,
    )

    await store.save_snapshot(state, reason="compaction")
    restored = await store.load_latest()
    checkpoint_tuple = await saver.aget_tuple(
        {
            "configurable": {
                "thread_id": config.thread_id,
                "checkpoint_ns": "agent_loop",
            }
        }
    )

    assert restored == state
    assert restored is not state
    assert checkpoint_tuple is not None
    assert set(checkpoint_tuple.checkpoint["channel_values"]) == {
        "loop_state"
    }
    assert checkpoint_tuple.metadata["reason"] == "compaction"
    RunRegistry.remove(config.run_id)


@pytest.mark.anyio
async def test_goal_hook_config_round_trips_in_separate_checkpoint_channel() -> None:
    config = _config("loop-checkpoint-goal-compatibility")
    saver = MemorySaver(serde=agent_checkpoint_serde())
    goal_spec = GoalSpec(original_query="Answer with evidence.")
    store = LangGraphCheckpointStore(
        saver,
        run_config=config,
        compatibility_config=GoalCompatibilityConfig(
            goal_spec=goal_spec
        ),
    )
    state = create_loop_state(task="Checkpoint loop", run_config=config)

    await store.save_snapshot(state, reason="approval_pause")
    restored_store = LangGraphCheckpointStore(saver, run_config=config)
    restored = await restored_store.load_latest()
    checkpoint_tuple = await saver.aget_tuple(
        {
            "configurable": {
                "thread_id": config.thread_id,
                "checkpoint_ns": "agent_loop",
            }
        }
    )

    assert restored == state
    assert "goal_spec" not in restored
    assert restored_store.compatibility_config.goal_spec == goal_spec
    assert checkpoint_tuple is not None
    assert set(checkpoint_tuple.checkpoint["channel_values"]) == {
        "loop_compatibility",
        "loop_state",
    }
    RunRegistry.remove(config.run_id)


@pytest.mark.anyio
async def test_sqlite_store_round_trips_between_store_instances(
    tmp_path: Path,
) -> None:
    config = _config("loop-checkpoint-sqlite")
    checkpoint_db = tmp_path / "loop-checkpoints.sqlite"
    saver = create_agent_checkpointer(checkpoint_db)
    store = LangGraphCheckpointStore(saver, run_config=config)
    state = create_loop_state(task="SQLite loop", run_config=config)
    state["latest_transition"] = LoopTransition(
        reason="approval_required",
        iteration=0,
    )

    await store.save_snapshot(state, reason="approval_pause")
    await aclose_agent_checkpointer(saver)

    reopened = create_agent_checkpointer(checkpoint_db)
    try:
        restored = await LangGraphCheckpointStore(
            reopened,
            run_config=config,
        ).load_latest()
        assert restored == state
    finally:
        await aclose_agent_checkpointer(reopened)
    RunRegistry.remove(config.run_id)


@pytest.mark.anyio
async def test_execution_record_writer_persists_each_transition() -> None:
    config = _config("loop-record-writer")
    store = LangGraphCheckpointStore(
        MemorySaver(serde=agent_checkpoint_serde()),
        run_config=config,
    )
    state = create_loop_state(task="Execute tool", run_config=config)
    call = ToolCallPlan.create("tool", {"value": "write"})
    state["pending_tool_calls"] = [call]
    await store.save_snapshot(state, reason="initial")

    for status in ("prepared", "started", "completed"):
        await store.write_execution_record(
            _record(
                call,
                status=status,
                idempotent=True,
            )
        )

    restored = await store.load_latest()

    assert restored is not None
    assert restored["tool_execution_records"][call.tool_call_id].status == "completed"
    assert restored["latest_transition"] == LoopTransition(
        reason="tool_execution",
        iteration=0,
        detail={
            "tool_call_id": call.tool_call_id,
            "execution_status": "completed",
        },
    )
    RunRegistry.remove(config.run_id)


@pytest.mark.anyio
async def test_non_idempotent_started_record_becomes_unknown_on_resume() -> None:
    config = _config("loop-non-idempotent-resume")
    saver = MemorySaver(serde=agent_checkpoint_serde())
    store = LangGraphCheckpointStore(saver, run_config=config)
    state = create_loop_state(task="Resume write", run_config=config)
    call = ToolCallPlan.create("write_tool", {"value": "write"})
    state["pending_tool_calls"] = [call]
    state["tool_execution_records"][call.tool_call_id] = _record(
        call,
        status="started",
        idempotent=False,
    )
    await store.save_snapshot(state, reason="crash_after_started")

    resumed = await LangGraphCheckpointStore(
        saver,
        run_config=config,
    ).load_for_resume()

    assert resumed is not None
    assert resumed["status"] == "paused"
    assert resumed["tool_execution_records"][call.tool_call_id].status == "unknown"
    assert resumed["pause"] is not None
    assert resumed["pause"].request is not None
    assert resumed["pause"].request.kind == "tool_reconciliation"
    RunRegistry.remove(config.run_id)


@pytest.mark.anyio
async def test_idempotent_started_record_keeps_operation_id_on_resume() -> None:
    config = _config("loop-idempotent-resume")
    saver = MemorySaver(serde=agent_checkpoint_serde())
    store = LangGraphCheckpointStore(saver, run_config=config)
    state = create_loop_state(task="Resume read", run_config=config)
    call = ToolCallPlan.create("read_tool", {"value": "read"})
    state["pending_tool_calls"] = [call]
    state["tool_execution_records"][call.tool_call_id] = _record(
        call,
        status="started",
        idempotent=True,
        operation_id="op-resume-stable",
    )
    await store.save_snapshot(state, reason="crash_after_started")

    resumed = await LangGraphCheckpointStore(
        saver,
        run_config=config,
    ).load_for_resume()

    assert resumed is not None
    assert resumed["status"] == "running"
    assert (
        resumed["tool_execution_records"][call.tool_call_id].operation_id
        == "op-resume-stable"
    )
    RunRegistry.remove(config.run_id)


@pytest.mark.anyio
async def test_reconciliation_response_authorizes_new_operation() -> None:
    config = _config("loop-reconciliation")
    saver = MemorySaver(serde=agent_checkpoint_serde())
    store = LangGraphCheckpointStore(saver, run_config=config)
    state = create_loop_state(task="Reconcile write", run_config=config)
    call = ToolCallPlan.create("write_tool", {"value": "write"})
    record = _record(
        call,
        status="unknown",
        idempotent=False,
        operation_id="op-old",
    )
    state["pending_tool_calls"] = [call]
    state["tool_execution_records"][call.tool_call_id] = record
    state["status"] = "paused"
    state["pause"] = LoopPause(
        reason="Tool outcome is ambiguous.",
        request=store.reconciliation_request(record),
    )
    await store.save_snapshot(state, reason="tool_reconciliation")

    reconciled = await store.apply_human_response(
        HumanInputResponse(
            request_id=state["pause"].request.request_id,
            decision="retry_new_operation",
        )
    )

    updated = reconciled["tool_execution_records"][call.tool_call_id]
    assert updated.status == "prepared"
    assert updated.operation_id != "op-old"
    assert reconciled["status"] == "running"
    assert reconciled["pause"] is None
    RunRegistry.remove(config.run_id)


@pytest.mark.anyio
async def test_human_response_request_id_must_match_pending_pause() -> None:
    config = _config("loop-request-mismatch")
    store = LangGraphCheckpointStore(
        MemorySaver(serde=agent_checkpoint_serde()),
        run_config=config,
    )
    state = create_loop_state(task="Approve", run_config=config)
    state["status"] = "paused"
    state["pause"] = LoopPause(
        reason="Need approval",
        request=HumanInputRequest(
            request_id="hir-approval",
            kind="tool_approval",
            question="Approve tool?",
            options=["allow_once", "deny", "abort"],
        ),
    )
    await store.save_snapshot(state, reason="approval_pause")

    with pytest.raises(HumanInputRequestIdMismatchError):
        await store.apply_human_response(
            HumanInputResponse(
                request_id="hir-wrong",
                decision="allow_once",
            )
        )
    RunRegistry.remove(config.run_id)


@pytest.mark.anyio
async def test_completed_record_loaded_from_store_is_not_replayed() -> None:
    config = _config("loop-completed-no-replay")
    saver = MemorySaver(serde=agent_checkpoint_serde())
    store = LangGraphCheckpointStore(saver, run_config=config)
    state = create_loop_state(task="Do not replay", run_config=config)
    call = ToolCallPlan.create("tool", {"value": "done"})
    state["pending_tool_calls"] = [call]
    state["tool_execution_records"][call.tool_call_id] = _record(
        call,
        status="completed",
        idempotent=True,
    )
    await store.save_snapshot(state, reason="tool_completed")
    restored = await store.load_for_resume()
    assert restored is not None

    calls = 0

    def runner(payload: _Input) -> _Output:
        nonlocal calls
        calls += 1
        return _Output(value=payload.value)

    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            name="tool",
            description="no replay",
            input_model=_Input,
            output_model=_Output,
            error_model=ToolError,
            permissions=ToolPermissions(),
            timeout_seconds=1,
            idempotent=True,
        ),
        runner=runner,
    )
    result = await ToolExecutionService(
        tool_registry=registry,
        record_writer=store,
    ).execute_batch(
        ToolBatchRequest(
            calls=(call,),
            run_config=config,
            allowed_tools=frozenset({"tool"}),
            execution_records=restored["tool_execution_records"],
        ),
        state=None,
    )

    assert result.skipped_completed_tool_call_ids == (call.tool_call_id,)
    assert calls == 0
    RunRegistry.remove(config.run_id)


class _FailingSaver(MemorySaver):
    async def aput(self, *args: object, **kwargs: object) -> object:
        del args, kwargs
        raise OSError("disk unavailable")


@pytest.mark.anyio
async def test_checkpoint_write_failure_is_visible() -> None:
    config = _config("loop-checkpoint-failure")
    store = LangGraphCheckpointStore(
        _FailingSaver(serde=agent_checkpoint_serde()),
        run_config=config,
    )
    state = create_loop_state(task="Fail visibly", run_config=config)

    with pytest.raises(CheckpointPersistenceError, match="disk unavailable"):
        await store.save_snapshot(state, reason="terminal")
    RunRegistry.remove(config.run_id)

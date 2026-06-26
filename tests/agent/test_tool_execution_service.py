from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field

import pytest
from pydantic import BaseModel

from rag.agent.core.checkpointing import agent_checkpoint_serde
from rag.agent.core.context import AgentRunConfig, RunRegistry
from rag.agent.core.definition import ToolPolicy
from rag.agent.core.human_input import HumanInputResponse
from rag.agent.core.tool_execution import (
    ToolBatchRequest,
    ToolExecutionRecord,
    ToolExecutionService,
    apply_tool_reconciliation,
    tool_arguments_digest,
)
from rag.agent.core.turn_contracts import ToolCallPlan
from rag.agent.tools.registry import ToolExecutionContext, ToolRegistry
from rag.agent.tools.spec import ToolError, ToolPermissions, ToolSpec
from rag.schema.runtime import AccessPolicy


class _Input(BaseModel):
    value: str


class _Output(BaseModel):
    value: str


def _config(
    run_id: str,
    *,
    max_parallel_calls: int = 4,
) -> AgentRunConfig:
    config = AgentRunConfig(
        run_id=run_id,
        thread_id=run_id,
        budget_total=100,
        max_depth=2,
        access_policy=AccessPolicy.default(),
        tool_policy=ToolPolicy(max_parallel_calls=max_parallel_calls),
    )
    RunRegistry.remove(run_id)
    RunRegistry.get_or_create(config)
    return config


def _spec(
    name: str = "tool",
    *,
    idempotent: bool = False,
    concurrency_safe: bool = False,
    max_retries: int = 0,
    requires_confirmation: bool = False,
) -> ToolSpec:
    return ToolSpec(
        name=name,
        description="tool execution service test",
        input_model=_Input,
        output_model=_Output,
        error_model=ToolError,
        permissions=ToolPermissions(),
        timeout_seconds=1.0,
        max_retries=max_retries,
        idempotent=idempotent,
        concurrency_safe=concurrency_safe,
        requires_confirmation=requires_confirmation,
    )


def _request(
    call: ToolCallPlan,
    *,
    run_id: str,
    approved: tuple[str, ...] = (),
    records: dict[str, ToolExecutionRecord] | None = None,
) -> ToolBatchRequest:
    return ToolBatchRequest(
        calls=(call,),
        run_config=_config(run_id),
        allowed_tools=frozenset({call.tool_name}),
        approved_tool_call_ids=approved,
        execution_records=records or {},
    )


@dataclass
class _RecordingWriter:
    durable = True

    records: list[ToolExecutionRecord] = field(default_factory=list)
    fail_on_status: str | None = None

    async def write_execution_record(self, record: ToolExecutionRecord) -> None:
        self.records.append(record.model_copy(deep=True))
        if record.status == self.fail_on_status:
            raise RuntimeError(f"cannot persist {record.status}")


@pytest.mark.anyio
async def test_approved_call_persists_transitions_and_passes_stable_operation_id() -> None:
    contexts: list[ToolExecutionContext] = []

    def runner(payload: _Input, context: ToolExecutionContext) -> _Output:
        contexts.append(context)
        return _Output(value=payload.value)

    registry = ToolRegistry()
    registry.register(_spec(requires_confirmation=True))
    registry.register_contextual_runner("tool", runner)
    writer = _RecordingWriter()
    call = ToolCallPlan.create("tool", {"value": "write"})

    result = await ToolExecutionService(
        tool_registry=registry,
        record_writer=writer,
    ).execute_batch(
        _request(
            call,
            run_id="execution-approved",
            approved=(call.tool_call_id,),
        ),
        state={},
    )

    assert result.status == "completed"
    assert result.tool_results[0].status == "ok"
    assert [record.status for record in writer.records] == [
        "prepared",
        "started",
        "completed",
    ]
    operation_ids = {record.operation_id for record in writer.records}
    assert len(operation_ids) == 1
    assert contexts[0].operation_id == writer.records[0].operation_id
    RunRegistry.remove("execution-approved")


@pytest.mark.anyio
async def test_approval_required_returns_typed_pause_without_invoking_runner() -> None:
    calls = 0

    def runner(payload: _Input) -> _Output:
        nonlocal calls
        calls += 1
        return _Output(value=payload.value)

    registry = ToolRegistry()
    registry.register(_spec(requires_confirmation=True), runner=runner)
    call = ToolCallPlan.create("tool", {"value": "sensitive"})

    result = await ToolExecutionService(tool_registry=registry).execute_batch(
        _request(call, run_id="execution-approval"),
        state={},
    )

    assert result.status == "paused"
    assert result.human_input_request is not None
    assert result.human_input_request.kind == "tool_approval"
    assert result.pending_tool_calls == (call,)
    assert result.execution_records == {}
    assert calls == 0
    RunRegistry.remove("execution-approval")


@pytest.mark.anyio
async def test_started_persistence_failure_prevents_tool_invocation() -> None:
    calls = 0

    def runner(payload: _Input) -> _Output:
        nonlocal calls
        calls += 1
        return _Output(value=payload.value)

    registry = ToolRegistry()
    registry.register(_spec(), runner=runner)
    writer = _RecordingWriter(fail_on_status="started")
    call = ToolCallPlan.create("tool", {"value": "blocked"})

    with pytest.raises(RuntimeError, match="cannot persist started"):
        await ToolExecutionService(
            tool_registry=registry,
            record_writer=writer,
        ).execute_batch(
            _request(call, run_id="execution-writer-failure"),
            state={},
        )

    assert calls == 0
    RunRegistry.remove("execution-writer-failure")


@pytest.mark.anyio
async def test_idempotent_retry_reuses_operation_id() -> None:
    seen_operation_ids: list[str | None] = []

    def runner(payload: _Input, context: ToolExecutionContext) -> _Output:
        seen_operation_ids.append(context.operation_id)
        if len(seen_operation_ids) == 1:
            raise RuntimeError("temporary")
        return _Output(value=payload.value)

    registry = ToolRegistry()
    registry.register(_spec(idempotent=True, max_retries=1))
    registry.register_contextual_runner("tool", runner)
    call = ToolCallPlan.create("tool", {"value": "retry"})

    result = await ToolExecutionService(tool_registry=registry).execute_batch(
        _request(call, run_id="execution-retry"),
        state={},
    )

    assert result.tool_results[0].retry_count == 1
    assert len(set(seen_operation_ids)) == 1
    assert result.execution_records[call.tool_call_id].attempt_count == 2
    RunRegistry.remove("execution-retry")


@pytest.mark.anyio
async def test_completed_record_is_not_replayed() -> None:
    calls = 0

    def runner(payload: _Input) -> _Output:
        nonlocal calls
        calls += 1
        return _Output(value=payload.value)

    registry = ToolRegistry()
    spec = _spec(idempotent=True)
    registry.register(spec, runner=runner)
    call = ToolCallPlan.create("tool", {"value": "done"})
    record = ToolExecutionRecord(
        tool_call_id=call.tool_call_id,
        tool_name=call.tool_name,
        operation_id="op-completed",
        arguments_digest=tool_arguments_digest(call.arguments),
        idempotent=True,
        status="completed",
        attempt_count=1,
    )

    result = await ToolExecutionService(tool_registry=registry).execute_batch(
        _request(
            call,
            run_id="execution-completed",
            records={call.tool_call_id: record},
        ),
        state={},
    )

    assert result.status == "completed"
    assert result.skipped_completed_tool_call_ids == (call.tool_call_id,)
    assert result.tool_results == ()
    assert calls == 0
    RunRegistry.remove("execution-completed")


@pytest.mark.anyio
async def test_non_idempotent_started_record_requires_reconciliation() -> None:
    registry = ToolRegistry()
    registry.register(_spec(idempotent=False), runner=lambda payload: _Output(value=payload.value))
    call = ToolCallPlan.create("tool", {"value": "ambiguous"})
    record = ToolExecutionRecord(
        tool_call_id=call.tool_call_id,
        tool_name=call.tool_name,
        operation_id="op-ambiguous",
        arguments_digest=tool_arguments_digest(call.arguments),
        idempotent=False,
        status="started",
        attempt_count=1,
    )

    result = await ToolExecutionService(tool_registry=registry).execute_batch(
        _request(
            call,
            run_id="execution-reconcile",
            records={call.tool_call_id: record},
        ),
        state={},
    )

    assert result.status == "reconciliation_required"
    assert result.human_input_request is not None
    assert result.human_input_request.kind == "tool_reconciliation"
    assert result.human_input_request.options == [
        "mark_completed",
        "mark_failed",
        "retry_new_operation",
    ]
    assert result.execution_records[call.tool_call_id].status == "unknown"
    RunRegistry.remove("execution-reconcile")


@pytest.mark.anyio
async def test_idempotent_started_record_resumes_with_same_operation_id() -> None:
    seen_operation_ids: list[str | None] = []

    def runner(payload: _Input, context: ToolExecutionContext) -> _Output:
        seen_operation_ids.append(context.operation_id)
        return _Output(value=payload.value)

    registry = ToolRegistry()
    registry.register(_spec(idempotent=True))
    registry.register_contextual_runner("tool", runner)
    call = ToolCallPlan.create("tool", {"value": "resume"})
    record = ToolExecutionRecord(
        tool_call_id=call.tool_call_id,
        tool_name=call.tool_name,
        operation_id="op-stable",
        arguments_digest=tool_arguments_digest(call.arguments),
        idempotent=True,
        status="started",
        attempt_count=1,
    )

    result = await ToolExecutionService(tool_registry=registry).execute_batch(
        _request(
            call,
            run_id="execution-idempotent-resume",
            records={call.tool_call_id: record},
        ),
        state={},
    )

    assert result.status == "completed"
    assert seen_operation_ids == ["op-stable"]
    assert result.execution_records[call.tool_call_id].operation_id == "op-stable"
    RunRegistry.remove("execution-idempotent-resume")


@pytest.mark.anyio
async def test_non_idempotent_transport_failure_becomes_unknown() -> None:
    calls = 0

    def runner(payload: _Input) -> _Output:
        nonlocal calls
        calls += 1
        raise RuntimeError(f"connection lost after {payload.value}")

    registry = ToolRegistry()
    registry.register(_spec(idempotent=False), runner=runner)
    call = ToolCallPlan.create("tool", {"value": "dispatch"})

    result = await ToolExecutionService(tool_registry=registry).execute_batch(
        _request(call, run_id="execution-unknown"),
        state={},
    )

    assert result.status == "reconciliation_required"
    assert result.tool_results[0].error is not None
    assert result.tool_results[0].error.code == "internal"
    assert result.execution_records[call.tool_call_id].status == "unknown"
    assert calls == 1
    RunRegistry.remove("execution-unknown")


@pytest.mark.anyio
async def test_parallel_batch_requires_idempotent_and_concurrency_safe() -> None:
    active = 0
    max_active = 0

    async def runner(payload: _Input) -> _Output:
        nonlocal active, max_active
        active += 1
        max_active = max(max_active, active)
        await asyncio.sleep(0.01)
        active -= 1
        return _Output(value=payload.value)

    registry = ToolRegistry()
    registry.register(
        _spec("safe_one", idempotent=True, concurrency_safe=True),
        runner=runner,
    )
    registry.register(
        _spec("safe_two", idempotent=True, concurrency_safe=True),
        runner=runner,
    )
    calls = (
        ToolCallPlan.create("safe_one", {"value": "one"}),
        ToolCallPlan.create("safe_two", {"value": "two"}),
    )
    request = ToolBatchRequest(
        calls=calls,
        run_config=_config("execution-parallel", max_parallel_calls=2),
        allowed_tools=frozenset({"safe_one", "safe_two"}),
    )

    result = await ToolExecutionService(tool_registry=registry).execute_batch(
        request,
        state={},
    )

    assert result.status == "completed"
    assert result.pending_tool_calls == ()
    assert max_active == 2
    RunRegistry.remove("execution-parallel")


@pytest.mark.anyio
async def test_non_idempotent_calls_execute_serially_even_if_concurrency_safe() -> None:
    executed: list[str] = []

    def runner(payload: _Input) -> _Output:
        executed.append(payload.value)
        return _Output(value=payload.value)

    registry = ToolRegistry()
    registry.register(
        _spec("write_one", idempotent=False, concurrency_safe=True),
        runner=runner,
    )
    registry.register(
        _spec("write_two", idempotent=False, concurrency_safe=True),
        runner=runner,
    )
    calls = (
        ToolCallPlan.create("write_one", {"value": "one"}),
        ToolCallPlan.create("write_two", {"value": "two"}),
    )
    request = ToolBatchRequest(
        calls=calls,
        run_config=_config("execution-non-idempotent-serial"),
        allowed_tools=frozenset({"write_one", "write_two"}),
    )

    result = await ToolExecutionService(tool_registry=registry).execute_batch(
        request,
        state={},
    )

    assert executed == ["one"]
    assert result.pending_tool_calls == (calls[1],)
    RunRegistry.remove("execution-non-idempotent-serial")


def test_retry_new_operation_requires_human_decision_and_changes_operation_id() -> None:
    call = ToolCallPlan.create("tool", {"value": "retry"})
    record = ToolExecutionRecord(
        tool_call_id=call.tool_call_id,
        tool_name=call.tool_name,
        operation_id="op-old",
        arguments_digest=tool_arguments_digest(call.arguments),
        idempotent=False,
        status="unknown",
        attempt_count=1,
    )

    reconciled = apply_tool_reconciliation(
        record,
        HumanInputResponse(
            request_id="hir-reconcile",
            decision="retry_new_operation",
        ),
    )

    assert reconciled.status == "prepared"
    assert reconciled.operation_id != "op-old"
    assert reconciled.attempt_count == 0


def test_execution_record_summary_is_bounded_and_checkpoint_safe(
    caplog: pytest.LogCaptureFixture,
) -> None:
    record = ToolExecutionRecord(
        tool_call_id="tc-summary",
        tool_name="tool",
        operation_id="op-summary",
        arguments_digest=tool_arguments_digest({"value": "summary"}),
        idempotent=True,
        status="completed",
        attempt_count=1,
        result_summary={
            "status": "ok",
            "output_model": "tests.LargeOutput",
            "output_preview": "x" * 500,
        },
        result_ref="memory://tool-results/tc-summary",
    )
    serde = agent_checkpoint_serde()

    with caplog.at_level(
        logging.WARNING,
        logger="langgraph.checkpoint.serde.jsonplus",
    ):
        restored = serde.loads_typed(serde.dumps_typed(record))

    assert restored == record
    assert len(restored.result_summary.output_preview) == 500
    assert restored.result_ref == "memory://tool-results/tc-summary"
    assert "not allowed" not in caplog.text.lower()

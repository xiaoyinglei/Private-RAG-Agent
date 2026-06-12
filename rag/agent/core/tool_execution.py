from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Mapping
from dataclasses import replace
from hashlib import sha256
from typing import Any, Literal, Protocol
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from rag.agent.core.agent_as_tool import AgentAsToolExecutionError
from rag.agent.core.approval_policy import (
    ApprovalDecision,
    ApprovalPolicy,
    merge_approval_requests,
)
from rag.agent.core.context import AgentRunConfig, RunRegistry
from rag.agent.core.definition import AgentDefinition
from rag.agent.core.human_input import HumanInputRequest, HumanInputResponse
from rag.agent.core.llm_context import AgentLLMContextOverflowError
from rag.agent.memory.models import ContextBudgetSnapshot
from rag.agent.state import AgentState, ToolCallPlan
from rag.agent.tools.rag_tools import RAG_SIGNAL_AWARE_TOOLS
from rag.agent.tools.registry import (
    ToolExecutionContext,
    ToolInputValidationError,
    ToolOutputValidationError,
    ToolRegistry,
    ToolRunnerMissingError,
)
from rag.agent.tools.spec import ToolError, ToolResult, ToolSpec
from rag.schema.query import RetrievalSignals

ToolExecutionStatus = Literal[
    "prepared",
    "started",
    "completed",
    "failed",
    "unknown",
]


class ToolExecutionSummary(BaseModel):
    status: Literal["ok", "error"]
    error_code: str | None = None
    output_model: str | None = None
    output_preview: str | None = Field(default=None, max_length=500)
    work_units_used: int = Field(default=0, ge=0)
    retry_count: int = Field(default=0, ge=0)


class ToolExecutionRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    tool_call_id: str
    tool_name: str
    operation_id: str
    arguments_digest: str
    idempotent: bool
    status: ToolExecutionStatus
    attempt_count: int = Field(default=0, ge=0)
    result_summary: ToolExecutionSummary | None = None
    result_ref: str | None = None
    last_error: ToolError | None = None

    @classmethod
    def prepare(
        cls,
        call: ToolCallPlan,
        spec: ToolSpec,
        *,
        operation_id: str | None = None,
    ) -> ToolExecutionRecord:
        return cls(
            tool_call_id=call.tool_call_id,
            tool_name=call.tool_name,
            operation_id=operation_id or f"op_{uuid4().hex}",
            arguments_digest=tool_arguments_digest(call.arguments),
            idempotent=spec.idempotent,
            status="prepared",
        )


class ToolBatchRequest(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    calls: tuple[ToolCallPlan, ...]
    run_config: AgentRunConfig
    allowed_tools: frozenset[str]
    approved_tool_call_ids: tuple[str, ...] = ()
    denied_tool_call_ids: tuple[str, ...] = ()
    execution_records: dict[str, ToolExecutionRecord] = Field(default_factory=dict)
    retrieval_signals: RetrievalSignals | None = None


class ToolBatchResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    status: Literal["completed", "paused", "reconciliation_required"]
    tool_results: tuple[ToolResult, ...] = ()
    pending_tool_calls: tuple[ToolCallPlan, ...] = ()
    execution_records: dict[str, ToolExecutionRecord] = Field(default_factory=dict)
    skipped_completed_tool_call_ids: tuple[str, ...] = ()
    human_input_request: HumanInputRequest | None = None
    run_config: AgentRunConfig
    context_budget: ContextBudgetSnapshot | None = None
    decision_reason: str | None = None
    record_persistence: Literal["durable", "volatile"] = "volatile"


class ExecutionRecordWriter(Protocol):
    durable: bool

    async def write_execution_record(
        self,
        record: ToolExecutionRecord,
    ) -> None: ...


class VolatileExecutionRecordWriter:
    """Explicitly non-durable writer used by legacy compatibility paths."""

    durable = False

    def __init__(self) -> None:
        self.records: dict[str, ToolExecutionRecord] = {}

    async def write_execution_record(
        self,
        record: ToolExecutionRecord,
    ) -> None:
        self.records[record.tool_call_id] = record.model_copy(deep=True)


class ToolExecutionService:
    def __init__(
        self,
        *,
        tool_registry: ToolRegistry,
        record_writer: ExecutionRecordWriter | None = None,
        approval_policy: ApprovalPolicy | None = None,
        require_idempotent_parallel: bool = True,
        pause_on_ambiguous: bool = True,
    ) -> None:
        self._tool_registry = tool_registry
        self._record_writer = record_writer or VolatileExecutionRecordWriter()
        self._approval_policy = approval_policy or ApprovalPolicy()
        self._require_idempotent_parallel = require_idempotent_parallel
        self._pause_on_ambiguous = pause_on_ambiguous

    async def execute_batch(
        self,
        request: ToolBatchRequest,
        *,
        state: AgentState | None,
        definition: AgentDefinition | None = None,
    ) -> ToolBatchResult:
        records = {
            call_id: record.model_copy(deep=True)
            for call_id, record in request.execution_records.items()
        }
        results: list[ToolResult] = []
        specs: dict[str, ToolSpec] = {}
        candidates: list[ToolCallPlan] = []

        for call in request.calls:
            error = self._preflight_error(call, request=request)
            if error is not None:
                results.append(error)
                continue
            specs[call.tool_name] = self._tool_registry.get(call.tool_name)
            candidates.append(call)

        approved_ids = set(request.approved_tool_call_ids)
        denied_ids = set(request.denied_tool_call_ids)
        executables: list[ToolCallPlan] = []
        approval_decisions: list[ApprovalDecision] = []
        for call in candidates:
            spec = specs[call.tool_name]
            decision = self._approval_policy.decide(
                tool_name=call.tool_name,
                arguments={**call.arguments, "tool_call_id": call.tool_call_id},
                spec=spec,
                requires_confirmation=(
                    call.tool_name
                    in request.run_config.tool_policy.require_confirmation_for
                ),
            )
            if decision.action == "deny" or call.tool_call_id in denied_ids:
                results.append(
                    _error_result(
                        call,
                        code="tool_denied",
                        message=decision.reason,
                        retryable=False,
                    )
                )
            elif decision.action == "ask" and call.tool_call_id not in approved_ids:
                approval_decisions.append(decision)
            else:
                executables.append(call)

        if approval_decisions:
            return self._result(
                status="paused",
                request=request,
                results=results,
                pending=request.calls,
                records=records,
                human_input_request=merge_approval_requests(approval_decisions),
                decision_reason="approval_required",
            )

        recovered: list[ToolCallPlan] = []
        skipped_completed: list[str] = []
        for call in executables:
            existing = records.get(call.tool_call_id)
            if existing is None:
                recovered.append(call)
                continue
            _validate_record_matches_call(existing, call)
            if existing.status == "completed":
                skipped_completed.append(call.tool_call_id)
                continue
            if existing.status == "failed":
                continue
            if existing.status in {"started", "unknown"} and not existing.idempotent:
                unknown = existing.model_copy(update={"status": "unknown"})
                if unknown != existing:
                    await self._write(unknown)
                    records[call.tool_call_id] = unknown
                return self._result(
                    status="reconciliation_required",
                    request=request,
                    results=results,
                    pending=request.calls,
                    records=records,
                    skipped_completed=skipped_completed,
                    human_input_request=_reconciliation_request(unknown),
                    decision_reason="tool_reconciliation",
                )
            recovered.append(call)

        selected, excess = _select_execution_batch(
            recovered,
            specs=specs,
            max_parallel_calls=request.run_config.tool_policy.max_parallel_calls,
            require_idempotent=self._require_idempotent_parallel,
        )
        selected = [
            _execution_call(
                call,
                retrieval_signals=request.retrieval_signals,
            )
            for call in selected
        ]

        started_records: dict[str, ToolExecutionRecord] = {}
        for call in selected:
            spec = specs[call.tool_name]
            record = records.get(call.tool_call_id)
            if record is None:
                record = ToolExecutionRecord.prepare(call, spec)
                await self._write(record)
            started = record.model_copy(
                update={
                    "status": "started",
                    "attempt_count": record.attempt_count + 1,
                    "last_error": None,
                }
            )
            await self._write(started)
            records[call.tool_call_id] = started
            started_records[call.tool_call_id] = started

        executed = await asyncio.gather(
            *[
                self._execute_one(
                    call,
                    record=started_records[call.tool_call_id],
                    spec=specs[call.tool_name],
                    run_config=request.run_config,
                    state=state,
                    definition=definition,
                )
                for call in selected
            ]
        )
        unknown_record: ToolExecutionRecord | None = None
        context_budget: ContextBudgetSnapshot | None = None
        for result, record in executed:
            results.append(result)
            records[record.tool_call_id] = record
            if record.status == "unknown" and unknown_record is None:
                unknown_record = record
            if (
                result.error is not None
                and result.error.code == "context_overflow"
                and isinstance(result.error.detail.get("context_budget"), dict)
            ):
                context_budget = ContextBudgetSnapshot.model_validate(
                    result.error.detail["context_budget"]
                )

        run_config = await _committed_run_config(request.run_config)
        if unknown_record is not None and self._pause_on_ambiguous:
            return self._result(
                status="reconciliation_required",
                request=request,
                results=results,
                pending=tuple(excess),
                records=records,
                skipped_completed=skipped_completed,
                human_input_request=_reconciliation_request(unknown_record),
                run_config=run_config,
                decision_reason="tool_reconciliation",
            )
        if context_budget is not None:
            return self._result(
                status="paused",
                request=request,
                results=results,
                pending=tuple(excess),
                records=records,
                skipped_completed=skipped_completed,
                run_config=run_config,
                context_budget=context_budget,
                decision_reason="context_overflow",
            )
        return self._result(
            status="completed",
            request=request,
            results=results,
            pending=tuple(excess),
            records=records,
            skipped_completed=skipped_completed,
            run_config=run_config,
        )

    def _preflight_error(
        self,
        call: ToolCallPlan,
        *,
        request: ToolBatchRequest,
    ) -> ToolResult | None:
        if call.tool_name in request.run_config.tool_policy.deny_tools:
            return _error_result(
                call,
                code="tool_denied",
                message=f"{call.tool_name} is denied by ToolPolicy",
                retryable=False,
            )
        try:
            spec = self._tool_registry.get(call.tool_name)
        except KeyError:
            return _error_result(
                call,
                code="tool_not_registered",
                message=f"{call.tool_name} is not registered",
                retryable=False,
            )
        if call.tool_name not in request.allowed_tools:
            return _error_result(
                call,
                code="tool_not_allowed",
                message=f"{call.tool_name} is not allowed for this agent",
                retryable=False,
            )
        try:
            spec.input_model.model_validate(call.arguments)
        except ValidationError as exc:
            return _error_result(
                call,
                code="invalid_arguments",
                message=str(exc),
                retryable=False,
                detail={"errors": exc.errors()},
            )
        return None

    async def _execute_one(
        self,
        call: ToolCallPlan,
        *,
        record: ToolExecutionRecord,
        spec: ToolSpec,
        run_config: AgentRunConfig,
        state: AgentState | None,
        definition: AgentDefinition | None,
    ) -> tuple[ToolResult, ToolExecutionRecord]:
        started_at = time.perf_counter()
        work_cost = max(0, spec.work_budget_cost)
        reserved = await _reserve_work_budget(
            call,
            run_config=run_config,
            work_cost=work_cost,
        )
        if isinstance(reserved, ToolResult):
            failed = _record_from_result(record, reserved, status="failed")
            await self._write(failed)
            return reserved, failed

        current = record
        retry_count = 0
        while True:
            try:
                output = await asyncio.wait_for(
                    self._tool_registry.run(
                        call.tool_name,
                        dict(call.arguments),
                        execution_context=ToolExecutionContext(
                            run_config=run_config,
                            operation_id=current.operation_id,
                            tool_call_id=call.tool_call_id,
                            state=state,
                            definition=definition,
                        ),
                    ),
                    timeout=spec.timeout_seconds,
                )
                break
            except ToolInputValidationError as exc:
                await _refund_work_budget(call, run_config, reserved)
                result = _error_result(
                    call,
                    code="invalid_arguments",
                    message=str(exc.validation_error),
                    retryable=False,
                    detail={"errors": exc.errors()},
                    retry_count=retry_count,
                    started_at=started_at,
                )
                failed = _record_from_result(current, result, status="failed")
                await self._write(failed)
                return result, failed
            except ToolRunnerMissingError:
                await _refund_work_budget(call, run_config, reserved)
                result = _error_result(
                    call,
                    code="tool_not_implemented",
                    message=f"{call.tool_name} has no registered callable runner",
                    retryable=False,
                    retry_count=retry_count,
                    started_at=started_at,
                )
                failed = _record_from_result(current, result, status="failed")
                await self._write(failed)
                return result, failed
            except ToolOutputValidationError as exc:
                await _commit_work_budget(
                    call,
                    run_config,
                    reserved,
                    work_cost,
                )
                result = _error_result(
                    call,
                    code="invalid_output",
                    message=str(exc.validation_error),
                    retryable=False,
                    detail={"errors": exc.errors()},
                    work_units_used=work_cost,
                    retry_count=retry_count,
                    started_at=started_at,
                )
                status = _ambiguous_failure_status(spec)
                failed = _record_from_result(current, result, status=status)
                await self._write(failed)
                return result, failed
            except AgentLLMContextOverflowError as exc:
                await _refund_work_budget(call, run_config, reserved)
                result = _error_result(
                    call,
                    code="context_overflow",
                    message=(
                        f"Required context does not fit the {exc.stage.value} "
                        "model budget."
                    ),
                    retryable=False,
                    detail={
                        "context_budget": exc.context_budget.model_dump(mode="json")
                    },
                    retry_count=retry_count,
                    started_at=started_at,
                )
                status = _ambiguous_failure_status(spec)
                failed = _record_from_result(current, result, status=status)
                await self._write(failed)
                return result, failed
            except TimeoutError:
                if spec.idempotent and retry_count < spec.max_retries:
                    retry_count += 1
                    current = current.model_copy(
                        update={
                            "attempt_count": current.attempt_count + 1,
                            "status": "started",
                        }
                    )
                    await self._write(current)
                    continue
                await _commit_work_budget(
                    call,
                    run_config,
                    reserved,
                    work_cost,
                )
                result = _error_result(
                    call,
                    code="timeout",
                    message=(
                        f"{call.tool_name} timed out after "
                        f"{spec.timeout_seconds}s"
                    ),
                    retryable=True,
                    work_units_used=work_cost,
                    retry_count=retry_count,
                    started_at=started_at,
                )
                status = _ambiguous_failure_status(spec)
                failed = _record_from_result(current, result, status=status)
                await self._write(failed)
                return result, failed
            except AgentAsToolExecutionError as exc:
                await _commit_work_budget(
                    call,
                    run_config,
                    reserved,
                    work_cost,
                )
                result = _error_result(
                    call,
                    code="subagent_failed",
                    message=str(exc),
                    retryable=False,
                    detail={
                        "agent_name": exc.agent_name,
                        "status": exc.status,
                        "stop_reason": exc.stop_reason or "",
                    },
                    work_units_used=work_cost,
                    retry_count=retry_count,
                    started_at=started_at,
                )
                failed = _record_from_result(current, result, status="failed")
                await self._write(failed)
                return result, failed
            except Exception as exc:
                if spec.idempotent and retry_count < spec.max_retries:
                    retry_count += 1
                    current = current.model_copy(
                        update={
                            "attempt_count": current.attempt_count + 1,
                            "status": "started",
                        }
                    )
                    await self._write(current)
                    continue
                await _commit_work_budget(
                    call,
                    run_config,
                    reserved,
                    work_cost,
                )
                result = _error_result(
                    call,
                    code="internal",
                    message=str(exc),
                    retryable=True,
                    work_units_used=work_cost,
                    retry_count=retry_count,
                    started_at=started_at,
                )
                status = _ambiguous_failure_status(spec)
                failed = _record_from_result(current, result, status=status)
                await self._write(failed)
                return result, failed

        await _commit_work_budget(call, run_config, reserved, work_cost)
        failure = _structured_output_failure(output)
        if failure is not None:
            result = _error_result(
                call,
                code="tool_failed",
                message=failure["message"],
                retryable=False,
                detail=failure["detail"],
                work_units_used=work_cost,
                retry_count=retry_count,
                started_at=started_at,
            )
            failed = _record_from_result(current, result, status="failed")
            await self._write(failed)
            return result, failed
        result = ToolResult(
            tool_call_id=call.tool_call_id,
            tool_name=call.tool_name,
            status="ok",
            output=output,
            latency_ms=(time.perf_counter() - started_at) * 1000,
            work_units_used=work_cost,
            retry_count=retry_count,
        )
        completed = _record_from_result(current, result, status="completed")
        await self._write(completed)
        return result, completed

    async def _write(self, record: ToolExecutionRecord) -> None:
        await self._record_writer.write_execution_record(record)

    def _result(
        self,
        *,
        status: Literal["completed", "paused", "reconciliation_required"],
        request: ToolBatchRequest,
        results: list[ToolResult],
        pending: tuple[ToolCallPlan, ...],
        records: dict[str, ToolExecutionRecord],
        skipped_completed: list[str] | None = None,
        human_input_request: HumanInputRequest | None = None,
        run_config: AgentRunConfig | None = None,
        context_budget: ContextBudgetSnapshot | None = None,
        decision_reason: str | None = None,
    ) -> ToolBatchResult:
        return ToolBatchResult(
            status=status,
            tool_results=tuple(results),
            pending_tool_calls=pending,
            execution_records=records,
            skipped_completed_tool_call_ids=tuple(skipped_completed or ()),
            human_input_request=human_input_request,
            run_config=run_config or request.run_config,
            context_budget=context_budget,
            decision_reason=decision_reason,
            record_persistence=(
                "durable"
                if self._record_writer.durable
                else "volatile"
            ),
        )


def tool_arguments_digest(arguments: Mapping[str, object]) -> str:
    encoded = json.dumps(
        dict(arguments),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def apply_tool_reconciliation(
    record: ToolExecutionRecord,
    response: HumanInputResponse,
) -> ToolExecutionRecord:
    if response.decision == "mark_completed":
        return record.model_copy(
            update={
                "status": "completed",
                "last_error": None,
            }
        )
    if response.decision == "mark_failed":
        return record.model_copy(update={"status": "failed"})
    if response.decision == "retry_new_operation":
        return record.model_copy(
            update={
                "operation_id": f"op_{uuid4().hex}",
                "status": "prepared",
                "attempt_count": 0,
                "result_summary": None,
                "result_ref": None,
                "last_error": None,
            }
        )
    raise ValueError(
        f"unsupported tool reconciliation decision: {response.decision}"
    )


def _validate_record_matches_call(
    record: ToolExecutionRecord,
    call: ToolCallPlan,
) -> None:
    if record.tool_name != call.tool_name:
        raise ValueError(
            f"execution record tool mismatch for {call.tool_call_id}"
        )
    if record.arguments_digest != tool_arguments_digest(call.arguments):
        raise ValueError(
            f"execution record arguments mismatch for {call.tool_call_id}"
        )


def _reconciliation_request(
    record: ToolExecutionRecord,
) -> HumanInputRequest:
    return HumanInputRequest(
        request_id=f"hir_{uuid4().hex[:12]}",
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


def _select_execution_batch(
    calls: list[ToolCallPlan],
    *,
    specs: dict[str, ToolSpec],
    max_parallel_calls: int,
    require_idempotent: bool,
) -> tuple[list[ToolCallPlan], list[ToolCallPlan]]:
    if not calls:
        return [], []
    limit = max(1, max_parallel_calls)
    first = specs[calls[0].tool_name]
    if not _parallel_safe(first, require_idempotent=require_idempotent):
        return [calls[0]], [
            call.model_copy(deep=True)
            for call in calls[1:]
        ]
    batch: list[ToolCallPlan] = []
    for call in calls:
        if len(batch) >= limit:
            break
        if not _parallel_safe(
            specs[call.tool_name],
            require_idempotent=require_idempotent,
        ):
            break
        batch.append(call)
    return batch, [
        call.model_copy(deep=True)
        for call in calls[len(batch):]
    ]


def _parallel_safe(
    spec: ToolSpec,
    *,
    require_idempotent: bool,
) -> bool:
    return spec.concurrency_safe and (
        spec.idempotent or not require_idempotent
    )


def _execution_call(
    call: ToolCallPlan,
    *,
    retrieval_signals: RetrievalSignals | None,
) -> ToolCallPlan:
    arguments = dict(call.arguments)
    if (
        call.tool_name in RAG_SIGNAL_AWARE_TOOLS
        and "retrieval_signals" not in arguments
    ):
        arguments["retrieval_signals"] = (
            retrieval_signals.model_dump(mode="json")
            if retrieval_signals is not None
            else {}
        )
    return call.model_copy(deep=True, update={"arguments": arguments})


async def _reserve_work_budget(
    call: ToolCallPlan,
    *,
    run_config: AgentRunConfig,
    work_cost: int,
) -> bool | ToolResult:
    if work_cost <= 0:
        return False
    try:
        handles = RunRegistry.get(run_config.run_id)
    except KeyError:
        return _error_result(
            call,
            code="runtime_handles_missing",
            message=f"Runtime handles missing for run_id={run_config.run_id}",
            retryable=False,
        )
    reserved = await handles.tool_work_ledger.reserve(
        call.tool_call_id,
        work_cost,
    )
    if reserved:
        return True
    return _error_result(
        call,
        code="budget_exhausted",
        message=f"Insufficient budget to execute {call.tool_name}",
        retryable=False,
        detail={"required": work_cost},
    )


async def _refund_work_budget(
    call: ToolCallPlan,
    run_config: AgentRunConfig,
    reserved: bool,
) -> None:
    if reserved:
        await RunRegistry.get(run_config.run_id).tool_work_ledger.refund(
            call.tool_call_id
        )


async def _commit_work_budget(
    call: ToolCallPlan,
    run_config: AgentRunConfig,
    reserved: bool,
    work_cost: int,
) -> None:
    if reserved:
        await RunRegistry.get(run_config.run_id).tool_work_ledger.commit(
            call.tool_call_id,
            work_cost,
        )


async def _committed_run_config(
    run_config: AgentRunConfig,
) -> AgentRunConfig:
    try:
        committed = await RunRegistry.get(
            run_config.run_id
        ).budget_ledger.committed()
    except KeyError:
        return run_config
    return replace(run_config, budget_committed=committed)


def _ambiguous_failure_status(
    spec: ToolSpec,
) -> Literal["failed", "unknown"]:
    return "failed" if spec.idempotent else "unknown"


def _record_from_result(
    record: ToolExecutionRecord,
    result: ToolResult,
    *,
    status: Literal["completed", "failed", "unknown"],
) -> ToolExecutionRecord:
    output = result.output
    preview: str | None = None
    output_model: str | None = None
    if output is not None:
        output_model = f"{output.__class__.__module__}.{output.__class__.__qualname__}"
        preview = output.model_dump_json(exclude_none=True)[:500]
    summary = ToolExecutionSummary(
        status=result.status,
        error_code=result.error.code if result.error is not None else None,
        output_model=output_model,
        output_preview=preview,
        work_units_used=result.work_units_used,
        retry_count=result.retry_count,
    )
    return record.model_copy(
        update={
            "status": status,
            "result_summary": summary,
            "last_error": result.error,
        }
    )


def _structured_output_failure(
    output: object,
) -> dict[str, Any] | None:
    if getattr(output, "ok", None) is not False:
        return None
    detail: dict[str, Any] = {}
    if hasattr(output, "model_dump"):
        raw_detail = output.model_dump(mode="json")
        if isinstance(raw_detail, dict):
            detail = raw_detail
    stderr = detail.get("stderr")
    stdout = detail.get("stdout")
    exit_code = detail.get("exit_code")
    if isinstance(stderr, str) and stderr.strip():
        message = stderr.strip().splitlines()[0]
    elif isinstance(stdout, str) and stdout.strip():
        message = stdout.strip().splitlines()[0]
    elif isinstance(exit_code, int):
        message = f"Tool returned ok=False with exit_code={exit_code}"
    else:
        message = "Tool returned ok=False"
    return {"message": message, "detail": detail}


def _error_result(
    call: ToolCallPlan,
    *,
    code: str,
    message: str,
    retryable: bool,
    detail: dict[str, Any] | None = None,
    work_units_used: int = 0,
    retry_count: int = 0,
    started_at: float | None = None,
) -> ToolResult:
    latency_ms = (
        0
        if started_at is None
        else (time.perf_counter() - started_at) * 1000
    )
    return ToolResult(
        tool_call_id=call.tool_call_id,
        tool_name=call.tool_name,
        status="error",
        error=ToolError(
            code=code,
            message=message,
            retryable=retryable,
            detail=detail or {},
        ),
        latency_ms=latency_ms,
        work_units_used=work_units_used,
        retry_count=retry_count,
    )


__all__ = [
    "ExecutionRecordWriter",
    "ToolBatchRequest",
    "ToolBatchResult",
    "ToolExecutionRecord",
    "ToolExecutionService",
    "ToolExecutionStatus",
    "ToolExecutionSummary",
    "VolatileExecutionRecordWriter",
    "apply_tool_reconciliation",
    "tool_arguments_digest",
]

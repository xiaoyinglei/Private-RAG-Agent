from __future__ import annotations

import asyncio
import time
from typing import Any

from pydantic import ValidationError

from rag.agent.core.agent_as_tool import AgentAsToolExecutionError
from rag.agent.core.approval_policy import ApprovalDecision, ApprovalPolicy, merge_approval_requests
from rag.agent.core.context import AgentRunConfig, RunRegistry
from rag.agent.goal_runtime import ObservationExtractor
from rag.agent.memory.compactor import MemoryCompactor
from rag.agent.planning import PlanTracker
from rag.agent.state import AgentState, ToolCallPlan, _merge_tool_results
from rag.agent.tools.rag_tools import RAG_SIGNAL_AWARE_TOOLS
from rag.agent.tools.registry import (
    ToolInputValidationError,
    ToolOutputValidationError,
    ToolRegistry,
    ToolRunnerMissingError,
)
from rag.agent.tools.spec import ToolError, ToolResult, ToolSpec


async def run_tools_raw(
    state: AgentState,
    *,
    tool_registry: ToolRegistry,
    allowed_tools: frozenset[str],
) -> dict[str, Any]:
    pending = state.get("pending_tool_calls", [])

    if not pending:
        return {}

    tool_policy = state["run_config"].tool_policy
    results: list[ToolResult] = []
    rest: list[ToolCallPlan] = []
    specs_by_name: dict[str, ToolSpec] = {}

    for call in pending:
        if call.tool_name in tool_policy.deny_tools:
            results.append(
                ToolResult(
                    tool_call_id=call.tool_call_id,
                    tool_name=call.tool_name,
                    status="error",
                    error=ToolError(
                        code="tool_denied",
                        message=f"{call.tool_name} is denied by ToolPolicy",
                        retryable=False,
                    ),
                    latency_ms=0,
                )
            )
            continue
        try:
            specs_by_name[call.tool_name] = tool_registry.get(call.tool_name)
        except KeyError:
            results.append(
                ToolResult(
                    tool_call_id=call.tool_call_id,
                    tool_name=call.tool_name,
                    status="error",
                    error=ToolError(
                        code="tool_not_registered",
                        message=f"{call.tool_name} is not registered",
                        retryable=False,
                    ),
                    latency_ms=0,
                )
            )
            continue
        if call.tool_name not in allowed_tools:
            results.append(
                ToolResult(
                    tool_call_id=call.tool_call_id,
                    tool_name=call.tool_name,
                    status="error",
                    error=ToolError(
                        code="tool_not_allowed",
                        message=f"{call.tool_name} is not allowed for this agent",
                        retryable=False,
                    ),
                    latency_ms=0,
                )
            )
            continue
        rest.append(call)

    approved_ids = set(state.get("approved_tool_call_ids", []))
    denied_ids = set(state.get("denied_tool_call_ids", []))
    approval_policy = ApprovalPolicy()

    executables: list[ToolCallPlan] = []
    ask_decisions: list[ApprovalDecision] = []

    for call in rest:
        spec = specs_by_name.get(call.tool_name)
        approval_arguments = {**call.arguments, "tool_call_id": call.tool_call_id}
        decision = approval_policy.decide(
            tool_name=call.tool_name, arguments=approval_arguments, spec=spec,
        )

        if decision.action == "deny":
            results.append(
                ToolResult(
                    tool_call_id=call.tool_call_id,
                    tool_name=call.tool_name,
                    status="error",
                    error=ToolError(code="tool_denied", message=decision.reason, retryable=False),
                    latency_ms=0,
                )
            )
        elif decision.action == "ask":
            if call.tool_call_id in approved_ids:
                executables.append(call)
            elif call.tool_call_id in denied_ids:
                results.append(
                    ToolResult(
                        tool_call_id=call.tool_call_id,
                        tool_name=call.tool_name,
                        status="error",
                        error=ToolError(code="tool_denied", message=decision.reason, retryable=False),
                        latency_ms=0,
                    )
                )
            else:
                ask_decisions.append(decision)
        else:  # allow
            if call.tool_call_id in denied_ids:
                results.append(
                    ToolResult(
                        tool_call_id=call.tool_call_id,
                        tool_name=call.tool_name,
                        status="error",
                        error=ToolError(code="tool_denied", message=decision.reason, retryable=False),
                        latency_ms=0,
                    )
                )
            else:
                executables.append(call)

    # 有新的 ASK 工具 → 保守暂停，不执行任何工具
    if ask_decisions:
        request = merge_approval_requests(ask_decisions)
        return {
            "status": "paused",
            "human_input_request": request,
            "needs_user_input": request.question,
            "pending_tool_calls": pending,
            "tool_results": results,
        }

    # 全部允许 / 已批准 → 执行
    signals = state.get("retrieval_signals")
    batch = [
        _execution_call(call, retrieval_signals=signals)
        for call in executables[: tool_policy.max_parallel_calls]
    ]
    excess = [
        call.model_copy(deep=True)
        for call in executables[tool_policy.max_parallel_calls :]
    ]

    gathered = await asyncio.gather(
        *[
            _execute_one_tool(call, run_config=state["run_config"], tool_registry=tool_registry)
            for call in batch
        ],
        return_exceptions=True,
    )
    for index, result_or_exc in enumerate(gathered):
        if isinstance(result_or_exc, BaseException):
            results.append(
                ToolResult(
                    tool_call_id=batch[index].tool_call_id,
                    tool_name=batch[index].tool_name,
                    status="error",
                    error=ToolError(code="internal", message=str(result_or_exc), retryable=True),
                    latency_ms=0,
                )
            )
        else:
            results.append(result_or_exc)

    return {"tool_results": results, "pending_tool_calls": excess}


async def run_tools_guarded(
    state: AgentState,
    *,
    tool_registry: ToolRegistry,
    allowed_tools: frozenset[str],
) -> dict[str, Any]:
    """Execute pending tools and return only observation-reduced, compacted state.

    LangGraph checkpoints node outputs. This wrapper keeps raw tool outputs local
    to the node so observation extraction can use them before large payloads are
    externalized.
    """

    raw_update = await run_tools_raw(
        state,
        tool_registry=tool_registry,
        allowed_tools=allowed_tools,
    )
    if not raw_update:
        return raw_update

    # Approval pauses do not execute tools, so there is no raw tool output to
    # externalize on this path. Preserve existing interrupt/resume semantics.
    if raw_update.get("status") == "paused":
        return raw_update

    update = dict(raw_update)
    transient_state = dict(state)
    raw_tool_results = list(raw_update.get("tool_results", []))
    if raw_tool_results:
        transient_state["tool_results"] = _merge_tool_results(
            list(state.get("tool_results", [])),
            raw_tool_results,
        )

    observation_update = ObservationExtractor().reduce_tool_results(transient_state)
    if observation_update:
        plan, events = PlanTracker().record_observation_progress(
            state.get("agent_plan"),
            observations=observation_update.get("structured_observations", []),
            satisfied_requirement_ids=observation_update.get("satisfied_requirements", []),
        )
        update.update(observation_update)
        if plan is not None:
            update["agent_plan"] = plan
        if events:
            update["plan_events"] = events

    try:
        memory_store = RunRegistry.get(state["run_config"].run_id).memory_store
    except KeyError:
        memory_store = None
    return MemoryCompactor(
        policy=state["run_config"].memory_policy,
        store=memory_store,
    ).compact_update(dict(state), update)


def _execution_call(
    call: ToolCallPlan,
    *,
    retrieval_signals: object,
) -> ToolCallPlan:
    arguments = dict(call.arguments)
    if call.tool_name in RAG_SIGNAL_AWARE_TOOLS and "retrieval_signals" not in arguments:
        arguments["retrieval_signals"] = (
            retrieval_signals.model_dump(mode="json")
            if hasattr(retrieval_signals, "model_dump")
            else {}
        )
    return call.model_copy(deep=True, update={"arguments": arguments})


async def _execute_one_tool(
    call: ToolCallPlan,
    *,
    run_config: AgentRunConfig,
    tool_registry: ToolRegistry,
) -> ToolResult:
    started_at = time.perf_counter()
    try:
        spec = tool_registry.get(call.tool_name)
    except KeyError:
        return _error_result(
            call,
            code="tool_not_registered",
            message=f"{call.tool_name} is not registered",
            retryable=False,
            started_at=started_at,
        )

    arguments = dict(call.arguments)
    try:
        spec.input_model.model_validate(arguments)
    except ValidationError as exc:
        return _error_result(
            call,
            code="invalid_arguments",
            message=str(exc),
            retryable=False,
            started_at=started_at,
            detail={"errors": exc.errors()},
        )

    if not tool_registry.has_runner(call.tool_name):
        return _error_result(
            call,
            code="tool_not_implemented",
            message=f"{call.tool_name} has no registered callable runner",
            retryable=False,
            started_at=started_at,
        )

    budget_cost = max(0, spec.token_budget_cost)
    reserved_budget = False
    if budget_cost > 0:
        try:
            handles = RunRegistry.get(run_config.run_id)
        except KeyError:
            return _error_result(
                call,
                code="runtime_handles_missing",
                message=f"Runtime handles missing for run_id={run_config.run_id}",
                retryable=False,
                started_at=started_at,
            )
        reserved_budget = await handles.budget_ledger.reserve(call.tool_call_id, budget_cost)
        if not reserved_budget:
            return _error_result(
                call,
                code="budget_exhausted",
                message=f"Insufficient budget to execute {call.tool_name}",
                retryable=False,
                started_at=started_at,
                detail={"required": budget_cost},
            )

    attempt = 0
    while True:
        try:
            output = await asyncio.wait_for(
                tool_registry.run(call.tool_name, arguments),
                timeout=spec.timeout_seconds,
            )
            break
        except ToolInputValidationError as exc:
            if reserved_budget:
                await RunRegistry.get(run_config.run_id).budget_ledger.refund(call.tool_call_id)
            return _error_result(
                call,
                code="invalid_arguments",
                message=str(exc.validation_error),
                retryable=False,
                started_at=started_at,
                detail={"errors": exc.errors()},
                retry_count=attempt,
            )
        except ToolRunnerMissingError:
            if reserved_budget:
                await RunRegistry.get(run_config.run_id).budget_ledger.refund(call.tool_call_id)
            return _error_result(
                call,
                code="tool_not_implemented",
                message=f"{call.tool_name} has no registered callable runner",
                retryable=False,
                started_at=started_at,
                retry_count=attempt,
            )
        except ToolOutputValidationError as exc:
            if reserved_budget:
                await RunRegistry.get(run_config.run_id).budget_ledger.commit(
                    call.tool_call_id,
                    budget_cost,
                )
            return _error_result(
                call,
                code="invalid_output",
                message=str(exc.validation_error),
                retryable=False,
                started_at=started_at,
                detail={"errors": exc.errors()},
                token_used=budget_cost,
                retry_count=attempt,
            )
        except TimeoutError:
            if attempt < spec.max_retries:
                attempt += 1
                continue
            if reserved_budget:
                await RunRegistry.get(run_config.run_id).budget_ledger.commit(
                    call.tool_call_id,
                    budget_cost,
                )
            return _error_result(
                call,
                code="timeout",
                message=f"{call.tool_name} timed out after {spec.timeout_seconds}s",
                retryable=True,
                started_at=started_at,
                token_used=budget_cost,
                retry_count=attempt,
            )
        except AgentAsToolExecutionError as exc:
            if reserved_budget:
                await RunRegistry.get(run_config.run_id).budget_ledger.commit(
                    call.tool_call_id,
                    budget_cost,
                )
            return _error_result(
                call,
                code="subagent_failed",
                message=str(exc),
                retryable=False,
                started_at=started_at,
                detail={
                    "agent_name": exc.agent_name,
                    "status": exc.status,
                    "stop_reason": exc.stop_reason or "",
                },
                token_used=budget_cost,
                retry_count=attempt,
            )
        except Exception as exc:
            if attempt < spec.max_retries:
                attempt += 1
                continue
            if reserved_budget:
                await RunRegistry.get(run_config.run_id).budget_ledger.commit(
                    call.tool_call_id,
                    budget_cost,
                )
            return _error_result(
                call,
                code="internal",
                message=str(exc),
                retryable=True,
                started_at=started_at,
                token_used=budget_cost,
                retry_count=attempt,
            )

    if reserved_budget:
        await RunRegistry.get(run_config.run_id).budget_ledger.commit(
            call.tool_call_id,
            budget_cost,
        )
    failure = _structured_output_failure(output)
    if failure is not None:
        return _error_result(
            call,
            code="tool_failed",
            message=failure["message"],
            retryable=False,
            started_at=started_at,
            detail=failure["detail"],
            token_used=budget_cost,
            retry_count=attempt,
        )
    return ToolResult(
        tool_call_id=call.tool_call_id,
        tool_name=call.tool_name,
        status="ok",
        output=output,
        latency_ms=(time.perf_counter() - started_at) * 1000,
        token_used=budget_cost,
        retry_count=attempt,
    )


def _structured_output_failure(output: object) -> dict[str, Any] | None:
    ok = getattr(output, "ok", None)
    if ok is not False:
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
    started_at: float,
    detail: dict[str, Any] | None = None,
    token_used: int = 0,
    retry_count: int = 0,
) -> ToolResult:
    return ToolResult(
        tool_call_id=call.tool_call_id,
        tool_name=call.tool_name,
        status="error",
        error=ToolError(code=code, message=message, retryable=retryable, detail=detail or {}),
        latency_ms=(time.perf_counter() - started_at) * 1000,
        token_used=token_used,
        retry_count=retry_count,
    )


execute_node = run_tools_raw
execute_observe_compact_node = run_tools_guarded

__all__ = [
    "execute_node",
    "execute_observe_compact_node",
    "run_tools_guarded",
    "run_tools_raw",
]

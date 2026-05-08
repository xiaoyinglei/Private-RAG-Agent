from __future__ import annotations

import asyncio
import time
from typing import Any

from pydantic import ValidationError

from rag.agent.core.context import AgentRunConfig, RuntimeRegistry
from rag.agent.state import AgentState, ToolCallPlan
from rag.agent.tools.registry import (
    ToolInputValidationError,
    ToolOutputValidationError,
    ToolRegistry,
    ToolRunnerMissingError,
)
from rag.agent.tools.spec import ToolError, ToolResult, ToolSpec


async def execute_node(
    state: AgentState,
    *,
    tool_registry: ToolRegistry,
    allowed_tools: frozenset[str],
) -> dict:
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

    confirmed = state.get("confirmed_tool_call_ids", set())
    needs_confirmation = [
        call
        for call in rest
        if (
            call.tool_name in tool_policy.require_confirmation_for
            or specs_by_name[call.tool_name].requires_confirmation
        )
        and call.tool_call_id not in confirmed
    ]
    if needs_confirmation:
        return {
            "status": "paused",
            "needs_user_input": f"Confirm tool execution: {[call.tool_name for call in needs_confirmation]}",
            "pending_tool_calls": needs_confirmation,
            "tool_results": results,
        }

    executables = rest[: tool_policy.max_parallel_calls]
    excess = rest[tool_policy.max_parallel_calls :]

    gathered = await asyncio.gather(
        *[
            _execute_one_tool(call, run_config=state["run_config"], tool_registry=tool_registry)
            for call in executables
        ],
        return_exceptions=True,
    )
    for index, result_or_exc in enumerate(gathered):
        if isinstance(result_or_exc, Exception):
            results.append(
                ToolResult(
                    tool_call_id=executables[index].tool_call_id,
                    tool_name=executables[index].tool_name,
                    status="error",
                    error=ToolError(code="internal", message=str(result_or_exc), retryable=True),
                    latency_ms=0,
                )
            )
        else:
            results.append(result_or_exc)

    return {"tool_results": results, "pending_tool_calls": excess}


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

    try:
        spec.input_model.model_validate(call.arguments)
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
            handles = RuntimeRegistry.get(run_config.run_id)
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

    try:
        output = await tool_registry.run(call.tool_name, call.arguments)
    except ToolInputValidationError as exc:
        if reserved_budget:
            await RuntimeRegistry.get(run_config.run_id).budget_ledger.refund(call.tool_call_id)
        return _error_result(
            call,
            code="invalid_arguments",
            message=str(exc.validation_error),
            retryable=False,
            started_at=started_at,
            detail={"errors": exc.errors()},
        )
    except ToolRunnerMissingError:
        if reserved_budget:
            await RuntimeRegistry.get(run_config.run_id).budget_ledger.refund(call.tool_call_id)
        return _error_result(
            call,
            code="tool_not_implemented",
            message=f"{call.tool_name} has no registered callable runner",
            retryable=False,
            started_at=started_at,
        )
    except ToolOutputValidationError as exc:
        if reserved_budget:
            await RuntimeRegistry.get(run_config.run_id).budget_ledger.commit(
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
        )
    except Exception as exc:
        if reserved_budget:
            await RuntimeRegistry.get(run_config.run_id).budget_ledger.commit(
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
        )

    if reserved_budget:
        await RuntimeRegistry.get(run_config.run_id).budget_ledger.commit(
            call.tool_call_id,
            budget_cost,
        )
    return ToolResult(
        tool_call_id=call.tool_call_id,
        tool_name=call.tool_name,
        status="ok",
        output=output,
        latency_ms=(time.perf_counter() - started_at) * 1000,
        token_used=budget_cost,
    )


def _error_result(
    call: ToolCallPlan,
    *,
    code: str,
    message: str,
    retryable: bool,
    started_at: float,
    detail: dict[str, Any] | None = None,
    token_used: int = 0,
) -> ToolResult:
    return ToolResult(
        tool_call_id=call.tool_call_id,
        tool_name=call.tool_name,
        status="error",
        error=ToolError(code=code, message=message, retryable=retryable, detail=detail or {}),
        latency_ms=(time.perf_counter() - started_at) * 1000,
        token_used=token_used,
    )

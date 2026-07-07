"""Execution choke point for model tool calls."""

from __future__ import annotations

import asyncio
import inspect
import json
import time
from collections.abc import Callable
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from rag.agent.tooling.registry import ToolRegistry, ToolRunner
from rag.agent.tooling.spec import ToolCall, ToolResult, ToolRisk, ToolSpec
from rag.agent.tooling.trace import ToolExecutionTrace
from rag.agent.workspace import WorkspacePathError


class CanUseToolResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decision: Literal["allow", "ask", "deny"]
    reason: str


CanUseToolFn = Callable[..., CanUseToolResult | dict[str, Any]]


def canUseTool(
    spec: ToolSpec,
    call: ToolCall,
    *,
    allow_write_tools: bool = False,
    allow_execute_tools: bool = False,
) -> CanUseToolResult:
    del call
    if spec.risk == ToolRisk.READ:
        return CanUseToolResult(decision="allow", reason="read tools are allowed")
    if spec.risk == ToolRisk.WRITE:
        if allow_write_tools:
            return CanUseToolResult(decision="allow", reason="write tools are allowed")
        return CanUseToolResult(decision="ask", reason="write tools require entry config")
    if spec.risk == ToolRisk.EXECUTE:
        if allow_execute_tools:
            return CanUseToolResult(decision="allow", reason="execute tools are allowed")
        return CanUseToolResult(decision="ask", reason="execute tools require entry config")
    if spec.risk == ToolRisk.NETWORK:
        return CanUseToolResult(decision="deny", reason="network tools are denied by default")
    if spec.risk == ToolRisk.DESTRUCTIVE:
        return CanUseToolResult(
            decision="deny",
            reason="destructive tools are denied by default",
        )
    return CanUseToolResult(decision="deny", reason=f"unsupported tool risk: {spec.risk}")


class ToolExecutor:
    """Validate and execute model tool calls against the sent schema surface."""

    def __init__(
        self,
        registry: ToolRegistry,
        *,
        allow_write_tools: bool = False,
        allow_execute_tools: bool = False,
        can_use_tool: CanUseToolFn | None = None,
    ) -> None:
        self._registry = registry
        self._allow_write_tools = allow_write_tools
        self._allow_execute_tools = allow_execute_tools
        self._can_use_tool = can_use_tool or canUseTool
        self.traces: list[ToolExecutionTrace] = []

    async def execute(
        self,
        call: ToolCall,
        *,
        sent_schema_names: list[str],
    ) -> ToolResult:
        start = time.monotonic()
        spec = self._registry.get(call.name)
        if spec is None:
            return self._record_result(
                start,
                ToolResult(
                    tool_call_id=call.id,
                    tool_name=call.name,
                    ok=False,
                    content=f"Unknown tool: {call.name}",
                    recoverable=True,
                    error_code="unknown_tool",
                ),
            )

        if call.name not in set(sent_schema_names):
            return self._record_result(
                start,
                ToolResult(
                    tool_call_id=call.id,
                    tool_name=call.name,
                    ok=False,
                    content=f"Tool schema was not sent for this request: {call.name}",
                    recoverable=True,
                    error_code="schema_not_sent",
                ),
            )

        validation_errors = _validate_arguments(spec.input_schema, call.arguments)
        if validation_errors:
            return self._record_result(
                start,
                ToolResult(
                    tool_call_id=call.id,
                    tool_name=call.name,
                    ok=False,
                    content="; ".join(validation_errors),
                    recoverable=True,
                    error_code="invalid_arguments",
                ),
            )

        can_use_tool_result = CanUseToolResult.model_validate(
            self._can_use_tool(
                spec,
                call,
                allow_write_tools=self._allow_write_tools,
                allow_execute_tools=self._allow_execute_tools,
            )
        )
        if can_use_tool_result.decision != "allow":
            return self._record_result(
                start,
                _can_use_tool_denied_result(call, can_use_tool_result),
                can_use_tool_result=can_use_tool_result,
            )

        runner = self._registry.get_runner(call.name)
        if runner is None:
            return self._record_result(
                start,
                ToolResult(
                    tool_call_id=call.id,
                    tool_name=call.name,
                    ok=False,
                    content=f"No runner installed for tool: {call.name}",
                    recoverable=True,
                    error_code="runner_missing",
                ),
                can_use_tool_result=can_use_tool_result,
            )

        try:
            raw = await asyncio.wait_for(
                _invoke_runner(runner, call.arguments),
                timeout=spec.timeout_seconds,
            )
        except asyncio.TimeoutError:
            return self._record_result(
                start,
                ToolResult(
                    tool_call_id=call.id,
                    tool_name=call.name,
                    ok=False,
                    content=f"Tool timed out after {spec.timeout_seconds:.1f}s",
                    recoverable=True,
                    error_code="timeout",
                ),
                can_use_tool_result=can_use_tool_result,
            )
        except Exception as exc:
            return self._record_result(
                start,
                _exception_result(call, exc),
                can_use_tool_result=can_use_tool_result,
            )

        return self._record_result(
            start,
            _success_result(call, spec, raw),
            can_use_tool_result=can_use_tool_result,
        )

    def _record_result(
        self,
        start: float,
        result: ToolResult,
        *,
        can_use_tool_result: CanUseToolResult | None = None,
    ) -> ToolResult:
        self.traces.append(
            ToolExecutionTrace(
                tool_call_id=result.tool_call_id,
                tool_name=result.tool_name,
                status="ok" if result.ok else "error",
                recoverable=result.recoverable,
                error_code=result.error_code,
                can_use_tool_decision=(
                    can_use_tool_result.decision if can_use_tool_result else None
                ),
                can_use_tool_reason=(
                    can_use_tool_result.reason if can_use_tool_result else None
                ),
                latency_ms=(time.monotonic() - start) * 1000,
            )
        )
        return result


class ToolExecutorLoopAdapter:
    """Adapter from the existing AgentLoop runner protocol to ToolExecutor."""

    def __init__(self, executor: ToolExecutor) -> None:
        self._executor = executor

    async def execute_batch(
        self,
        request: Any,
        *,
        state: Any,
        definition: Any = None,
    ) -> Any:
        del definition
        from rag.agent.core.tool_execution import ToolBatchResult

        sent_schema_names = _state_sent_schema_names(state)
        legacy_results = []
        for plan in request.calls:
            result = await self._executor.execute(
                ToolCall(
                    id=plan.tool_call_id,
                    name=plan.tool_name,
                    arguments=dict(plan.arguments),
                ),
                sent_schema_names=sent_schema_names,
            )
            legacy_results.append(_to_legacy_result(result, self._last_latency_ms()))

        return ToolBatchResult(
            status="completed",
            tool_results=tuple(legacy_results),
            pending_tool_calls=(),
            execution_records=dict(getattr(request, "execution_records", {}) or {}),
            run_config=request.run_config,
            record_persistence="volatile",
        )

    def _last_latency_ms(self) -> float:
        if not self._executor.traces:
            return 0.0
        return self._executor.traces[-1].latency_ms


class ToolingAdapterOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    content: str
    data: dict[str, Any] = Field(default_factory=dict)


async def _invoke_runner(runner: ToolRunner, args: dict[str, Any]) -> Any:
    if inspect.iscoroutinefunction(runner):
        return await runner(args)
    return await asyncio.to_thread(runner, args)


def _state_sent_schema_names(state: Any) -> list[str]:
    if isinstance(state, dict):
        raw = state.get("tooling_sent_schema_names", [])
    else:
        raw = getattr(state, "tooling_sent_schema_names", [])
    if not isinstance(raw, list):
        return []
    return [str(name) for name in raw]


def _to_legacy_result(result: ToolResult, latency_ms: float) -> Any:
    from rag.agent.tools.spec import (
        ToolError as LegacyToolError,
        ToolResult as LegacyToolResult,
    )

    if result.ok:
        return LegacyToolResult(
            tool_call_id=result.tool_call_id,
            tool_name=result.tool_name,
            status="ok",
            output=ToolingAdapterOutput(content=result.content, data=result.data),
            latency_ms=latency_ms,
        )
    return LegacyToolResult(
        tool_call_id=result.tool_call_id,
        tool_name=result.tool_name,
        status="error",
        error=LegacyToolError(
            code=result.error_code or "tool_error",
            message=result.content,
            retryable=result.recoverable,
            detail=result.data,
        ),
        latency_ms=latency_ms,
    )


def _success_result(call: ToolCall, spec: ToolSpec, raw: Any) -> ToolResult:
    if isinstance(raw, ToolResult):
        return raw
    data = _to_data(raw, spec.output_limit_chars)
    if isinstance(raw, dict) and isinstance(raw.get("content"), str):
        content = _limit_text(raw["content"], spec.output_limit_chars)
    else:
        content = _limit_text(
            json.dumps(data, ensure_ascii=False, sort_keys=True, default=str),
            spec.output_limit_chars,
        )
    return ToolResult(
        tool_call_id=call.id,
        tool_name=call.name,
        ok=True,
        content=content,
        data=data,
        recoverable=True,
    )


def _can_use_tool_denied_result(
    call: ToolCall,
    decision: CanUseToolResult,
) -> ToolResult:
    error_code = (
        "permission_required"
        if decision.decision == "ask"
        else "permission_denied"
    )
    return ToolResult(
        tool_call_id=call.id,
        tool_name=call.name,
        ok=False,
        content=decision.reason,
        data={"can_use_tool": decision.model_dump()},
        recoverable=True,
        error_code=error_code,
    )


def _exception_result(call: ToolCall, exc: Exception) -> ToolResult:
    error_code = getattr(exc, "error_code", None)
    if not error_code:
        error_code = _error_code_for_exception(exc)
    return ToolResult(
        tool_call_id=call.id,
        tool_name=call.name,
        ok=False,
        content=str(exc),
        recoverable=True,
        error_code=error_code,
    )


def _error_code_for_exception(exc: Exception) -> str:
    if isinstance(exc, FileNotFoundError):
        return "file_not_found"
    if isinstance(exc, WorkspacePathError):
        return "permission_denied"
    if isinstance(exc, NotADirectoryError):
        return "invalid_arguments"
    if isinstance(exc, (ValidationError, ValueError, TypeError)):
        return "invalid_arguments"
    if isinstance(exc, PermissionError):
        return "permission_denied"
    return "runner_error"


def _to_data(raw: Any, limit: int) -> dict[str, Any]:
    if isinstance(raw, BaseModel):
        return _limit_value(raw.model_dump(), limit)
    if isinstance(raw, dict):
        return _limit_value(raw, limit)
    return {"value": _limit_value(raw, limit)}


def _limit_value(value: Any, limit: int) -> Any:
    if isinstance(value, str):
        return _limit_text(value, limit)
    if isinstance(value, list):
        return [_limit_value(item, limit) for item in value]
    if isinstance(value, tuple):
        return [_limit_value(item, limit) for item in value]
    if isinstance(value, dict):
        return {str(key): _limit_value(item, limit) for key, item in value.items()}
    return value


def _limit_text(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    return value[:limit] + "\n[truncated]"


def _validate_arguments(schema: dict[str, Any], args: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if schema.get("type") and schema.get("type") != "object":
        return errors
    required = schema.get("required") or []
    for name in required:
        if name not in args:
            errors.append(f"Missing required argument: {name}")
    properties = schema.get("properties") or {}
    if schema.get("additionalProperties") is False:
        for name in args:
            if name not in properties:
                errors.append(f"Unexpected argument: {name}")
    for name, value in args.items():
        prop_schema = properties.get(name)
        if not isinstance(prop_schema, dict):
            continue
        expected_type = prop_schema.get("type")
        if expected_type and not _matches_json_type(value, expected_type):
            errors.append(f"Argument {name} must be {expected_type}")
    return errors


def _matches_json_type(value: Any, expected_type: Any) -> bool:
    if isinstance(expected_type, list):
        return any(_matches_json_type(value, item) for item in expected_type)
    if expected_type == "string":
        return isinstance(value, str)
    if expected_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected_type == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected_type == "boolean":
        return isinstance(value, bool)
    if expected_type == "array":
        return isinstance(value, list)
    if expected_type == "object":
        return isinstance(value, dict)
    if expected_type == "null":
        return value is None
    return True

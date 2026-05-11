from __future__ import annotations

import asyncio

import pytest
from pydantic import BaseModel

from rag.agent.core.context import AgentRunConfig, RuntimeRegistry
from rag.agent.graphs.nodes.execute import execute_node
from rag.agent.state import AgentState, ToolCallPlan
from rag.agent.tools.registry import ToolRegistry
from rag.agent.tools.spec import ToolError, ToolPermissions, ToolSpec
from rag.schema.runtime import AccessPolicy, ExecutionLocationPreference


class RuntimeInput(BaseModel):
    value: str


class RuntimeOutput(BaseModel):
    value: str


def _runtime_spec(*, timeout_seconds: float = 1.0, max_retries: int = 0) -> ToolSpec:
    return ToolSpec(
        name="runtime_tool",
        description="Runtime behavior test tool",
        input_model=RuntimeInput,
        output_model=RuntimeOutput,
        error_model=ToolError,
        permissions=ToolPermissions(),
        timeout_seconds=timeout_seconds,
        max_retries=max_retries,
    )


def _state(*, call: ToolCallPlan, run_id: str) -> AgentState:
    config = AgentRunConfig(
        run_id=run_id,
        thread_id=run_id,
        budget_total=100,
        max_depth=2,
        access_policy=AccessPolicy.default(),
        execution_location_preference=ExecutionLocationPreference.LOCAL_FIRST,
    )
    RuntimeRegistry.remove(run_id)
    RuntimeRegistry.get_or_create(config)
    return {
        "messages": [],
        "evidence": [],
        "citations": [],
        "tool_results": [],
        "task": "Run a tool",
        "run_config": config,
        "plan": None,
        "iteration": 0,
        "status": "running",
        "route_reason": None,
        "stop_reason": None,
        "needs_user_input": None,
        "pending_tool_calls": [call],
        "approved_tool_call_ids": [],
        "denied_tool_call_ids": [],
        "user_decision": None,
        "next_subtasks": None,
        "working_summary": None,
        "extracted_facts": [],
        "context_budget": None,
        "subtask_results": {},
        "terminal_subtasks": set(),
        "successful_subtasks": set(),
        "final_answer": None,
        "groundedness_flag": False,
        "insufficient_evidence_flag": False,
    }


@pytest.mark.anyio
async def test_execute_node_retries_retryable_runner_failure_before_success() -> None:
    attempts: list[str] = []

    def runner(payload: RuntimeInput) -> RuntimeOutput:
        attempts.append(payload.value)
        if len(attempts) == 1:
            raise RuntimeError("temporary failure")
        return RuntimeOutput(value=payload.value)

    registry = ToolRegistry()
    registry.register(_runtime_spec(max_retries=1), runner=runner)
    call = ToolCallPlan.create("runtime_tool", {"value": "ok"})

    update = await execute_node(
        _state(call=call, run_id="tool-retry"),
        tool_registry=registry,
        allowed_tools=frozenset({"runtime_tool"}),
    )

    [result] = update["tool_results"]
    assert result.status == "ok"
    assert result.output == RuntimeOutput(value="ok")
    assert result.retry_count == 1
    assert attempts == ["ok", "ok"]
    RuntimeRegistry.remove("tool-retry")


@pytest.mark.anyio
async def test_execute_node_times_out_async_runner() -> None:
    async def runner(payload: RuntimeInput) -> RuntimeOutput:
        await asyncio.sleep(0.05)
        return RuntimeOutput(value=payload.value)

    registry = ToolRegistry()
    registry.register(_runtime_spec(timeout_seconds=0.01), runner=runner)
    call = ToolCallPlan.create("runtime_tool", {"value": "late"})

    update = await execute_node(
        _state(call=call, run_id="tool-timeout"),
        tool_registry=registry,
        allowed_tools=frozenset({"runtime_tool"}),
    )

    [result] = update["tool_results"]
    assert result.status == "error"
    assert result.error is not None
    assert result.error.code == "timeout"
    assert result.error.retryable is True
    assert result.retry_count == 0
    RuntimeRegistry.remove("tool-timeout")

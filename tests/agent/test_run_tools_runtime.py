from __future__ import annotations

import asyncio

import pytest
from pydantic import BaseModel

from rag.agent.core.context import AgentRunConfig, RunRegistry
from rag.agent.core.definition import AgentDefinition
from rag.agent.core.llm_context import AgentLLMContextOverflowError
from rag.agent.graphs.nodes.execute import run_tools_raw
from rag.agent.memory.models import ContextBudgetSnapshot
from rag.agent.state import AgentState, ToolCallPlan
from rag.agent.tools.registry import ToolExecutionContext, ToolRegistry
from rag.agent.tools.spec import ToolError, ToolPermissions, ToolSpec
from rag.schema.llm import LLMCallStage
from rag.schema.runtime import AccessPolicy


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


def _state(
    *,
    call: ToolCallPlan,
    run_id: str,
    access_policy: AccessPolicy | None = None,
) -> AgentState:
    config = AgentRunConfig(
        run_id=run_id,
        thread_id=run_id,
        budget_total=100,
        max_depth=2,
        access_policy=access_policy or AccessPolicy.default(),
    )
    RunRegistry.remove(run_id)
    RunRegistry.get_or_create(config)
    return {
        "messages": [],
        "evidence": [],
        "citations": [],
        "tool_results": [],
        "task": "Run a tool",
        "run_config": config,
        "iteration": 0,
        "status": "running",
        "decision_reason": None,
        "stop_reason": None,
        "needs_user_input": None,
        "pending_tool_calls": [call],
        "approved_tool_call_ids": [],
        "denied_tool_call_ids": [],
        "user_decision": None,
        "working_summary": None,
        "extracted_facts": [],
        "context_budget": None,
        "final_answer": None,
        "groundedness_flag": False,
        "insufficient_evidence_flag": False,
    }


@pytest.mark.anyio
async def test_run_tools_raw_passes_trusted_run_config_to_contextual_runner() -> None:
    access_policy = AccessPolicy(allowed_runtimes=frozenset())
    seen_contexts: list[ToolExecutionContext] = []

    def runner(
        payload: RuntimeInput,
        context: ToolExecutionContext,
    ) -> RuntimeOutput:
        seen_contexts.append(context)
        return RuntimeOutput(value=payload.value)

    registry = ToolRegistry()
    registry.register(_runtime_spec())
    registry.register_contextual_runner("runtime_tool", runner)
    call = ToolCallPlan.create("runtime_tool", {"value": "ok"})

    state = _state(
        call=call,
        run_id="tool-execution-context",
        access_policy=access_policy,
    )
    definition = AgentDefinition(
        agent_type="runtime",
        description="runtime",
        system_prompt="trusted system",
        allowed_tools=["runtime_tool"],
    )
    update = await run_tools_raw(
        state,
        tool_registry=registry,
        allowed_tools=frozenset({"runtime_tool"}),
        definition=definition,
    )

    [result] = update["tool_results"]
    assert result.status == "ok"
    assert seen_contexts[0].run_config.access_policy is access_policy
    assert seen_contexts[0].state is state
    assert seen_contexts[0].definition is definition
    RunRegistry.remove("tool-execution-context")


@pytest.mark.anyio
async def test_run_tools_raw_retries_retryable_runner_failure_before_success() -> None:
    attempts: list[str] = []

    def runner(payload: RuntimeInput) -> RuntimeOutput:
        attempts.append(payload.value)
        if len(attempts) == 1:
            raise RuntimeError("temporary failure")
        return RuntimeOutput(value=payload.value)

    registry = ToolRegistry()
    registry.register(_runtime_spec(max_retries=1), runner=runner)
    call = ToolCallPlan.create("runtime_tool", {"value": "ok"})

    update = await run_tools_raw(
        _state(call=call, run_id="tool-retry"),
        tool_registry=registry,
        allowed_tools=frozenset({"runtime_tool"}),
    )

    [result] = update["tool_results"]
    assert result.status == "ok"
    assert result.output == RuntimeOutput(value="ok")
    assert result.retry_count == 1
    assert attempts == ["ok", "ok"]
    RunRegistry.remove("tool-retry")


@pytest.mark.anyio
async def test_context_overflow_pauses_without_retrying_tool() -> None:
    calls = 0

    def runner(
        payload: RuntimeInput,
        context: ToolExecutionContext,
    ) -> RuntimeOutput:
        nonlocal calls
        del payload, context
        calls += 1
        raise AgentLLMContextOverflowError(
            stage=LLMCallStage.LLM_GENERATE,
            context_budget=ContextBudgetSnapshot(
                max_context_tokens=10,
                overflow=True,
                degraded=True,
                required_truncated=["call_context"],
                warnings=["context_overflow"],
            ),
        )

    registry = ToolRegistry()
    registry.register(_runtime_spec(max_retries=3))
    registry.register_contextual_runner("runtime_tool", runner)
    call = ToolCallPlan.create("runtime_tool", {"value": "large"})
    definition = AgentDefinition(
        agent_type="runtime",
        description="runtime",
        system_prompt="system",
        allowed_tools=["runtime_tool"],
    )

    update = await run_tools_raw(
        _state(call=call, run_id="tool-context-overflow"),
        tool_registry=registry,
        allowed_tools=frozenset({"runtime_tool"}),
        definition=definition,
    )

    assert update["status"] == "paused"
    assert update["decision_reason"] == "context_overflow"
    assert isinstance(update["context_budget"], ContextBudgetSnapshot)
    [result] = update["tool_results"]
    assert result.error is not None
    assert result.error.code == "context_overflow"
    assert result.error.retryable is False
    assert result.retry_count == 0
    assert calls == 1
    RunRegistry.remove("tool-context-overflow")


@pytest.mark.anyio
async def test_run_tools_raw_times_out_async_runner() -> None:
    async def runner(payload: RuntimeInput) -> RuntimeOutput:
        await asyncio.sleep(0.05)
        return RuntimeOutput(value=payload.value)

    registry = ToolRegistry()
    registry.register(_runtime_spec(timeout_seconds=0.01), runner=runner)
    call = ToolCallPlan.create("runtime_tool", {"value": "late"})

    update = await run_tools_raw(
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
    RunRegistry.remove("tool-timeout")

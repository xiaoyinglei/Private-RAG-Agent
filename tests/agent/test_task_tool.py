"""Integration tests for the generic task tool.

Verifies end-to-end: task tool → child loop → tool execution → result.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from rag.agent.builtin.generic import GENERIC_AGENT
from rag.agent.core.context import AgentRunConfig
from rag.agent.core.delegation import AgentDelegationRequest, ParentAgentContext
from rag.agent.core.turn_contracts import ToolCallPlan
from rag.agent.service import AgentRunRequest, _TaskChildRunner
from rag.agent.tools.registry import ToolRegistry
from rag.agent.tools.task_tool import TaskInput, TaskOutput, TaskToolRunner, task_tool_spec
from rag.schema.runtime import AccessPolicy

# ── Helpers ──────────────────────────────────────────────────────


class _SimpleDecisionProvider:
    """Returns a finish decision with the task as the answer."""

    async def next_turn(self, state, *, definition, budget_remaining):
        from rag.agent.loop.state import ModelTurnDraft

        task = state.get("task", "")
        return ModelTurnDraft(action="finish", final_answer=f"answer: {task}")


class _DelegatedTaskResult:
    run_id = "child-run"
    status = "done"
    final_answer = "delegated answer"
    stop_reason = None
    tool_results = []
    evidence = []
    citations = []


class _CapturingDelegatedRunner:
    def __init__(self) -> None:
        self.calls: list[tuple[AgentDelegationRequest, ParentAgentContext]] = []

    def run_delegated_task(
        self,
        *,
        request: AgentDelegationRequest,
        parent_state: ParentAgentContext,
    ) -> _DelegatedTaskResult:
        self.calls.append((request, parent_state))
        return _DelegatedTaskResult()


def _make_run_config(**overrides) -> AgentRunConfig:
    defaults = dict(
        run_id="test-run",
        thread_id="test-thread",
        budget_total=8000,
        work_budget_total=20000,
        agent_type="generic",
        max_depth=2,
        access_policy=AccessPolicy.default(),
    )
    defaults.update(overrides)
    return AgentRunConfig(**defaults)


# ── Unit tests ───────────────────────────────────────────────────


class TestTaskToolSpec:
    def test_spec_is_correctly_defined(self) -> None:
        assert task_tool_spec.name == "task"
        assert task_tool_spec.input_model is TaskInput
        assert task_tool_spec.output_model is TaskOutput

    def test_task_input_validates(self) -> None:
        inp = TaskInput(task="Analyze Q3 revenue")
        assert inp.task == "Analyze Q3 revenue"
        assert inp.context_summary is None
        assert inp.tool_query is None
        assert inp.token_budget is None

    def test_task_input_with_all_fields(self) -> None:
        inp = TaskInput(
            task="Analyze Q3 revenue",
            context_summary="Previous quarter was 1.2M",
            tool_query="search financial data",
            token_budget=5000,
        )
        assert inp.context_summary == "Previous quarter was 1.2M"
        assert inp.token_budget == 5000


class TestTaskOutput:
    def test_from_run_result(self) -> None:
        result = MagicMock()
        result.final_answer = "Revenue is 1.5M"
        result.evidence = []
        result.citations = []
        result.status = "done"
        result.run_id = "child-run-123"
        result.stop_reason = None

        output = TaskOutput.from_run_result(result)
        assert output.conclusion == "Revenue is 1.5M"
        assert output.status == "done"
        assert output.child_run_id == "child-run-123"

    def test_error_result(self) -> None:
        output = TaskOutput.error_result("Something failed")
        assert output.conclusion == "Something failed"
        assert output.status == "failed"

    def test_from_run_result_paused(self) -> None:
        result = MagicMock()
        result.final_answer = None
        result.evidence = []
        result.citations = []
        result.status = "paused"
        result.run_id = "child-run-456"
        result.stop_reason = "needs_input"

        output = TaskOutput.from_run_result(result)
        assert output.status == "paused"
        assert output.stop_reason == "needs_input"


class TestTaskToolRunner:
    def test_child_policy_removes_task(self) -> None:
        policy = GENERIC_AGENT.to_runtime_policy()
        runner = _TaskChildRunner(
            policy=policy,
            tool_registry=ToolRegistry(),
            model_turn_provider=None,
            retrieval_hint_provider=None,
        )
        child = runner._child_policy()

        assert "task" not in child.core_tool_names
        assert "task" in policy.core_tool_names
        assert child.max_depth == policy.max_depth - 1

    def test_child_policy_at_zero_depth(self) -> None:
        from dataclasses import replace

        policy = replace(GENERIC_AGENT.to_runtime_policy(), max_depth=0)
        runner = _TaskChildRunner(
            policy=policy,
            tool_registry=ToolRegistry(),
            model_turn_provider=None,
            retrieval_hint_provider=None,
        )
        child = runner._child_policy()

        assert child.max_depth == 0

    @pytest.mark.anyio
    async def test_uses_injected_delegated_runner(self) -> None:
        policy = GENERIC_AGENT.to_runtime_policy()
        delegated_runner = _CapturingDelegatedRunner()
        parent_config = _make_run_config(max_depth=2)
        runner = TaskToolRunner(
            policy=policy,
            tool_registry=ToolRegistry(),
            delegated_runner=delegated_runner,
        )

        output = await runner.run(
            TaskInput(
                task="Summarize the data",
                context_summary="Q1: 1M",
                token_budget=1234,
            ),
            parent_config=parent_config,
        )

        assert output.status == "done"
        assert output.conclusion == "delegated answer"
        assert len(delegated_runner.calls) == 1
        request, parent_state = delegated_runner.calls[0]
        assert request.agent_type == "task_child"
        assert request.estimated_tokens == 1234
        assert "Summarize the data" in request.prompt
        assert "Q1: 1M" in request.prompt
        assert parent_state["run_config"] == parent_config


# ── Integration tests ────────────────────────────────────────────


@pytest.mark.anyio
async def test_task_tool_end_to_end() -> None:
    """Full round-trip: task tool → child loop → tool execution → result."""
    from rag.agent.builtin_registry import create_builtin_tool_registry
    from rag.agent.core.agent_service_factory import AgentServiceFactory

    # Create a service with the generic agent and all builtin tools
    tool_registry = create_builtin_tool_registry()
    decision_provider = _SimpleDecisionProvider()

    factory = AgentServiceFactory(
        tool_registry=tool_registry,
        model_registry=None,
        model_turn_provider=decision_provider,
    )
    service = factory.create(GENERIC_AGENT)

    # Create a task call
    call = ToolCallPlan.create(
        "task",
        {"task": "What is 2+2?"},
    )

    result = await service.run(
        AgentRunRequest(
            task="Solve a math problem",
            run_id="task-integration-test",
            thread_id="task-integration-test",
            pending_tool_calls=[call],
        ),
    )

    # The task tool should have executed
    assert len(result.tool_results) == 1
    tr = result.tool_results[0]
    assert tr.status == "ok"

    output = TaskOutput.model_validate(tr.output)
    assert output.status == "done"
    assert "answer:" in output.conclusion


@pytest.mark.anyio
async def test_task_tool_with_context_summary() -> None:
    """Task tool passes context_summary to child."""
    from rag.agent.builtin_registry import create_builtin_tool_registry
    from rag.agent.core.agent_service_factory import AgentServiceFactory

    tool_registry = create_builtin_tool_registry()
    decision_provider = _SimpleDecisionProvider()

    factory = AgentServiceFactory(
        tool_registry=tool_registry,
        model_registry=None,
        model_turn_provider=decision_provider,
    )
    service = factory.create(GENERIC_AGENT)

    call = ToolCallPlan.create(
        "task",
        {
            "task": "Summarize the data",
            "context_summary": "Q1: 1M, Q2: 1.2M, Q3: 1.5M",
        },
    )

    result = await service.run(
        AgentRunRequest(
            task="Analyze quarterly data",
            run_id="task-context-test",
            thread_id="task-context-test",
            pending_tool_calls=[call],
        ),
    )

    assert result.tool_results[0].status == "ok"
    output = TaskOutput.model_validate(result.tool_results[0].output)
    assert output.status == "done"
    # Child should have received the context
    assert "Q1: 1M" in output.conclusion or "Summarize" in output.conclusion

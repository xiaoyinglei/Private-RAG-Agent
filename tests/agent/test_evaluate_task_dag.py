from __future__ import annotations

import pytest

from rag.agent.core.context import AgentRunConfig, RuntimeRegistry
from rag.agent.core.definition import AgentDefinition
from rag.agent.core.task import SubTaskNode, SubTaskStatus, TaskDAG, TaskEdge
from rag.agent.graphs.nodes.evaluate import evaluate_node, route_after_evaluate
from rag.agent.state import AgentState
from rag.schema.runtime import AccessPolicy


def _definition() -> AgentDefinition:
    return AgentDefinition(
        agent_type="orchestrator",
        description="Orchestrator",
        system_prompt="Coordinate sub agents.",
        allowed_tools=[],
        estimated_token_budget=100,
    )


def _subtask(subtask_id: str, *, priority: int = 1, estimated_tokens: int | None = 40) -> SubTaskNode:
    return SubTaskNode(
        subtask_id=subtask_id,
        agent_type="research",
        prompt=f"Do {subtask_id}",
        priority=priority,
        estimated_tokens=estimated_tokens,
    )


def _state(
    *,
    run_id: str,
    plan: TaskDAG,
    budget_total: int = 100,
    terminal_subtasks: set[str] | None = None,
    successful_subtasks: set[str] | None = None,
) -> AgentState:
    config = AgentRunConfig(
        run_id=run_id,
        thread_id=run_id,
        budget_total=budget_total,
        max_depth=2,
        access_policy=AccessPolicy.default(),
    )
    RuntimeRegistry.remove(run_id)
    RuntimeRegistry.get_or_create(config)
    return {
        "messages": [],
        "evidence": [],
        "citations": [],
        "tool_results": [],
        "task": "Coordinate subtasks",
        "run_config": config,
        "plan": plan,
        "iteration": 0,
        "status": "running",
        "route_reason": None,
        "stop_reason": None,
        "needs_user_input": None,
        "pending_tool_calls": [],
        "approved_tool_call_ids": [],
        "denied_tool_call_ids": [],
        "user_decision": None,
        "next_subtasks": None,
        "working_summary": None,
        "extracted_facts": [],
        "context_budget": None,
        "subtask_results": {},
        "terminal_subtasks": terminal_subtasks or set(),
        "successful_subtasks": successful_subtasks or set(),
        "final_answer": None,
        "groundedness_flag": False,
        "insufficient_evidence_flag": False,
    }


@pytest.mark.anyio
async def test_evaluate_task_dag_schedules_ready_subtasks_and_reserves_budget() -> None:
    plan = TaskDAG(subtasks=[_subtask("s1", priority=5), _subtask("s2", priority=1)])
    result = await evaluate_node(
        _state(run_id="dag-ready", plan=plan),
        definition=_definition(),
    )

    assert result["status"] == "running"
    assert [subtask.subtask_id for subtask in result["next_subtasks"]] == ["s1", "s2"]
    remaining = await RuntimeRegistry.get("dag-ready").budget_ledger.remaining()
    assert remaining == 20
    RuntimeRegistry.remove("dag-ready")


@pytest.mark.anyio
async def test_evaluate_task_dag_marks_budget_exhausted_subtask_failed() -> None:
    plan = TaskDAG(
        subtasks=[
            _subtask("s1", priority=5, estimated_tokens=80),
            _subtask("s2", priority=1, estimated_tokens=80),
        ]
    )
    result = await evaluate_node(
        _state(run_id="dag-budget", plan=plan, budget_total=100),
        definition=_definition(),
    )

    assert [subtask.subtask_id for subtask in result["next_subtasks"]] == ["s1"]
    assert result["terminal_subtasks"] == {"s2"}
    failed = result["subtask_results"]["s2"]
    assert failed.status is SubTaskStatus.FAILED
    assert "Insufficient budget" in failed.error_message
    RuntimeRegistry.remove("dag-budget")


@pytest.mark.anyio
async def test_evaluate_task_dag_uses_subtask_default_budget_when_estimate_is_missing() -> None:
    plan = TaskDAG(subtasks=[_subtask("s1", estimated_tokens=None)])
    result = await evaluate_node(
        _state(run_id="dag-default-budget", plan=plan, budget_total=20000),
        definition=_definition(),
    )

    assert result["status"] == "running"
    assert [subtask.subtask_id for subtask in result["next_subtasks"]] == ["s1"]
    remaining = await RuntimeRegistry.get("dag-default-budget").budget_ledger.remaining()
    assert remaining == 10000
    RuntimeRegistry.remove("dag-default-budget")


@pytest.mark.anyio
async def test_evaluate_task_dag_fails_when_dependencies_cannot_be_satisfied() -> None:
    plan = TaskDAG(
        subtasks=[_subtask("s1"), _subtask("s2")],
        edges=[TaskEdge(from_id="s1", to_id="s2")],
    )

    result = await evaluate_node(
        _state(run_id="dag-deadlock", plan=plan, terminal_subtasks={"s1"}, successful_subtasks=set()),
        definition=_definition(),
    )

    assert result["status"] == "failed"
    assert result["stop_reason"] == "deadlock_in_task_dag"
    RuntimeRegistry.remove("dag-deadlock")


@pytest.mark.anyio
async def test_evaluate_task_dag_completes_when_all_subtasks_terminal() -> None:
    plan = TaskDAG(subtasks=[_subtask("s1"), _subtask("s2")])

    result = await evaluate_node(
        _state(
            run_id="dag-done",
            plan=plan,
            terminal_subtasks={"s1", "s2"},
            successful_subtasks={"s1"},
        ),
        definition=_definition(),
    )

    assert result["status"] == "done"
    assert result["stop_reason"] == "all_subtasks_terminal"
    RuntimeRegistry.remove("dag-done")


def test_route_after_evaluate_sends_ready_subtasks_to_execute_subagent() -> None:
    subtask = _subtask("s1")
    state = _state(
        run_id="dag-route",
        plan=TaskDAG(subtasks=[subtask]),
    )
    state["next_subtasks"] = [subtask]

    route = route_after_evaluate(state)

    assert isinstance(route, list)
    [send] = route
    assert send.node == "execute_subagent"
    assert send.arg["subtask"] == subtask
    assert send.arg["run_config"] == state["run_config"]
    RuntimeRegistry.remove("dag-route")

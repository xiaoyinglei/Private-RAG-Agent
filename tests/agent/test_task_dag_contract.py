from __future__ import annotations

import pytest

from rag.agent.core.task import (
    DEFAULT_SUBTASK_TOKEN_BUDGET,
    SubTaskNode,
    SubTaskResult,
    SubTaskStatus,
    TaskDAG,
    TaskEdge,
)


def _subtask(subtask_id: str, *, priority: int = 1) -> SubTaskNode:
    return SubTaskNode(
        subtask_id=subtask_id,
        agent_type="research",
        prompt=f"Do {subtask_id}",
        priority=priority,
    )


def test_task_dag_requires_unique_subtask_ids() -> None:
    with pytest.raises(ValueError, match="duplicate subtask_id"):
        TaskDAG(subtasks=[_subtask("s1"), _subtask("s1")])


def test_task_dag_rejects_edges_to_unknown_subtasks() -> None:
    with pytest.raises(ValueError, match="unknown subtask"):
        TaskDAG(
            subtasks=[_subtask("s1")],
            edges=[TaskEdge(from_id="s1", to_id="missing")],
        )


def test_task_dag_rejects_cycles() -> None:
    with pytest.raises(ValueError, match="cycle"):
        TaskDAG(
            subtasks=[_subtask("s1"), _subtask("s2")],
            edges=[
                TaskEdge(from_id="s1", to_id="s2"),
                TaskEdge(from_id="s2", to_id="s1"),
            ],
        )


def test_ready_subtasks_only_use_successful_dependencies() -> None:
    dag = TaskDAG(
        subtasks=[_subtask("s1"), _subtask("s2"), _subtask("s3")],
        edges=[
            TaskEdge(from_id="s1", to_id="s3"),
            TaskEdge(from_id="s2", to_id="s3"),
        ],
    )

    assert [task.subtask_id for task in dag.ready_subtasks(successful=set(), terminal=set())] == [
        "s1",
        "s2",
    ]
    assert dag.ready_subtasks(successful={"s1"}, terminal={"s1", "s2"}) == []
    assert [
        task.subtask_id
        for task in dag.ready_subtasks(successful={"s1", "s2"}, terminal={"s1", "s2"})
    ] == ["s3"]


def test_ready_subtasks_are_sorted_by_priority_then_id() -> None:
    dag = TaskDAG(subtasks=[_subtask("low", priority=1), _subtask("high", priority=5)])

    assert [task.subtask_id for task in dag.ready_subtasks(successful=set(), terminal=set())] == [
        "high",
        "low",
    ]


def test_subtask_default_token_budget_is_10000() -> None:
    assert _subtask("s1").estimated_tokens == DEFAULT_SUBTASK_TOKEN_BUDGET


def test_failed_subtask_result_requires_error_message() -> None:
    with pytest.raises(ValueError, match="error_message is required"):
        SubTaskResult(subtask=_subtask("s1"), status=SubTaskStatus.FAILED)


def test_completed_subtask_result_rejects_error_message() -> None:
    with pytest.raises(ValueError, match="error_message must be None"):
        SubTaskResult(
            subtask=_subtask("s1"),
            status=SubTaskStatus.COMPLETED,
            error_message="should not be set",
        )

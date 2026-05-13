from __future__ import annotations

import pytest
from pydantic import BaseModel

from rag.agent.core.context import AgentRunConfig, RuntimeRegistry
from rag.agent.core.task import SubTaskNode, SubTaskStatus
from rag.agent.graphs.nodes.execute_subagent import execute_subagent_node
from rag.agent.service import AgentRunResult
from rag.agent.state import AgentState
from rag.agent.tools.spec import ToolError, ToolResult
from rag.schema.query import AnswerCitation, EvidenceItem
from rag.schema.runtime import AccessPolicy


class SummaryOutput(BaseModel):
    text: str


def _subtask() -> SubTaskNode:
    return SubTaskNode(
        subtask_id="s1",
        agent_type="research",
        prompt="Research policy",
        priority=1,
        estimated_tokens=40,
    )


def _state(run_id: str, subtask: SubTaskNode) -> AgentState:
    config = AgentRunConfig(
        run_id=run_id,
        thread_id=run_id,
        budget_total=100,
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
        "plan": None,
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
        "terminal_subtasks": set(),
        "successful_subtasks": set(),
        "final_answer": None,
        "groundedness_flag": False,
        "insufficient_evidence_flag": False,
        "subtask": subtask,
    }


class _SuccessfulRunner:
    async def run_subtask(self, *, subtask: SubTaskNode, parent_state: AgentState) -> AgentRunResult:
        del parent_state
        evidence = EvidenceItem(
            evidence_id="ev1",
            doc_id=1,
            citation_anchor="doc#1",
            text="Authoritative child evidence",
            score=0.9,
        )
        citation = AnswerCitation(
            citation_id="cit1",
            evidence_id="ev1",
            record_type="section",
            citation_anchor="doc#1",
        )
        return AgentRunResult(
            run_id=f"child-{subtask.subtask_id}",
            thread_id=f"child-{subtask.subtask_id}",
            status="done",
            final_answer="Child finding",
            tool_results=[
                ToolResult(
                    tool_call_id="tc1",
                    tool_name="llm_summarize",
                    status="ok",
                    output=SummaryOutput(text="Child finding"),
                    latency_ms=1,
                    token_used=12,
                )
            ],
            evidence=[evidence],
            citations=[citation],
        )


class _FailingRunner:
    async def run_subtask(self, *, subtask: SubTaskNode, parent_state: AgentState) -> AgentRunResult:
        del subtask, parent_state
        raise RuntimeError("child failed")


class _FailedStatusRunner:
    async def run_subtask(self, *, subtask: SubTaskNode, parent_state: AgentState) -> AgentRunResult:
        del parent_state
        return AgentRunResult(
            run_id=f"child-{subtask.subtask_id}",
            thread_id=f"child-{subtask.subtask_id}",
            status="failed",
            stop_reason="budget_exhausted",
            tool_results=[
                ToolResult(
                    tool_call_id="tc-failed",
                    tool_name="llm_summarize",
                    status="error",
                    error=ToolError(
                        code="budget_exhausted",
                        message="child budget exhausted",
                        retryable=False,
                    ),
                    latency_ms=1,
                    token_used=7,
                )
            ],
        )


class _RecordingRunner(_SuccessfulRunner):
    def __init__(self) -> None:
        self.calls = 0

    async def run_subtask(self, *, subtask: SubTaskNode, parent_state: AgentState) -> AgentRunResult:
        self.calls += 1
        return await super().run_subtask(subtask=subtask, parent_state=parent_state)


@pytest.mark.anyio
async def test_execute_subagent_success_commits_budget_and_marks_successful() -> None:
    subtask = _subtask()
    state = _state("subagent-success", subtask)
    handles = RuntimeRegistry.get("subagent-success")
    assert await handles.budget_ledger.reserve(subtask.subtask_id, subtask.estimated_tokens or 0)

    update = await execute_subagent_node(state, subagent_runner=_SuccessfulRunner())

    result = update["subtask_results"][subtask.subtask_id]
    assert result.status is SubTaskStatus.COMPLETED
    assert result.findings == ["Child finding"]
    assert result.evidence[0].evidence_id == "ev1"
    assert update["terminal_subtasks"] == {"s1"}
    assert update["successful_subtasks"] == {"s1"}
    assert update["evidence"][0].evidence_id == "ev1"
    assert update["citations"][0].citation_id == "cit1"
    assert await handles.budget_ledger.remaining() == 88
    RuntimeRegistry.remove("subagent-success")


@pytest.mark.anyio
async def test_execute_subagent_failure_refunds_budget_and_does_not_mark_successful() -> None:
    subtask = _subtask()
    state = _state("subagent-failure", subtask)
    handles = RuntimeRegistry.get("subagent-failure")
    assert await handles.budget_ledger.reserve(subtask.subtask_id, subtask.estimated_tokens or 0)

    update = await execute_subagent_node(state, subagent_runner=_FailingRunner())

    result = update["subtask_results"][subtask.subtask_id]
    assert result.status is SubTaskStatus.FAILED
    assert result.error_message == "child failed"
    assert update["terminal_subtasks"] == {"s1"}
    assert "successful_subtasks" not in update
    assert await handles.budget_ledger.remaining() == 100
    RuntimeRegistry.remove("subagent-failure")


@pytest.mark.anyio
async def test_execute_subagent_failed_result_marks_subtask_failed_without_refund() -> None:
    subtask = _subtask()
    state = _state("subagent-result-failed", subtask)
    handles = RuntimeRegistry.get("subagent-result-failed")
    assert await handles.budget_ledger.reserve(subtask.subtask_id, subtask.estimated_tokens or 0)

    update = await execute_subagent_node(state, subagent_runner=_FailedStatusRunner())

    result = update["subtask_results"][subtask.subtask_id]
    assert result.status is SubTaskStatus.FAILED
    assert result.error_message == "Subagent failed: budget_exhausted"
    assert update["terminal_subtasks"] == {"s1"}
    assert "successful_subtasks" not in update
    assert await handles.budget_ledger.remaining() == 93
    RuntimeRegistry.remove("subagent-result-failed")


@pytest.mark.anyio
async def test_execute_subagent_missing_runtime_handles_fails_before_running_child() -> None:
    subtask = _subtask()
    state = _state("subagent-runtime-missing", subtask)
    RuntimeRegistry.remove("subagent-runtime-missing")
    runner = _RecordingRunner()

    update = await execute_subagent_node(state, subagent_runner=runner)

    assert update["status"] == "failed"
    assert update["stop_reason"] == "runtime_handles_missing"
    assert runner.calls == 0

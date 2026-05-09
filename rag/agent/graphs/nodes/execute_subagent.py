from __future__ import annotations

from collections.abc import Awaitable
from inspect import isawaitable
from typing import Protocol

from rag.agent.core.context import RuntimeRegistry
from rag.agent.core.task import SubTaskNode, SubTaskResult, SubTaskStatus
from rag.agent.state import AgentState
from rag.agent.tools.spec import ToolResult
from rag.schema.query import AnswerCitation, EvidenceItem


class SubAgentRunResult(Protocol):
    status: str
    final_answer: str | None
    stop_reason: str | None
    tool_results: list[ToolResult]
    evidence: list[EvidenceItem]
    citations: list[AnswerCitation]


class SubAgentRunner(Protocol):
    def run_subtask(
        self,
        *,
        subtask: SubTaskNode,
        parent_state: AgentState,
    ) -> SubAgentRunResult | Awaitable[SubAgentRunResult]: ...


async def execute_subagent_node(
    state: AgentState,
    *,
    subagent_runner: SubAgentRunner,
) -> dict:
    subtask = state.get("subtask")
    if not isinstance(subtask, SubTaskNode):
        return {
            "status": "failed",
            "stop_reason": "subtask_missing",
        }

    run_config = state["run_config"]
    try:
        raw_result = subagent_runner.run_subtask(subtask=subtask, parent_state=state)
        result = await raw_result if isawaitable(raw_result) else raw_result
    except Exception as exc:
        await RuntimeRegistry.get(run_config.run_id).budget_ledger.refund(subtask.subtask_id)
        return {
            "subtask_results": {
                subtask.subtask_id: SubTaskResult(
                    subtask=subtask,
                    status=SubTaskStatus.FAILED,
                    error_message=str(exc),
                )
            },
            "terminal_subtasks": {subtask.subtask_id},
        }

    actual_tokens = sum(tool_result.token_used for tool_result in result.tool_results)
    await RuntimeRegistry.get(run_config.run_id).budget_ledger.commit(
        subtask.subtask_id,
        actual_tokens,
    )
    if result.status != "done":
        return {
            "subtask_results": {
                subtask.subtask_id: SubTaskResult(
                    subtask=subtask,
                    status=SubTaskStatus.FAILED,
                    error_message=_subagent_failure_message(result),
                )
            },
            "terminal_subtasks": {subtask.subtask_id},
        }

    subtask_result = SubTaskResult(
        subtask=subtask,
        status=SubTaskStatus.COMPLETED,
        findings=[result.final_answer] if result.final_answer else [],
        evidence=result.evidence,
        citations=result.citations,
    )
    return {
        "evidence": result.evidence,
        "citations": result.citations,
        "subtask_results": {subtask.subtask_id: subtask_result},
        "terminal_subtasks": {subtask.subtask_id},
        "successful_subtasks": {subtask.subtask_id},
    }


def _subagent_failure_message(result: SubAgentRunResult) -> str:
    reason = result.stop_reason or result.status
    return f"Subagent failed: {reason}"

from __future__ import annotations

from collections.abc import Awaitable
from inspect import isawaitable
from typing import Any, Protocol

from pydantic import BaseModel

from rag.agent.core.task import SubTaskResult, SubTaskStatus
from rag.agent.state import AgentState
from rag.agent.tools.spec import ToolResult
from rag.schema.query import AnswerCitation, EvidenceItem


class SynthesisRunResult(Protocol):
    status: str
    final_answer: str | None
    stop_reason: str | None
    tool_results: list[ToolResult]
    evidence: list[EvidenceItem]
    citations: list[AnswerCitation]
    groundedness_flag: bool
    insufficient_evidence_flag: bool


class SynthesisRunner(Protocol):
    def run_synthesis(
        self,
        *,
        parent_state: AgentState,
    ) -> SynthesisRunResult | Awaitable[SynthesisRunResult]: ...


async def synthesize_node(
    state: AgentState,
    *,
    synthesis_runner: SynthesisRunner | None = None,
) -> dict[str, Any]:
    tool_results = state.get("tool_results", [])
    evidence = state.get("evidence", [])

    # fast_path 无检索结果 → 不可空 synthesize
    if state.get("execution_mode") == "fast_path":
        has_content = bool(evidence) or any(tr.status == "ok" for tr in tool_results)
        if not has_content:
            return {
                "status": "failed",
                "stop_reason": "insufficient_evidence",
                "final_answer": "当前无检索证据，无法生成回答。请重试或提供更多信息。",
                "insufficient_evidence_flag": True,
            }

    if synthesis_runner is not None and _should_delegate_to_synthesis_agent(state):
        try:
            raw_result = synthesis_runner.run_synthesis(parent_state=state)
            result = await raw_result if isawaitable(raw_result) else raw_result
        except Exception as exc:
            fallback = _legacy_synthesize_node(state)
            return {
                **fallback,
                "stop_reason": f"synthesis_agent_failed: {exc}",
                "insufficient_evidence_flag": True,
            }
        return _synthesis_agent_update(state, result)

    return _legacy_synthesize_node(state)


def _legacy_synthesize_node(state: AgentState) -> dict[str, Any]:
    tool_results = state.get("tool_results", [])
    ok_results = [result for result in tool_results if result.status == "ok"]
    error_results = [result for result in tool_results if result.status == "error"]
    subtask_results = list(state.get("subtask_results", {}).values())
    status = state.get("status")
    final_status = "failed" if status == "failed" else "done"
    return {
        "status": final_status,
        "final_answer": _final_answer(
            ok_results,
            error_results,
            subtask_results,
            status=status,
            stop_reason=state.get("stop_reason"),
        ),
        "groundedness_flag": bool(ok_results) or _has_subtask_evidence(subtask_results),
        "insufficient_evidence_flag": (
            state.get("insufficient_evidence_flag", False)
            or bool(error_results)
            or _has_failed_subtask(subtask_results)
            or _has_insufficient_output(ok_results)
        ),
    }


def _should_delegate_to_synthesis_agent(state: AgentState) -> bool:
    if state.get("status") == "failed":
        return False
    tool_results = state.get("tool_results", [])
    if _has_grounded_answer_tool_result(tool_results):
        return False
    subtask_results = list(state.get("subtask_results", {}).values())
    return (
        bool(state.get("evidence"))
        or any(result.status == "ok" for result in tool_results)
        or any(
            result.status is SubTaskStatus.COMPLETED and result.findings
            for result in subtask_results
        )
    )


def _has_grounded_answer_tool_result(tool_results: list[ToolResult]) -> bool:
    for result in tool_results:
        if result.status != "ok" or result.tool_name != "rag_search_answer" or result.output is None:
            continue
        text = getattr(result.output, "text", None)
        if not isinstance(text, str) or not text.strip():
            continue
        if bool(getattr(result.output, "insufficient_evidence", False)):
            continue
        return True
    return False


def _synthesis_agent_update(state: AgentState, result: SynthesisRunResult) -> dict[str, Any]:
    fallback = _legacy_synthesize_node(state)
    if result.status != "done" or not result.final_answer:
        return {
            **fallback,
            "stop_reason": result.stop_reason or result.status,
            "insufficient_evidence_flag": True,
        }

    return {
        "status": "done",
        "final_answer": result.final_answer,
        "tool_results": result.tool_results,
        "evidence": result.evidence,
        "citations": result.citations,
        "groundedness_flag": (
            fallback["groundedness_flag"]
            or result.groundedness_flag
            or bool(result.evidence)
            or bool(result.citations)
        ),
        "insufficient_evidence_flag": (
            fallback["insufficient_evidence_flag"]
            or result.insufficient_evidence_flag
        ),
    }


def _final_answer(
    ok_results: list[ToolResult],
    error_results: list[ToolResult],
    subtask_results: list[SubTaskResult],
    *,
    status: str | None,
    stop_reason: str | None,
) -> str:
    answer_parts = [
        text
        for result in ok_results
        if result.output is not None and (text := _output_text(result.output))
    ]
    if answer_parts:
        return "\n\n".join(answer_parts)
    subtask_findings = [
        finding
        for result in subtask_results
        if result.status is SubTaskStatus.COMPLETED
        for finding in result.findings
        if finding.strip()
    ]
    if subtask_findings:
        return "\n\n".join(subtask_findings)
    if error_results:
        error_codes = ", ".join(
            result.error.code for result in error_results if result.error is not None
        )
        if error_codes:
            return f"No answer was generated because tool execution failed: {error_codes}."
        return "No answer was generated because tool execution failed."
    failed_subtasks = [
        subtask_result.subtask.subtask_id
        for subtask_result in subtask_results
        if subtask_result.status is SubTaskStatus.FAILED
    ]
    if failed_subtasks:
        return f"No answer was generated because subtask execution failed: {', '.join(failed_subtasks)}."
    if status == "failed" and stop_reason:
        return f"Agent failed: {stop_reason}."
    return "No answer was generated because no tool results were available."


def _output_text(output: BaseModel) -> str:
    text = getattr(output, "text", None)
    if isinstance(text, str) and text.strip():
        return text.strip()
    return output.model_dump_json(exclude_none=True)


def _has_insufficient_output(ok_results: list[ToolResult]) -> bool:
    return any(
        bool(getattr(result.output, "insufficient_evidence", False))
        for result in ok_results
        if result.output is not None
    )


def _has_subtask_evidence(subtask_results: list[SubTaskResult]) -> bool:
    return any(result.evidence for result in subtask_results)


def _has_failed_subtask(subtask_results: list[SubTaskResult]) -> bool:
    return any(result.status is SubTaskStatus.FAILED for result in subtask_results)

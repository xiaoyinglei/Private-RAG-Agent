from __future__ import annotations

from pydantic import BaseModel

from rag.agent.core.task import SubTaskResult, SubTaskStatus
from rag.agent.state import AgentState
from rag.agent.tools.spec import ToolResult


def synthesize_node(state: AgentState) -> dict:
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

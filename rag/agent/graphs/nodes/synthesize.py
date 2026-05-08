from __future__ import annotations

from pydantic import BaseModel

from rag.agent.state import AgentState
from rag.agent.tools.spec import ToolResult


def synthesize_node(state: AgentState) -> dict:
    tool_results = state.get("tool_results", [])
    ok_results = [result for result in tool_results if result.status == "ok"]
    error_results = [result for result in tool_results if result.status == "error"]
    status = state.get("status")
    final_status = "failed" if status == "failed" else "done"
    return {
        "status": final_status,
        "final_answer": _final_answer(ok_results, error_results),
        "groundedness_flag": bool(ok_results),
        "insufficient_evidence_flag": (
            state.get("insufficient_evidence_flag", False)
            or bool(error_results)
            or _has_insufficient_output(ok_results)
        ),
    }


def _final_answer(ok_results: list[ToolResult], error_results: list[ToolResult]) -> str:
    answer_parts = [
        text
        for result in ok_results
        if result.output is not None and (text := _output_text(result.output))
    ]
    if answer_parts:
        return "\n\n".join(answer_parts)
    if error_results:
        error_codes = ", ".join(
            result.error.code for result in error_results if result.error is not None
        )
        if error_codes:
            return f"No answer was generated because tool execution failed: {error_codes}."
        return "No answer was generated because tool execution failed."
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

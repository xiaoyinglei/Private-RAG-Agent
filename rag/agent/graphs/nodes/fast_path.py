from __future__ import annotations

from rag.agent.graphs.nodes.execute import execute_node
from rag.agent.state import AgentState, ToolCallPlan
from rag.agent.tools.fast_path_tools import RAGSearchAnswerOutput
from rag.agent.tools.registry import ToolRegistry
from rag.agent.tools.spec import ToolResult


async def fast_path_node(
    state: AgentState,
    *,
    tool_registry: ToolRegistry,
    allowed_tools: frozenset[str],
) -> dict:
    signals = state.get("retrieval_signals")
    call = ToolCallPlan.create(
        "rag_search_answer",
        {
            "query": state["task"],
            "top_k": 8,
            "retrieval_signals": (
                signals.model_dump(mode="json") if signals is not None else {}
            ),
        },
    )
    execution_state = dict(state)
    execution_state["pending_tool_calls"] = [call]
    update = await execute_node(
        execution_state,  # type: ignore[arg-type]
        tool_registry=tool_registry,
        allowed_tools=allowed_tools,
    )
    tool_results = list(update.get("tool_results", []))
    if not tool_results:
        return {
            "status": "failed",
            "stop_reason": "fast_path_no_tool_result",
            "final_answer": "No answer was generated because fast path returned no tool result.",
            "insufficient_evidence_flag": True,
        }
    result = tool_results[0]
    if result.status == "error" or result.output is None:
        return _failed_fast_path_update(result)

    output = RAGSearchAnswerOutput.model_validate(result.output)
    if not output.text.strip():
        return {
            "status": "failed",
            "stop_reason": "insufficient_evidence",
            "final_answer": "当前无检索证据，无法生成回答。请重试或提供更多信息。",
            "tool_results": tool_results,
            "insufficient_evidence_flag": True,
        }

    return {
        "status": "done",
        "final_answer": output.text,
        "tool_results": tool_results,
        "evidence": output.evidence,
        "citations": output.citations,
        "groundedness_flag": (
            output.groundedness_flag or bool(output.evidence) or bool(output.citations)
        ),
        "insufficient_evidence_flag": output.insufficient_evidence,
        "pending_tool_calls": [],
    }


def _failed_fast_path_update(result: ToolResult) -> dict:
    error_code = result.error.code if result.error is not None else "unknown"
    return {
        "status": "failed",
        "stop_reason": error_code,
        "final_answer": f"No answer was generated because fast path failed: {error_code}.",
        "tool_results": [result],
        "insufficient_evidence_flag": True,
    }

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from typing import Any

from langchain_core.messages import BaseMessage
from pydantic import BaseModel

from rag.agent.tools.spec import ToolResult


def _derive_groundedness(tool_results: list[ToolResult]) -> bool:
    """Derive groundedness_flag from the last RAG generation ToolResult.output."""
    for result in reversed(tool_results):
        if result.status == "ok" and result.output is not None:
            if bool(getattr(result.output, "groundedness_flag", False)):
                return True
    return False


def _derive_insufficient_evidence(tool_results: list[ToolResult]) -> bool:
    """Derive insufficient_evidence_flag from the last RAG generation ToolResult.output."""
    for result in reversed(tool_results):
        if result.status == "ok" and result.output is not None:
            if bool(getattr(result.output, "insufficient_evidence", False)) or bool(
                getattr(result.output, "insufficient_evidence_flag", False)
            ):
                return True
    return False


def normalize_loop_state(
    state: Mapping[str, Any],
    *,
    observed: dict[str, object] | None = None,
) -> dict[str, object]:
    """Return the loop state in the same behavioral comparison shape."""

    terminal = state.get("terminal")
    pause = state.get("pause")
    normalized: dict[str, object] = {
        "status": state.get("status"),
        "stop_reason": getattr(terminal, "stop_reason", None),
        "decision_reason": (None if state.get("latest_transition") is None else state["latest_transition"].reason),
        "final_answer": state.get("final_answer"),
        "final_output": _normalize_value(state.get("final_output")),
        "output_validation_errors": _normalize_value(state.get("output_validation_errors", [])),
        "iteration": state.get("iteration", 0),
        "groundedness_flag": _derive_groundedness(list(state.get("tool_results", []))),
        "insufficient_evidence_flag": _derive_insufficient_evidence(list(state.get("tool_results", []))),
        "needs_user_input": getattr(pause, "reason", None),
        "human_input_request": _normalize_value(state.get("approval_request")),
        "pending_tool_calls": [
            {
                "tool_call_id": call.tool_call_id,
                "tool_name": call.tool_name,
                "arguments": _normalize_value(call.plan.arguments),
            }
            for call in state.get("pending_tool_calls", [])
        ],
        "tool_results": [
            {
                "tool_call_id": result.tool_call_id,
                "tool_name": result.tool_name,
                "status": result.status,
                "output": _normalize_value(result.output),
                "error": _normalize_value(result.error),
                "work_units_used": result.work_units_used,
                "retry_count": result.retry_count,
            }
            for result in state.get("tool_results", [])
        ],
        "evidence": _normalize_value(state.get("evidence", [])),
        "citations": _normalize_value(state.get("citations", [])),
        "retrieval_signals": _normalize_value(state.get("retrieval_signals")),
        "retrieval_signals_debug": _normalize_value(state.get("retrieval_signals_debug")),
        "structured_observations": _normalize_value(state.get("structured_observations", [])),
        "answer_candidates": _normalize_value(state.get("answer_candidates", [])),
        "evidence_refs": _normalize_value(state.get("evidence_refs", [])),
        "computation_results": _normalize_value(state.get("computation_results", [])),
        "context_units": _normalize_value(state.get("context_units", [])),
        "locators": _normalize_value(state.get("locators", [])),
        "satisfied_requirements": [],
        "open_gap_ids": [],
        "messages": [_normalize_message(message) for message in state.get("messages", [])],
        "working_summary": _normalize_value(state.get("working_summary")),
        "memory_refs": _normalize_value(state.get("memory_refs", [])),
        "memory_budget": _normalize_value(state.get("memory_budget")),
        "memory_warnings": list(state.get("memory_warnings", [])),
        "runtime_diagnostics": _normalize_value(state.get("runtime_diagnostics", [])),
        "stop_hook_feedback": _normalize_value(state.get("stop_hook_feedback", [])),
    }
    if observed is not None:
        normalized["observed"] = _normalize_value(observed)
    return normalized


def _normalize_message(message: object) -> dict[str, object]:
    if not isinstance(message, BaseMessage):
        return {"value": _normalize_value(message)}
    return {
        "type": message.type,
        "id": message.id,
        "content": _normalize_value(message.content),
    }


def _normalize_value(value: object, *, key: str | None = None) -> object:
    if isinstance(value, BaseModel):
        return _normalize_value(value.model_dump(mode="json"), key=key)
    if is_dataclass(value) and not isinstance(value, type):
        return _normalize_value(asdict(value), key=key)
    if isinstance(value, dict):
        return {
            str(item_key): _normalize_value(item_value, key=str(item_key))
            for item_key, item_value in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_normalize_value(item) for item in value]
    if key == "request_id":
        return "<request_id>"
    if key in {"run_id", "thread_id"}:
        return f"<{key}>"
    if key == "delegation_id":
        return "<delegation_id>"
    if key == "ref_id":
        return "<memory_ref_id>"
    if key == "updated_at":
        return "<timestamp>"
    if key == "path" and isinstance(value, str) and ".agent_memory" in value:
        return "<memory_path>"
    if isinstance(value, float):
        return round(value, 6)
    return value

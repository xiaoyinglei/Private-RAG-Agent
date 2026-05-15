from __future__ import annotations

from typing import Any

from rag.agent.state import AgentState


def observe_node(state: AgentState) -> dict[str, Any]:
    results = state.get("tool_results", [])
    if not results:
        return {}
    return {"insufficient_evidence_flag": any(result.status == "error" for result in results)}

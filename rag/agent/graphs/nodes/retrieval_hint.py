from __future__ import annotations

from collections.abc import Awaitable
from typing import Any, Protocol

from rag.agent.state import AgentState


class RetrievalHintProvider(Protocol):
    """Retrieval metadata injection point. It cannot select an execution branch."""

    def hint(
        self,
        state: AgentState,
    ) -> dict[str, Any] | Awaitable[dict[str, Any]]: ...


def retrieval_hint_node(state: AgentState) -> dict[str, Any]:
    """Provide retrieval hint metadata without selecting an execution branch."""
    if state.get("pending_tool_calls"):
        return {"decision_reason": "pending_tool_calls"}

    retrieval_signals = state.get("retrieval_signals")
    if retrieval_signals is not None and retrieval_signals.allow_graph_expansion:
        return {"decision_reason": "graph_expansion_requested"}

    return {"decision_reason": "agent_research"}

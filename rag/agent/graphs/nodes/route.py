from __future__ import annotations

from collections.abc import Awaitable
from typing import Any, Protocol

from rag.agent.state import AgentState


class RouteProvider(Protocol):
    """route 节点的 LLM 决策注入点。返回 state update dict。"""

    def route(
        self,
        state: AgentState,
    ) -> dict[str, Any] | Awaitable[dict[str, Any]]: ...


def route_node(state: AgentState) -> dict[str, Any]:
    """Agent route 节点。基于 AgentState 中的 task 和 retrieval_signals 做路由。

    不调用 RAG Core 的 QueryUnderstandingService。
    路由标准按执行需求定义，不按固定任务枚举。
    """
    if state.get("pending_tool_calls"):
        return {"status": "direct", "execution_mode": "direct", "route_reason": "pending_tool_calls"}

    retrieval_signals = state.get("retrieval_signals")
    if retrieval_signals is not None and retrieval_signals.allow_graph_expansion:
        return {
            "status": "decompose", "execution_mode": "decompose",
            "route_reason": "graph_expansion_requested",
        }

    return {"status": "direct", "execution_mode": "direct", "route_reason": "agent_research"}


def route_after_route(state: AgentState) -> str:
    if state.get("status") == "fast_path":
        return "fast_path"
    if state.get("status") == "direct":
        return "execute"
    if state.get("status") == "decompose":
        if state.get("decompose_disabled_single_agent_mode"):
            return "execute"
        return "plan"
    if state.get("status") == "failed":
        return "synthesize"
    return "execute"

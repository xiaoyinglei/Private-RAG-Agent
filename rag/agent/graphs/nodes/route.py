from __future__ import annotations

from collections.abc import Awaitable
from typing import Protocol

from rag.agent.state import AgentState


class RouteProvider(Protocol):
    """route 节点的 LLM 决策注入点。返回 state update dict。"""

    def route(
        self,
        state: AgentState,
    ) -> dict | Awaitable[dict]: ...


def route_node(state: AgentState) -> dict:
    """Agent route 节点。基于 AgentState 中的 task 和 retrieval_signals 做路由。

    不调用 RAG Core 的 QueryUnderstandingService。
    路由标准按执行需求定义，不按固定任务枚举。
    """
    if state.get("pending_tool_calls"):
        return {"status": "direct", "route_reason": "pending_tool_calls"}

    retrieval_signals = state.get("retrieval_signals")
    if retrieval_signals is not None and retrieval_signals.allow_graph_expansion:
        return {"status": "decompose", "route_reason": "graph_expansion_requested"}

    return {"status": "direct", "route_reason": "agent_research"}


def route_after_route(state: AgentState) -> str:
    # fast_path 暂走 execute 路径（fast_path_node 未实现，直接 synthesize 无数据）
    if state.get("status") == "fast_path":
        return "execute"
    if state.get("status") == "failed":
        return "synthesize"
    if state.get("status") == "decompose":
        return "plan"
    return "execute"

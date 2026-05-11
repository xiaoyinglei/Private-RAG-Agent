from __future__ import annotations

from rag.agent.state import AgentState


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
    if state.get("status") == "fast_path":
        return "synthesize"
    if state.get("status") == "failed":
        return "synthesize"
    if state.get("status") == "decompose":
        return "plan"
    return "execute"

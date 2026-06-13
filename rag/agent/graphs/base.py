from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any, NotRequired

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from rag.agent.service import AgentRunRequest, AgentRunResult


class AgentKernelGraphState(TypedDict):
    """Small outer-orchestration state around one complete kernel run."""

    request: AgentRunRequest
    run_id: str
    result: NotRequired[AgentRunResult]


def build_outer_agent_graph(
    *,
    run_kernel: Callable[
        [AgentRunRequest],
        Awaitable[AgentRunResult],
    ],
    checkpointer: BaseCheckpointSaver[str] | MemorySaver | None = None,
) -> Any:
    graph = StateGraph(AgentKernelGraphState)

    async def invoke_agent_loop(
        state: AgentKernelGraphState,
    ) -> dict[str, object]:
        result = await run_kernel(state["request"])
        return {
            "run_id": result.run_id,
            "result": result,
        }

    graph.add_node("agent_loop", invoke_agent_loop)
    graph.add_edge(START, "agent_loop")
    graph.add_edge("agent_loop", END)
    return graph.compile(checkpointer=checkpointer)


__all__ = ["AgentKernelGraphState", "build_outer_agent_graph"]

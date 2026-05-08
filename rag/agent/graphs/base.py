from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from rag.agent.core.definition import AgentDefinition
from rag.agent.graphs.nodes.evaluate import (
    EvaluateDecisionProvider,
    evaluate_node,
    route_after_evaluate,
)
from rag.agent.graphs.nodes.execute import execute_node
from rag.agent.graphs.nodes.observe import observe_node
from rag.agent.graphs.nodes.pause import pause_node
from rag.agent.graphs.nodes.route import route_after_route, route_node
from rag.agent.graphs.nodes.synthesize import synthesize_node
from rag.agent.state import AgentState
from rag.agent.tools.registry import ToolRegistry
from rag.retrieval.analysis import QueryUnderstandingService


def build_agent_graph(
    *,
    definition: AgentDefinition,
    tool_registry: ToolRegistry,
    query_understanding_service: object | None = None,
    evaluate_decision_provider: EvaluateDecisionProvider | None = None,
):
    graph = StateGraph(AgentState)
    understanding_service = query_understanding_service or QueryUnderstandingService(enable_llm=False)
    allowed_tools = frozenset(definition.allowed_tools)

    def bound_route_node(state: AgentState) -> dict:
        return route_node(state, query_understanding_service=understanding_service)

    async def bound_execute_node(state: AgentState) -> dict:
        return await execute_node(state, tool_registry=tool_registry, allowed_tools=allowed_tools)

    async def bound_evaluate_node(state: AgentState) -> dict:
        return await evaluate_node(
            state,
            definition=definition,
            decision_provider=evaluate_decision_provider,
        )

    graph.add_node("route", bound_route_node)
    graph.add_node("execute", bound_execute_node)
    graph.add_node("observe", observe_node)
    graph.add_node("evaluate", bound_evaluate_node)
    graph.add_node("pause", pause_node)
    graph.add_node("synthesize", synthesize_node)

    graph.add_edge(START, "route")
    graph.add_conditional_edges(
        "route",
        route_after_route,
        {
            "execute": "execute",
            "synthesize": "synthesize",
        },
    )
    graph.add_conditional_edges(
        "execute",
        route_after_execute,
        {
            "observe": "observe",
            "pause": "pause",
        },
    )
    graph.add_edge("observe", "evaluate")
    graph.add_conditional_edges(
        "evaluate",
        route_after_evaluate,
        {
            "execute": "execute",
            "pause": "pause",
            "synthesize": "synthesize",
        },
    )
    graph.add_edge("pause", END)
    graph.add_edge("synthesize", END)

    return graph.compile()


def route_after_execute(state: AgentState) -> str:
    if state.get("status") == "paused":
        return "pause"
    return "observe"

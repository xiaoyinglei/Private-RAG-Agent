from __future__ import annotations

from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from rag.agent.core.definition import AgentDefinition
from rag.agent.graphs.nodes.execute import execute_node
from rag.agent.graphs.nodes.goal_runtime import (
    controller_node,
    initialize_goal_node,
    reduce_observations_node,
    route_after_controller,
)
from rag.agent.graphs.nodes.llm_decide import ToolDecisionProvider, llm_decide_node
from rag.agent.graphs.nodes.pause import pause_node
from rag.agent.graphs.nodes.retrieval_hint import RetrievalHintProvider
from rag.agent.graphs.nodes.synthesize import SynthesisRunner, synthesize_node
from rag.agent.state import AgentState
from rag.agent.tools.registry import ToolRegistry


def build_agent_graph(
    *,
    definition: AgentDefinition,
    tool_registry: ToolRegistry,
    tool_decision_provider: ToolDecisionProvider | None = None,
    retrieval_hint_provider: RetrievalHintProvider | None = None,
    synthesis_runner: SynthesisRunner | None = None,
    checkpointer: BaseCheckpointSaver[str] | MemorySaver | None = None,
) -> Any:
    graph = StateGraph(AgentState)
    allowed_tools = frozenset(definition.allowed_tools)
    effective_synthesis_runner = (
        None if definition.agent_type == "synthesize" else synthesis_runner
    )
    effective_retrieval_hint_provider = retrieval_hint_provider

    async def bound_initialize_goal_node(state: AgentState) -> dict[str, Any]:
        return await initialize_goal_node(
            state,
            retrieval_hint_provider=effective_retrieval_hint_provider,
        )

    async def bound_controller_node(state: AgentState) -> dict[str, Any]:
        return controller_node(
            state,
            definition=definition,
            has_tool_decision_provider=tool_decision_provider is not None,
        )

    async def bound_execute_node(state: AgentState) -> dict[str, Any]:
        return await execute_node(state, tool_registry=tool_registry, allowed_tools=allowed_tools)

    async def bound_llm_decide_node(state: AgentState) -> dict[str, Any]:
        return await llm_decide_node(
            state,
            definition=definition,
            decision_provider=tool_decision_provider,
        )

    async def bound_synthesize_node(state: AgentState) -> dict[str, Any]:
        return await synthesize_node(
            state,
            synthesis_runner=effective_synthesis_runner,
        )

    graph.add_node("initialize_goal", bound_initialize_goal_node)
    graph.add_node("controller", bound_controller_node)
    graph.add_node("execute", bound_execute_node)
    graph.add_node("reduce_observations", reduce_observations_node)
    graph.add_node("llm_decide", bound_llm_decide_node)
    graph.add_node("pause", pause_node)
    graph.add_node("finalize", bound_synthesize_node)

    graph.add_edge(START, "initialize_goal")
    graph.add_edge("initialize_goal", "controller")
    graph.add_conditional_edges(
        "controller",
        route_after_controller,
        {
            "execute": "execute",
            "llm_decide": "llm_decide",
            "pause": "pause",
            "finalize": "finalize",
        },
    )
    graph.add_conditional_edges(
        "execute",
        route_after_execute,
        {
            "reduce_observations": "reduce_observations",
            "pause": "pause",
            "finalize": "finalize",
        },
    )
    graph.add_edge("reduce_observations", "controller")
    graph.add_edge("llm_decide", "controller")
    graph.add_conditional_edges(
        "pause",
        route_after_pause,
        {
            "execute": "execute",
            "controller": "controller",
            "end": END,
        },
    )
    graph.add_edge("finalize", END)

    return graph.compile(checkpointer=checkpointer)


def route_after_execute(state: AgentState) -> str:
    if state.get("status") == "paused":
        return "pause"
    if state.get("status") == "failed":
        return "finalize"
    return "reduce_observations"


def route_after_pause(state: AgentState) -> str:
    decision = state.get("user_decision", "")
    if decision == "allow_once":
        return "execute"
    if decision == "abort":
        return "end"
    # deny, continue, and other responses return to the parent controller.
    return "controller"

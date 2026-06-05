from __future__ import annotations

from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from rag.agent.core.definition import AgentDefinition
from rag.agent.graphs.nodes import goal_runtime as graph_goal_nodes
from rag.agent.graphs.nodes.execute import run_tools_guarded
from rag.agent.graphs.nodes.llm_decide import ToolDecisionProvider, decide_next
from rag.agent.graphs.nodes.pause import pause_node
from rag.agent.graphs.nodes.retrieval_hint import RetrievalHintProvider
from rag.agent.graphs.nodes.synthesize import SynthesisRunner, build_answer
from rag.agent.state import AgentState
from rag.agent.tools.registry import ToolRegistry

init_goal = graph_goal_nodes.init_goal
control_turn = graph_goal_nodes.control_turn
route_after_control = graph_goal_nodes.route_after_control


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

    async def bound_init_goal(state: AgentState) -> dict[str, Any]:
        return await init_goal(
            state,
            retrieval_hint_provider=effective_retrieval_hint_provider,
        )

    async def bound_control_turn(state: AgentState) -> dict[str, Any]:
        return control_turn(
            state,
            definition=definition,
            has_tool_decision_provider=tool_decision_provider is not None,
        )

    async def bound_run_tools(state: AgentState) -> dict[str, Any]:
        return await run_tools_guarded(
            state,
            tool_registry=tool_registry,
            allowed_tools=allowed_tools,
        )

    async def bound_decide_next(state: AgentState) -> dict[str, Any]:
        return await decide_next(
            state,
            definition=definition,
            decision_provider=tool_decision_provider,
        )

    async def bound_build_answer(state: AgentState) -> dict[str, Any]:
        return await build_answer(
            state,
            synthesis_runner=effective_synthesis_runner,
        )

    graph.add_node("initialize_goal", bound_init_goal)
    graph.add_node("controller", bound_control_turn)
    graph.add_node("execute", bound_run_tools)
    graph.add_node("llm_decide", bound_decide_next)
    graph.add_node("pause", pause_node)
    graph.add_node("finalize", bound_build_answer)

    graph.add_edge(START, "initialize_goal")
    graph.add_edge("initialize_goal", "controller")
    graph.add_conditional_edges(
        "controller",
        route_after_control,
        {
            "execute": "execute",
            "llm_decide": "llm_decide",
            "pause": "pause",
            "finalize": "finalize",
        },
    )
    graph.add_conditional_edges(
        "execute",
        route_after_tools,
        {
            "controller": "controller",
            "pause": "pause",
            "finalize": "finalize",
        },
    )
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


def route_after_tools(state: AgentState) -> str:
    if state.get("status") == "paused":
        return "pause"
    if state.get("status") == "failed":
        return "finalize"
    return "controller"


def route_after_pause(state: AgentState) -> str:
    decision = state.get("user_decision", "")
    if decision == "allow_once":
        return "execute"
    if decision == "abort":
        return "end"
    # deny, continue, and other responses return to the parent controller.
    return "controller"


route_after_execute = route_after_tools

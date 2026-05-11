from __future__ import annotations

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from rag.agent.core.definition import AgentDefinition
from rag.agent.core.task import SubTaskNode
from rag.agent.graphs.nodes.evaluate import (
    EvaluateDecisionProvider,
    evaluate_node,
    route_after_evaluate,
)
from rag.agent.graphs.nodes.execute import execute_node
from rag.agent.graphs.nodes.execute_subagent import (
    SubAgentRunner,
    SubAgentRunResult,
    execute_subagent_node,
)
from rag.agent.graphs.nodes.observe import observe_node
from rag.agent.graphs.nodes.pause import pause_node
from rag.agent.graphs.nodes.plan import PlanProvider, plan_node, route_after_plan
from rag.agent.graphs.nodes.route import RouteProvider, route_after_route, route_node
from rag.agent.graphs.nodes.synthesize import synthesize_node
from rag.agent.state import AgentState
from rag.agent.tools.registry import ToolRegistry


class _MissingSubAgentRunner:
    async def run_subtask(
        self,
        *,
        subtask: SubTaskNode,
        parent_state: AgentState,
    ) -> SubAgentRunResult:
        del subtask, parent_state
        raise RuntimeError("subagent_runner_missing")


def build_agent_graph(
    *,
    definition: AgentDefinition,
    tool_registry: ToolRegistry,
    evaluate_decision_provider: EvaluateDecisionProvider | None = None,
    plan_provider: PlanProvider | None = None,
    route_provider: RouteProvider | None = None,
    subagent_runner: SubAgentRunner | None = None,
    checkpointer: MemorySaver | None = None,
):
    graph = StateGraph(AgentState)
    allowed_tools = frozenset(definition.allowed_tools)
    effective_subagent_runner = subagent_runner or _MissingSubAgentRunner()
    effective_route_provider = route_provider

    async def bound_route_node(state: AgentState) -> dict:
        if effective_route_provider is not None:
            result = effective_route_provider.route(state)
            if hasattr(result, "__await__"):
                result = await result
            return result
        return route_node(state)

    async def bound_execute_node(state: AgentState) -> dict:
        return await execute_node(state, tool_registry=tool_registry, allowed_tools=allowed_tools)

    async def bound_execute_subagent_node(state: AgentState) -> dict:
        return await execute_subagent_node(state, subagent_runner=effective_subagent_runner)

    async def bound_evaluate_node(state: AgentState) -> dict:
        return await evaluate_node(
            state,
            definition=definition,
            decision_provider=evaluate_decision_provider,
        )

    async def bound_plan_node(state: AgentState) -> dict:
        return await plan_node(
            state,
            definition=definition,
            plan_provider=plan_provider,
        )

    graph.add_node("route", bound_route_node)
    graph.add_node("plan", bound_plan_node)
    graph.add_node("execute", bound_execute_node)
    graph.add_node("execute_subagent", bound_execute_subagent_node)
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
            "plan": "plan",
            "synthesize": "synthesize",
        },
    )
    graph.add_conditional_edges(
        "plan",
        route_after_plan,
        {
            "evaluate": "evaluate",
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
    graph.add_edge("execute_subagent", "evaluate")
    graph.add_edge("observe", "evaluate")
    graph.add_conditional_edges(
        "evaluate",
        route_after_evaluate,
        {
            "execute": "execute",
            "execute_subagent": "execute_subagent",
            "pause": "pause",
            "synthesize": "synthesize",
        },
    )
    graph.add_conditional_edges(
        "pause",
        route_after_pause,
        {
            "execute": "execute",
            "evaluate": "evaluate",
            "end": END,
        },
    )
    graph.add_edge("synthesize", END)

    return graph.compile(checkpointer=checkpointer)


def route_after_execute(state: AgentState) -> str:
    if state.get("status") == "paused":
        return "pause"
    return "observe"


def route_after_pause(state: AgentState) -> str:
    decision = state.get("user_decision", "")
    if decision == "allow_once":
        return "execute"
    if decision == "abort":
        return "end"
    # deny, continue, 及其他 → 重新评估
    return "evaluate"

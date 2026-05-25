from __future__ import annotations

from inspect import isawaitable
from typing import Any

from rag.agent.core.definition import AgentDefinition
from rag.agent.goal_runtime import (
    GoalInitializer,
    StateReducer,
)
from rag.agent.graphs.nodes.retrieval_hint import RetrievalHintProvider, retrieval_hint_node
from rag.agent.loop.controller import AgentLoopController
from rag.agent.state import AgentState


async def initialize_goal_node(
    state: AgentState,
    *,
    retrieval_hint_provider: RetrievalHintProvider | None = None,
) -> dict[str, Any]:
    goal = state.get("goal_spec")
    update: dict[str, Any] = {}
    if goal is None:
        goal = GoalInitializer().initialize(state.get("task", ""))
        update.update(
            {
                "goal_spec": goal,
                "goal_requirements": goal.requirement_ids,
                "open_gaps": goal.open_gaps(),
                "decision_reason": state.get("decision_reason") or "goal_initialized",
            }
        )

    hint_update = await _retrieval_hint(
        state,
        retrieval_hint_provider=retrieval_hint_provider,
    )
    update.update(_retrieval_hint_update(hint_update))
    return update


def controller_node(
    state: AgentState,
    *,
    definition: AgentDefinition,
    has_tool_decision_provider: bool,
) -> dict[str, Any]:
    return AgentLoopController(
        definition=definition,
        has_tool_decision_provider=has_tool_decision_provider,
    ).advance(state)


def reduce_observations_node(state: AgentState) -> dict[str, Any]:
    return StateReducer().reduce_tool_results(dict(state))


def route_after_controller(state: AgentState) -> str:
    next_node = state.get("controller_next")
    if next_node in {
        "execute",
        "llm_decide",
        "pause",
        "finalize",
    }:
        return str(next_node)
    return "finalize"


async def _retrieval_hint(
    state: AgentState,
    *,
    retrieval_hint_provider: RetrievalHintProvider | None,
) -> dict[str, Any]:
    if retrieval_hint_provider is not None:
        result = retrieval_hint_provider.hint(state)
        if isawaitable(result):
            result = await result
        return dict(result)
    return retrieval_hint_node(state)


def _retrieval_hint_update(hint_update: dict[str, Any]) -> dict[str, Any]:
    update: dict[str, Any] = {}
    for key in ("retrieval_signals", "retrieval_signals_debug", "decision_reason"):
        if key in hint_update:
            update[key] = hint_update[key]
    return update


__all__ = [
    "controller_node",
    "initialize_goal_node",
    "reduce_observations_node",
    "route_after_controller",
]

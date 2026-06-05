from __future__ import annotations

from inspect import isawaitable
from typing import Any

from rag.agent.core.context import RunRegistry
from rag.agent.core.definition import AgentDefinition
from rag.agent.goal_runtime import (
    GoalBuilder,
    ObservationExtractor,
)
from rag.agent.graphs.nodes.retrieval_hint import RetrievalHintProvider, retrieval_hint_node
from rag.agent.loop.controller import TurnController
from rag.agent.memory.compactor import MemoryCompactor
from rag.agent.planning import PlanTracker
from rag.agent.state import AgentState


async def init_goal(
    state: AgentState,
    *,
    retrieval_hint_provider: RetrievalHintProvider | None = None,
) -> dict[str, Any]:
    goal = state.get("goal_spec")
    update: dict[str, Any] = {}
    if goal is None:
        goal = GoalBuilder().initialize(state.get("task", ""))
        update.update(
            {
                "goal_spec": goal,
                "goal_requirements": goal.requirement_ids,
                "open_gaps": goal.open_gaps(),
                "decision_reason": state.get("decision_reason") or "goal_initialized",
            }
        )
    if state.get("agent_plan") is None:
        open_gaps = update.get("open_gaps") or state.get("open_gaps", []) or goal.open_gaps()
        plan, events = PlanTracker().initialize(
            task=state.get("task", ""),
            open_gaps=open_gaps,
        )
        pending_calls = state.get("pending_tool_calls", [])
        if pending_calls:
            plan, progress_events = PlanTracker().record_decision_progress(
                plan,
                tool_call_ids=[call.tool_call_id for call in pending_calls],
                tool_names=[call.tool_name for call in pending_calls],
            )
            events = [*events, *progress_events]
        update["agent_plan"] = plan
        update["plan_events"] = events

    hint_update = await _retrieval_hint(
        state,
        retrieval_hint_provider=retrieval_hint_provider,
    )
    update.update(_retrieval_hint_update(hint_update))
    return update


def control_turn(
    state: AgentState,
    *,
    definition: AgentDefinition,
    has_tool_decision_provider: bool,
) -> dict[str, Any]:
    return TurnController(
        definition=definition,
        has_tool_decision_provider=has_tool_decision_provider,
    ).advance(state)


def extract_obs_legacy(state: AgentState) -> dict[str, Any]:
    update = ObservationExtractor().reduce_tool_results(dict(state))
    if not update:
        return update
    plan, events = PlanTracker().record_observation_progress(
        state.get("agent_plan"),
        observations=update.get("structured_observations", []),
        satisfied_requirement_ids=update.get("satisfied_requirements", []),
    )
    if plan is not None:
        update["agent_plan"] = plan
    if events:
        update["plan_events"] = events
    try:
        memory_store = RunRegistry.get(state["run_config"].run_id).memory_store
    except KeyError:
        memory_store = None
    return MemoryCompactor(
        policy=state["run_config"].memory_policy,
        store=memory_store,
    ).compact_update(dict(state), update)


def route_after_control(state: AgentState) -> str:
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
    "GoalBuilder",
    "ObservationExtractor",
    "TurnController",
    "control_turn",
    "extract_obs_legacy",
    "init_goal",
    "route_after_control",
]

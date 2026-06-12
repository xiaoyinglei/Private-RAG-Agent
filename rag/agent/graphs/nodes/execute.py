from __future__ import annotations

from typing import Any

from rag.agent.core.context import RunRegistry
from rag.agent.core.definition import AgentDefinition
from rag.agent.core.tool_execution import ToolBatchRequest, ToolExecutionService
from rag.agent.goal_runtime import ObservationExtractor
from rag.agent.memory.compactor import MemoryCompactor
from rag.agent.planning import PlanTracker
from rag.agent.state import AgentState, _merge_tool_results
from rag.agent.tools.registry import ToolRegistry


async def run_tools_raw(
    state: AgentState,
    *,
    tool_registry: ToolRegistry,
    allowed_tools: frozenset[str],
    definition: AgentDefinition | None = None,
) -> dict[str, Any]:
    """Legacy graph adapter over the neutral tool execution service."""

    pending = state.get("pending_tool_calls", [])
    if not pending:
        return {}

    result = await ToolExecutionService(
        tool_registry=tool_registry,
        require_idempotent_parallel=False,
        pause_on_ambiguous=False,
    ).execute_batch(
        ToolBatchRequest(
            calls=tuple(pending),
            run_config=state["run_config"],
            allowed_tools=allowed_tools,
            approved_tool_call_ids=tuple(
                state.get("approved_tool_call_ids", [])
            ),
            denied_tool_call_ids=tuple(
                state.get("denied_tool_call_ids", [])
            ),
            retrieval_signals=state.get("retrieval_signals"),
        ),
        state=state,
        definition=definition,
    )

    if result.human_input_request is not None:
        return {
            "status": "paused",
            "human_input_request": result.human_input_request,
            "needs_user_input": result.human_input_request.question,
            "pending_tool_calls": list(result.pending_tool_calls),
            "tool_results": list(result.tool_results),
        }

    update: dict[str, Any] = {
        "tool_results": list(result.tool_results),
        "pending_tool_calls": list(result.pending_tool_calls),
        "run_config": result.run_config,
    }
    if result.context_budget is not None:
        error = next(
            (
                tool_result.error
                for tool_result in result.tool_results
                if tool_result.error is not None
                and tool_result.error.code == "context_overflow"
            ),
            None,
        )
        update.update(
            {
                "status": "paused",
                "needs_user_input": (
                    error.message
                    if error is not None
                    else "Required context exceeds the model budget."
                ),
                "decision_reason": "context_overflow",
                "controller_next": "pause",
                "context_budget": result.context_budget,
            }
        )
    return update


async def run_tools_guarded(
    state: AgentState,
    *,
    tool_registry: ToolRegistry,
    allowed_tools: frozenset[str],
    definition: AgentDefinition | None = None,
) -> dict[str, Any]:
    """Execute, extract observations, and compact before graph checkpointing."""

    raw_update = await run_tools_raw(
        state,
        tool_registry=tool_registry,
        allowed_tools=allowed_tools,
        definition=definition,
    )
    if not raw_update:
        return raw_update
    if raw_update.get("status") == "paused":
        return raw_update

    update = dict(raw_update)
    transient_state = dict(state)
    raw_tool_results = list(raw_update.get("tool_results", []))
    if raw_tool_results:
        transient_state["tool_results"] = _merge_tool_results(
            list(state.get("tool_results", [])),
            raw_tool_results,
        )

    observation_update = ObservationExtractor().reduce_tool_results(
        transient_state
    )
    if observation_update:
        plan, events = PlanTracker().record_observation_progress(
            state.get("agent_plan"),
            observations=observation_update.get(
                "structured_observations",
                [],
            ),
            satisfied_requirement_ids=observation_update.get(
                "satisfied_requirements",
                [],
            ),
        )
        update.update(observation_update)
        if plan is not None:
            update["agent_plan"] = plan
        if events:
            update["plan_events"] = events

    try:
        memory_store = RunRegistry.get(
            state["run_config"].run_id
        ).memory_store
    except KeyError:
        memory_store = None
    return MemoryCompactor(
        policy=state["run_config"].memory_policy,
        store=memory_store,
    ).compact_update(dict(state), update)


__all__ = ["run_tools_guarded", "run_tools_raw"]

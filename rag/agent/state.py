from __future__ import annotations

from collections.abc import Iterable

from langchain_core.messages import BaseMessage

from rag.agent.core.context import AgentRunConfig
from rag.agent.core.runtime_diagnostics import RuntimeDiagnostic
from rag.agent.core.turn_contracts import ThinkOutput, ToolCallPlan
from rag.agent.loop.state import LoopState, create_loop_state
from rag.agent.planning import AgentPlan, PlanEvent, PlanUpdate

AgentState = LoopState


def create_agent_state(
    *,
    task: str,
    run_config: AgentRunConfig,
    messages: Iterable[BaseMessage] = (),
    pending_tool_calls: Iterable[ToolCallPlan] = (),
    approved_tool_call_ids: Iterable[str] = (),
    denied_tool_call_ids: Iterable[str] = (),
    runtime_diagnostics: Iterable[RuntimeDiagnostic] = (),
) -> AgentState:
    """Compatibility factory returning the canonical LoopState."""

    state = create_loop_state(
        task=task,
        run_config=run_config,
        messages=messages,
        pending_tool_calls=pending_tool_calls,
        runtime_diagnostics=runtime_diagnostics,
    )
    state["approved_tool_call_ids"] = list(approved_tool_call_ids)
    state["denied_tool_call_ids"] = list(denied_tool_call_ids)
    return state


def agent_state_to_loop_state(state: AgentState) -> LoopState:
    """Compatibility identity adapter for callers using the former name."""

    return state


__all__ = [
    "AgentPlan",
    "AgentState",
    "PlanEvent",
    "PlanUpdate",
    "ThinkOutput",
    "ToolCallPlan",
    "agent_state_to_loop_state",
    "create_agent_state",
]

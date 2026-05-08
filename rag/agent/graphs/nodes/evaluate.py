from __future__ import annotations

from collections.abc import Awaitable
from inspect import isawaitable
from typing import Protocol

from pydantic import ValidationError

from rag.agent.core.definition import AgentDefinition
from rag.agent.state import AgentState, ThinkOutput


class EvaluateDecisionProvider(Protocol):
    def decide(
        self,
        state: AgentState,
        *,
        definition: AgentDefinition,
        budget_remaining: int,
    ) -> ThinkOutput | dict[str, object] | Awaitable[ThinkOutput | dict[str, object]]: ...


async def evaluate_node(
    state: AgentState,
    *,
    definition: AgentDefinition,
    decision_provider: EvaluateDecisionProvider | None = None,
) -> dict:
    iteration = state.get("iteration", 0)

    from rag.agent.core.context import RuntimeRegistry

    try:
        handles = RuntimeRegistry.get(state["run_config"].run_id)
    except KeyError:
        return {"status": "failed", "stop_reason": "runtime_handles_missing"}
    budget_remaining = await handles.budget_ledger.remaining()
    if budget_remaining <= 0:
        return {"status": "failed", "stop_reason": "budget_exhausted"}

    pending = state.get("pending_tool_calls", [])
    executed_batch = bool(state.get("tool_results"))
    next_iteration = iteration + 1 if executed_batch else iteration
    if pending and next_iteration >= definition.max_iterations:
        return {"status": "failed", "stop_reason": "max_iterations", "iteration": next_iteration}

    if pending:
        return {"status": "running", "iteration": next_iteration}

    if decision_provider is not None:
        decision = await _call_decision_provider(
            decision_provider,
            state,
            definition=definition,
            budget_remaining=budget_remaining,
        )
        if decision is not None:
            return _apply_decision(decision, next_iteration=next_iteration)

    return {"status": "done", "stop_reason": "no_pending_tools", "iteration": next_iteration}


async def _call_decision_provider(
    decision_provider: EvaluateDecisionProvider,
    state: AgentState,
    *,
    definition: AgentDefinition,
    budget_remaining: int,
) -> ThinkOutput | None:
    try:
        raw_decision = decision_provider.decide(
            state,
            definition=definition,
            budget_remaining=budget_remaining,
        )
        if isawaitable(raw_decision):
            raw_decision = await raw_decision
        return ThinkOutput.model_validate(raw_decision)
    except ValidationError:
        return None
    except Exception as exc:
        return ThinkOutput(
            action="pause",
            thought="evaluate decision provider failed",
            needs_user_input=f"Evaluate decision provider failed: {exc}",
            confidence=0.0,
        )


def _apply_decision(decision: ThinkOutput, *, next_iteration: int) -> dict:
    if decision.action == "execute":
        if not decision.tool_calls:
            return {
                "status": "failed",
                "stop_reason": "empty_tool_calls",
                "iteration": next_iteration,
            }
        return {
            "status": "running",
            "pending_tool_calls": decision.tool_calls,
            "route_reason": "evaluate_decision",
            "iteration": next_iteration + 1,
        }
    if decision.action == "pause":
        return {
            "status": "paused",
            "needs_user_input": decision.needs_user_input,
            "iteration": next_iteration,
        }
    if decision.action == "synthesize":
        return {
            "status": "done",
            "stop_reason": decision.stop_reason or "synthesize",
            "iteration": next_iteration,
        }
    return {"status": "running", "iteration": next_iteration}


def route_after_evaluate(state: AgentState) -> str:
    if state.get("status") == "paused":
        return "pause"
    if state.get("status") in {"done", "failed"}:
        return "synthesize"
    return "execute"

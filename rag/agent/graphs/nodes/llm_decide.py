from __future__ import annotations

from collections.abc import Awaitable
from inspect import isawaitable
from typing import Any, Protocol

from pydantic import ValidationError

from rag.agent.core.context import RuntimeRegistry
from rag.agent.core.definition import AgentDefinition
from rag.agent.memory.injector import ContextInjector
from rag.agent.memory.models import InjectedContext
from rag.agent.state import AgentState, ThinkOutput, ToolCallPlan


class ToolDecisionProvider(Protocol):
    def decide(
        self,
        state: AgentState,
        *,
        definition: AgentDefinition,
        budget_remaining: int,
        context: InjectedContext,
    ) -> ThinkOutput | dict[str, object] | Awaitable[ThinkOutput | dict[str, object]]: ...


async def llm_decide_node(
    state: AgentState,
    *,
    definition: AgentDefinition,
    decision_provider: ToolDecisionProvider | None = None,
) -> dict[str, Any]:
    if decision_provider is None:
        return {
            "status": "done",
            "stop_reason": "no_decision_provider",
            "controller_next": "finalize",
        }

    try:
        handles = RuntimeRegistry.get(state["run_config"].run_id)
    except KeyError:
        return {
            "status": "failed",
            "stop_reason": "runtime_handles_missing",
            "controller_next": "finalize",
        }

    budget_remaining = await handles.budget_ledger.remaining()
    if budget_remaining <= 0:
        return {
            "status": "failed",
            "stop_reason": "budget_exhausted",
            "controller_next": "finalize",
        }

    context = ContextInjector(
        max_context_tokens=state["run_config"].budget_total,
    ).assemble(definition=definition, state=state)
    decision = await _call_decision_provider(
        decision_provider,
        state,
        definition=definition,
        budget_remaining=budget_remaining,
        context=context,
    )
    update = _apply_decision(
        decision,
        next_iteration=state.get("iteration", 0),
        used_tool_call_ids=_used_tool_call_ids(state),
    )
    update["context_budget"] = context.context_budget
    return update


async def _call_decision_provider(
    decision_provider: ToolDecisionProvider,
    state: AgentState,
    *,
    definition: AgentDefinition,
    budget_remaining: int,
    context: InjectedContext,
) -> ThinkOutput:
    try:
        raw_decision = decision_provider.decide(
            state,
            definition=definition,
            budget_remaining=budget_remaining,
            context=context,
        )
        if isawaitable(raw_decision):
            raw_decision = await raw_decision
        return ThinkOutput.model_validate(raw_decision)
    except ValidationError:
        return ThinkOutput(
            action="pause",
            thought="LLM decision provider returned invalid output",
            needs_user_input="LLM decision provider failed to produce a valid decision",
            confidence=0.0,
        )
    except Exception as exc:
        return ThinkOutput(
            action="pause",
            thought="LLM decision provider failed",
            needs_user_input=f"LLM decision provider failed: {exc}",
            confidence=0.0,
        )


def _apply_decision(
    decision: ThinkOutput,
    *,
    next_iteration: int,
    used_tool_call_ids: set[str] | None = None,
) -> dict[str, Any]:
    if decision.action == "execute":
        tool_calls = _normalize_tool_call_ids(
            decision.tool_calls,
            used_tool_call_ids=used_tool_call_ids or set(),
        )
        if not tool_calls:
            return {
                "status": "failed",
                "stop_reason": "empty_tool_calls",
                "iteration": next_iteration,
                "controller_next": "finalize",
            }
        return {
            "status": "running",
            "pending_tool_calls": tool_calls,
            "decision_reason": "llm_decision",
            "iteration": next_iteration,
            "controller_next": "execute",
        }
    if decision.action == "pause":
        return {
            "status": "paused",
            "needs_user_input": decision.needs_user_input,
            "iteration": next_iteration,
            "controller_next": "pause",
        }
    if decision.action == "synthesize":
        return {
            "status": "done",
            "stop_reason": decision.stop_reason or "synthesize",
            "iteration": next_iteration,
            "controller_next": "finalize",
        }
    return {"status": "running", "iteration": next_iteration, "controller_next": "finalize"}


def _used_tool_call_ids(state: AgentState) -> set[str]:
    used = {
        result.tool_call_id
        for result in state.get("tool_results", [])
        if getattr(result, "tool_call_id", None)
    }
    used.update(
        call.tool_call_id
        for call in state.get("pending_tool_calls", [])
        if getattr(call, "tool_call_id", None)
    )
    return used


def _normalize_tool_call_ids(
    tool_calls: list[ToolCallPlan],
    *,
    used_tool_call_ids: set[str],
) -> list[ToolCallPlan]:
    normalized: list[ToolCallPlan] = []
    used = set(used_tool_call_ids)
    for call in tool_calls:
        if call.tool_call_id not in used:
            used.add(call.tool_call_id)
            normalized.append(call)
            continue
        replacement = ToolCallPlan.create(call.tool_name, dict(call.arguments))
        while replacement.tool_call_id in used:
            replacement = ToolCallPlan.create(call.tool_name, dict(call.arguments))
        used.add(replacement.tool_call_id)
        normalized.append(replacement)
    return normalized


__all__ = ["ToolDecisionProvider", "llm_decide_node"]

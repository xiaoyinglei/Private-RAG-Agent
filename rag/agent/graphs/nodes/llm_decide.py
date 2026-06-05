from __future__ import annotations

from collections.abc import Awaitable, Sequence
from inspect import isawaitable
from typing import Any, Protocol

from pydantic import ValidationError

from rag.agent.core.context import RunRegistry
from rag.agent.core.definition import AgentDefinition
from rag.agent.memory.injector import ContextBuilder
from rag.agent.memory.models import InjectedContext
from rag.agent.planning import AgentPlan, PlanEvent, PlanTracker
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


async def decide_next(
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
        handles = RunRegistry.get(state["run_config"].run_id)
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

    context = ContextBuilder(
        max_context_tokens=state["run_config"].max_context_tokens
        or state["run_config"].budget_total,
    ).assemble(definition=definition, state=state)
    if context.context_budget.overflow:
        return {
            "status": "paused",
            "needs_user_input": "Context budget overflow; cannot safely call the decision model.",
            "decision_reason": "context_overflow",
            "controller_next": "pause",
            "context_budget": context.context_budget,
        }
    decision = await _call_decision_provider(
        decision_provider,
        state,
        definition=definition,
        budget_remaining=budget_remaining,
        context=context,
    )
    finalization_authorized = _finalization_authorized(state)
    decision = _redirect_premature_synthesis_to_summarize(
        decision,
        state=state,
        definition=definition,
        context=context,
        finalization_authorized=finalization_authorized,
    )
    update = _apply_decision(
        decision,
        next_iteration=state.get("iteration", 0),
        used_tool_call_ids=_used_tool_call_ids(state),
        finalization_authorized=finalization_authorized,
        current_plan=state.get("agent_plan"),
        allowed_tool_names=frozenset(definition.allowed_tools),
        open_gap_ids=_open_gap_ids(state),
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
    finalization_authorized: bool = False,
    current_plan: AgentPlan | None = None,
    allowed_tool_names: frozenset[str] = frozenset(),
    open_gap_ids: Sequence[str] = (),
) -> dict[str, Any]:
    planning_update: dict[str, Any] = {}
    working_plan = current_plan
    plan_events: list[PlanEvent] = []
    if working_plan is not None and decision.plan_update is not None:
        working_plan, plan_update_events = PlanTracker().apply_llm_update(
            working_plan,
            decision.plan_update,
            allowed_tool_names=allowed_tool_names,
            open_gap_ids=frozenset(open_gap_ids),
        )
        plan_events.extend(plan_update_events)

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
        if working_plan is not None:
            working_plan, progress_events = PlanTracker().record_decision_progress(
                working_plan,
                tool_call_ids=[call.tool_call_id for call in tool_calls],
                tool_names=[call.tool_name for call in tool_calls],
            )
            plan_events.extend(progress_events)
            planning_update["agent_plan"] = working_plan
        if plan_events:
            planning_update["plan_events"] = plan_events
        return {
            "status": "running",
            "pending_tool_calls": tool_calls,
            "decision_reason": "llm_decision",
            "iteration": next_iteration,
            "controller_next": "execute",
            **planning_update,
        }
    if decision.action == "pause":
        if working_plan is not None:
            planning_update["agent_plan"] = working_plan
        if plan_events:
            planning_update["plan_events"] = plan_events
        return {
            "status": "paused",
            "needs_user_input": decision.needs_user_input,
            "iteration": next_iteration,
            "controller_next": "pause",
            **planning_update,
        }
    if decision.action == "synthesize":
        if not finalization_authorized:
            if working_plan is not None:
                planning_update["agent_plan"] = working_plan
            if plan_events:
                planning_update["plan_events"] = plan_events
            return {
                "status": "paused",
                "stop_reason": "premature_synthesis",
                "needs_user_input": (
                    "The model requested finalization before required goal conditions were satisfied."
                ),
                "insufficient_evidence_flag": True,
                "iteration": next_iteration,
                "controller_next": "pause",
                **planning_update,
            }
        if working_plan is not None:
            working_plan, completion_events = PlanTracker().record_completion(working_plan)
            plan_events.extend(completion_events)
            planning_update["agent_plan"] = working_plan
        if plan_events:
            planning_update["plan_events"] = plan_events
        return {
            "status": "done",
            "stop_reason": "goal_satisfied",
            "iteration": next_iteration,
            "controller_next": "finalize",
            **planning_update,
        }
    if working_plan is not None:
        planning_update["agent_plan"] = working_plan
    if plan_events:
        planning_update["plan_events"] = plan_events
    return {"status": "running", "iteration": next_iteration, "controller_next": "finalize", **planning_update}


def _redirect_premature_synthesis_to_summarize(
    decision: ThinkOutput,
    *,
    state: AgentState,
    definition: AgentDefinition,
    context: InjectedContext,
    finalization_authorized: bool,
) -> ThinkOutput:
    if decision.action != "synthesize" or finalization_authorized:
        return decision
    if "llm_summarize" not in definition.allowed_tools:
        return decision
    if _has_attempted_summarize(state) or not _has_summarizable_context(state):
        return decision

    return ThinkOutput(
        action="execute",
        tool_calls=[
            ToolCallPlan.create(
                "llm_summarize",
                {
                    "task": state.get("task", ""),
                    "context_sections": _summarization_context_sections(context),
                    "evidence_ids": _tool_output_values(state, "evidence_ids"),
                    "citation_ids": _tool_output_values(state, "citation_ids"),
                },
            )
        ],
        thought=(
            "Model requested finalization before goal authorization; "
            "redirecting to llm_summarize to produce an answer candidate."
        ),
        confidence=decision.confidence,
    )


def _has_attempted_summarize(state: AgentState) -> bool:
    return any(
        getattr(result, "tool_name", None) == "llm_summarize"
        for result in state.get("tool_results", [])
    )


def _has_summarizable_context(state: AgentState) -> bool:
    if state.get("evidence"):
        return True
    summarizable_tools = {
        "vector_search",
        "keyword_search",
        "grounding",
        "rerank",
        "asset_read_slice",
        "asset_analyze",
        "rag_search_answer",
        "read_file",
        "run_python",
    }
    return any(
        getattr(result, "status", None) == "ok"
        and (
            getattr(result, "tool_name", None) in summarizable_tools
            or str(getattr(result, "tool_name", "")).startswith("agent_")
        )
        for result in state.get("tool_results", [])
    )


def _summarization_context_sections(context: InjectedContext) -> list[str]:
    sections: list[str] = []
    for name in ("tool_results", "evidence", "working_memory", "message_tail"):
        try:
            section = context.section(name)
        except KeyError:
            continue
        if section.content.strip():
            sections.append(f"[{section.name}]\n{section.content}")
    return sections


def _tool_output_values(state: AgentState, field: str) -> list[str]:
    values: list[str] = []
    seen: set[str] = set()
    for result in state.get("tool_results", []):
        output = getattr(result, "output", None)
        for item in getattr(output, field, []) or []:
            text = str(item)
            if text and text not in seen:
                values.append(text)
                seen.add(text)
    return values


def _finalization_authorized(state: AgentState) -> bool:
    report = state.get("satisfaction_report")
    return bool(
        getattr(report, "is_done", False)
        and getattr(report, "reason", None) == "goal_satisfied"
        and not getattr(report, "open_gaps", [])
        and not getattr(report, "conflicts", [])
    )


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


def _open_gap_ids(state: AgentState) -> list[str]:
    gap_ids: list[str] = []
    for gap in state.get("open_gaps", []):
        gap_id = getattr(gap, "gap_id", gap if isinstance(gap, str) else None)
        if isinstance(gap_id, str) and gap_id:
            gap_ids.append(gap_id)
    return gap_ids


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


llm_decide_node = decide_next

__all__ = ["ToolDecisionProvider", "decide_next", "llm_decide_node"]

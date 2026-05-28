from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from rag.agent.core.context import RuntimeRegistry
from rag.agent.core.definition import AgentDefinition
from rag.agent.goal_runtime import (
    ContextBinding,
    ContextBindingAssessor,
    ContextUnit,
    SatisfactionChecker,
    SatisfactionReport,
)
from rag.agent.state import AgentState


class BindingAssessor(Protocol):
    def assess_bindings(
        self,
        state: dict[str, Any],
        *,
        context_units: Sequence[ContextUnit] | None = None,
    ) -> list[ContextBinding]: ...


class GoalChecker(Protocol):
    def check(self, state: dict[str, Any]) -> SatisfactionReport: ...


@dataclass(slots=True)
class AgentLoopController:
    """Advance the goal-driven agent loop by one control decision."""

    definition: AgentDefinition
    has_tool_decision_provider: bool
    binding_assessor: BindingAssessor = field(default_factory=ContextBindingAssessor)
    checker: GoalChecker = field(default_factory=SatisfactionChecker)

    def advance(self, state: AgentState) -> dict[str, Any]:
        try:
            RuntimeRegistry.get(state["run_config"].run_id)
        except KeyError:
            return {
                "status": "failed",
                "stop_reason": "runtime_handles_missing",
                "controller_next": "finalize",
            }

        if state.get("status") == "paused":
            return {"controller_next": "pause"}
        if state.get("status") in {"done", "failed"}:
            return {"controller_next": "finalize"}
        if state.get("pending_tool_calls"):
            if state.get("iteration", 0) >= self.definition.max_iterations:
                return {
                    "status": "failed",
                    "stop_reason": "max_iterations",
                    "controller_next": "finalize",
                }
            return {"status": "running", "controller_next": "execute"}

        assessed_bindings = self.binding_assessor.assess_bindings(
            dict(state),
            context_units=state.get("context_units", []),
        )
        effective_bindings = {
            getattr(binding, "key", str(index)): binding
            for index, binding in enumerate(
                [*state.get("context_bindings", []), *assessed_bindings]
            )
        }
        assessed_state = {
            **dict(state),
            "context_bindings": list(effective_bindings.values()),
        }
        report = self.checker.check(assessed_state)
        progress_made = bool(
            _gap_ids(state.get("open_gaps", [])) - _gap_ids(report.open_gaps)
        )
        if progress_made and report.is_stuck and not report.is_done:
            report = report.model_copy(update={"is_stuck": False, "reason": "open_gaps"})
        update: dict[str, Any] = {
            "satisfaction_report": report,
            "open_gaps": report.open_gaps,
            "conflicts": report.conflicts,
            "context_bindings": assessed_bindings,
        }
        if progress_made:
            update["no_progress_count"] = 0
        if report.is_done:
            update.update(
                {
                    "status": "done" if report.reason == "goal_satisfied" else "failed",
                    "stop_reason": report.reason,
                    "controller_next": "finalize",
                }
            )
            if report.reason != "goal_satisfied":
                update["insufficient_evidence_flag"] = True
            return update

        if report.is_stuck:
            update.update(
                {
                    "status": "paused",
                    "needs_user_input": "Agent made no progress toward the current goal.",
                    "controller_next": "pause",
                }
            )
            return update
        if self.has_tool_decision_provider:
            update["controller_next"] = "llm_decide"
            return update
        update.update(
            {
                "status": "paused",
                "needs_user_input": "No tool decision provider is available to close the remaining goal gaps.",
                "controller_next": "pause",
            }
        )
        return update


def _gap_ids(gaps: Sequence[object]) -> set[str]:
    ids: set[str] = set()
    for gap in gaps:
        if isinstance(gap, str):
            ids.add(gap)
            continue
        gap_id = getattr(gap, "gap_id", None)
        if isinstance(gap_id, str) and gap_id:
            ids.add(gap_id)
    return ids


__all__ = ["AgentLoopController", "BindingAssessor", "GoalChecker"]

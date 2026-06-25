from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from rag.agent.compat.goal_contract import GoalContractEvaluator, GoalSpec
from rag.agent.core.definition import AgentDefinition
from rag.agent.core.observations import ComputationResult, ContextBinding, EvidenceRef
from rag.agent.core.output_finalizer import (
    OutputValidationExhaustedError,
    StructuredOutputFinalizer,
    validated_final_output,
)
from rag.agent.core.output_models import ValidatedFinalOutput
from rag.agent.loop.state import (
    LoopState,
    StopHookFeedback,
    append_stop_hook_feedback,
    append_stop_hook_warning,
)
from rag.agent.loop.substate import FinishState
from rag.agent.tools.spec import ToolResult


class StopVerdict(BaseModel):
    model_config = ConfigDict(frozen=True)

    action: Literal["accept", "warn", "block", "halt"]
    code: str = Field(min_length=1, max_length=120)
    message: str | None = Field(default=None, max_length=1000)
    detail: dict[str, object] = Field(default_factory=dict)
    final_output: ValidatedFinalOutput | None = None


class StopHookOutcome(BaseModel):
    model_config = ConfigDict(frozen=True)

    action: Literal["accept", "warn", "block", "halt"]
    code: str
    message: str | None = None
    detail: dict[str, object] = Field(default_factory=dict)
    verdicts: tuple[StopVerdict, ...] = ()
    final_output: ValidatedFinalOutput | None = None

    @property
    def accepted(self) -> bool:
        return self.action in {"accept", "warn"}

    @property
    def blocked(self) -> bool:
        return self.action == "block"

    @property
    def halted(self) -> bool:
        return self.action == "halt"


class StopHook(Protocol):
    async def evaluate(
        self,
        *,
        state: LoopState,
        candidate: str,
    ) -> StopVerdict: ...


@dataclass(frozen=True)
class StopHookBinding:
    name: str
    hook: StopHook
    critical: bool


class StopHookRunner:
    def __init__(
        self,
        *,
        hooks: list[StopHookBinding] | tuple[StopHookBinding, ...],
        max_blocks: int,
    ) -> None:
        if max_blocks < 1:
            raise ValueError("max_blocks must be positive")
        self._hooks = tuple(hooks)
        self._max_blocks = max_blocks

    async def evaluate(
        self,
        *,
        state: LoopState,
        candidate: str,
    ) -> StopHookOutcome:
        verdicts: list[StopVerdict] = []
        final_output: ValidatedFinalOutput | None = None
        warned = False
        for binding in self._hooks:
            try:
                verdict = await binding.hook.evaluate(
                    state=state,
                    candidate=candidate,
                )
            except Exception as exc:
                verdict = StopVerdict(
                    action="halt" if binding.critical else "warn",
                    code=f"{binding.name}_failed",
                    message=str(exc) or type(exc).__name__,
                    detail={"error_type": type(exc).__name__},
                )
            verdicts.append(verdict)
            if verdict.final_output is not None:
                final_output = verdict.final_output

            if verdict.action == "warn":
                warned = True
                append_stop_hook_warning(
                    state,
                    StopHookFeedback(
                        code=verdict.code,
                        message=verdict.message or verdict.code,
                    ),
                )
                state["finish_state"] = FinishState(
                    feedback=list(state.get("stop_hook_feedback", [])),
                    warnings=list(state.get("stop_hook_warnings", [])),
                )
                continue
            if verdict.action == "block":
                feedback = append_stop_hook_feedback(
                    state,
                    StopHookFeedback(
                        code=verdict.code,
                        message=verdict.message or verdict.code,
                    ),
                )
                state["finish_state"] = FinishState(
                    feedback=list(state.get("stop_hook_feedback", [])),
                    warnings=list(state.get("stop_hook_warnings", [])),
                )
                if feedback.occurrences >= self._max_blocks:
                    return StopHookOutcome(
                        action="halt",
                        code="stop_hook_block_limit",
                        message=("Equivalent stop-hook feedback reached the configured block limit."),
                        detail={
                            "blocked_code": verdict.code,
                            "occurrences": feedback.occurrences,
                        },
                        verdicts=tuple(verdicts),
                        final_output=final_output,
                    )
                return StopHookOutcome(
                    action="block",
                    code=verdict.code,
                    message=verdict.message,
                    detail=verdict.detail,
                    verdicts=tuple(verdicts),
                    final_output=final_output,
                )
            if verdict.action == "halt":
                return StopHookOutcome(
                    action="halt",
                    code=verdict.code,
                    message=verdict.message,
                    detail=verdict.detail,
                    verdicts=tuple(verdicts),
                    final_output=final_output,
                )

        return StopHookOutcome(
            action="warn" if warned else "accept",
            code="accepted_with_warnings" if warned else "accepted",
            verdicts=tuple(verdicts),
            final_output=final_output,
        )


class StructuredOutputStopHook:
    def __init__(
        self,
        *,
        definition: AgentDefinition,
        finalizer: StructuredOutputFinalizer | None,
    ) -> None:
        self._definition = definition
        self._finalizer = finalizer

    async def evaluate(
        self,
        *,
        state: LoopState,
        candidate: str,
    ) -> StopVerdict:
        if self._finalizer is None:
            raise RuntimeError("structured output is configured without a finalizer")
        try:
            output = await _await_output(
                self._finalizer.finalize(
                    definition=self._definition,
                    state=state,
                    candidate_text=candidate,
                )
            )
        except OutputValidationExhaustedError as exc:
            return StopVerdict(
                action="halt",
                code="structured_output_invalid",
                message=str(exc),
                detail={
                    "attempts": exc.attempts,
                    "validation_errors": exc.validation_errors,
                },
            )
        return StopVerdict(
            action="accept",
            code="structured_output_valid",
            final_output=validated_final_output(output),
        )


class GoalContractStopHook:
    def __init__(self, *, goal_spec: GoalSpec) -> None:
        self._goal_spec = goal_spec

    @staticmethod
    def _collect_evidence_refs(
        tool_results: list[ToolResult],
    ) -> list[EvidenceRef]:
        """Derive evidence_refs from tool_results instead of deprecated state field."""
        refs: list[EvidenceRef] = []
        for r in tool_results:
            if r.status == "ok" and r.output is not None:
                er = getattr(r.output, "evidence_refs", None)
                if isinstance(er, list):
                    for item in er:
                        if isinstance(item, EvidenceRef):
                            refs.append(item)
                        elif isinstance(item, dict):
                            refs.append(EvidenceRef.model_validate(item))
        return refs

    @staticmethod
    def _collect_computation_results(
        tool_results: list[ToolResult],
    ) -> list[ComputationResult]:
        """Derive computation_results from tool_results instead of deprecated state field."""
        results: list[ComputationResult] = []
        for r in tool_results:
            if r.status == "ok" and r.output is not None:
                cr = getattr(r.output, "computation_results", None)
                if isinstance(cr, list):
                    for item in cr:
                        if isinstance(item, ComputationResult):
                            results.append(item)
                        elif isinstance(item, dict):
                            results.append(ComputationResult.model_validate(item))
        return results

    @staticmethod
    def _collect_context_bindings(
        tool_results: list[ToolResult],
    ) -> list[ContextBinding]:
        """Derive context_bindings from tool_results instead of deprecated state field."""
        bindings: list[ContextBinding] = []
        for r in tool_results:
            if r.status == "ok" and r.output is not None:
                cb = getattr(r.output, "context_bindings", None)
                if isinstance(cb, list):
                    for item in cb:
                        if isinstance(item, ContextBinding):
                            bindings.append(item)
                        elif isinstance(item, dict):
                            bindings.append(ContextBinding.model_validate(item))
        return bindings

    async def evaluate(
        self,
        *,
        state: LoopState,
        candidate: str,
    ) -> StopVerdict:
        tool_results = list(state.get("tool_results", []))
        evaluation = GoalContractEvaluator().evaluate(
            goal_spec=self._goal_spec,
            candidate=candidate,
            evidence_refs=self._collect_evidence_refs(tool_results),
            computation_results=self._collect_computation_results(tool_results),
            context_bindings=self._collect_context_bindings(tool_results),
        )
        if evaluation.satisfied:
            return StopVerdict(
                action="accept",
                code="goal_contract_satisfied",
            )
        return StopVerdict(
            action="block",
            code="goal_contract_unsatisfied",
            message="; ".join(issue.description for issue in evaluation.issues)
            or "Explicit goal contract is not satisfied.",
            detail={
                "unsatisfied_issue_ids": evaluation.issue_ids,
            },
        )


def build_stop_hooks(
    *,
    definition: AgentDefinition,
    output_finalizer: StructuredOutputFinalizer | None = None,
    goal_spec: GoalSpec | None = None,
) -> tuple[StopHookBinding, ...]:
    hooks: list[StopHookBinding] = []
    if goal_spec is not None:
        hooks.append(
            StopHookBinding(
                name="goal_contract",
                hook=GoalContractStopHook(goal_spec=goal_spec),
                critical=True,
            )
        )
    if definition.output_model is not None:
        hooks.append(
            StopHookBinding(
                name="structured_output",
                hook=StructuredOutputStopHook(
                    definition=definition,
                    finalizer=output_finalizer,
                ),
                critical=True,
            )
        )
    return tuple(hooks)


async def _await_output(value: object) -> BaseModel:
    from inspect import isawaitable

    if isawaitable(value):
        value = await value
    if not isinstance(value, BaseModel):
        raise TypeError("structured output finalizer returned a non-model value")
    return value


__all__ = [
    "GoalContractStopHook",
    "StopHook",
    "StopHookBinding",
    "StopHookOutcome",
    "StopHookRunner",
    "StopVerdict",
    "StructuredOutputStopHook",
    "build_stop_hooks",
]

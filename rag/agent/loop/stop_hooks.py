from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from rag.agent.compat.goal_contract import GoalContractEvaluator, GoalSpec
from rag.agent.core.definition import AgentDefinition
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
                continue
            if verdict.action == "block":
                feedback = append_stop_hook_feedback(
                    state,
                    StopHookFeedback(
                        code=verdict.code,
                        message=verdict.message or verdict.code,
                    ),
                )
                if feedback.occurrences >= self._max_blocks:
                    return StopHookOutcome(
                        action="halt",
                        code="stop_hook_block_limit",
                        message=(
                            "Equivalent stop-hook feedback reached the "
                            "configured block limit."
                        ),
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
            raise RuntimeError(
                "structured output is configured without a finalizer"
            )
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

    async def evaluate(
        self,
        *,
        state: LoopState,
        candidate: str,
    ) -> StopVerdict:
        evaluation = GoalContractEvaluator().evaluate(
            goal_spec=self._goal_spec,
            candidate=candidate,
            evidence_refs=state["evidence_refs"],
            computation_results=state["computation_results"],
            context_bindings=state["context_bindings"],
        )
        if evaluation.satisfied:
            return StopVerdict(
                action="accept",
                code="goal_contract_satisfied",
            )
        return StopVerdict(
            action="block",
            code="goal_contract_unsatisfied",
            message="; ".join(
                issue.description
                for issue in evaluation.issues
            )
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

from __future__ import annotations

from collections.abc import Awaitable
from inspect import isawaitable
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from rag.agent.loop.state import (
    LoopState,
    ModelTurn,
    ModelTurnDraft,
    materialize_model_turn,
)

MAX_FINALIZATION_EVENTS = 20


class SynthesisRunResult(Protocol):
    final_answer: str | None


class CompatibilitySynthesisRunner(Protocol):
    def run_synthesis(
        self,
        *,
        parent_state: object,
    ) -> SynthesisRunResult | Awaitable[SynthesisRunResult]: ...


class FinalizationEvent(BaseModel):
    model_config = ConfigDict(frozen=True)

    code: str = Field(min_length=1, max_length=120)
    status: str = Field(min_length=1, max_length=40)
    message: str | None = Field(default=None, max_length=1000)


class FinishCandidateBuildError(RuntimeError):
    pass


class FinishCandidateBuilder:
    """Normalize finish intent before strict ModelTurn validation."""

    def __init__(
        self,
        *,
        synthesis_runner: CompatibilitySynthesisRunner | None = None,
    ) -> None:
        self._synthesis_runner = synthesis_runner
        self._events: list[FinalizationEvent] = []

    @property
    def events(self) -> tuple[FinalizationEvent, ...]:
        return tuple(self._events)

    async def build(
        self,
        draft: ModelTurnDraft,
        *,
        state: LoopState,
    ) -> ModelTurn:
        if draft.tool_calls or draft.action != "finish" or _nonempty(
            draft.final_answer
        ):
            return materialize_model_turn(draft)
        if self._synthesis_runner is None:
            error = FinishCandidateBuildError(
                "finish intent has no candidate and no compatibility builder"
            )
            self._record(
                FinalizationEvent(
                    code="finish_candidate_builder_failed",
                    status="error",
                    message=str(error),
                )
            )
            raise error

        try:
            result = self._synthesis_runner.run_synthesis(
                parent_state=state,
            )
            if isawaitable(result):
                result = await result
        except Exception as exc:
            self._record(
                FinalizationEvent(
                    code="finish_candidate_builder_failed",
                    status="error",
                    message=str(exc) or type(exc).__name__,
                )
            )
            raise FinishCandidateBuildError(
                f"compatibility finish candidate builder failed: {exc}"
            ) from exc

        candidate = result.final_answer
        if not _nonempty(candidate):
            error = FinishCandidateBuildError(
                "compatibility finish candidate builder returned no answer"
            )
            self._record(
                FinalizationEvent(
                    code="finish_candidate_builder_failed",
                    status="error",
                    message=str(error),
                )
            )
            raise error
        self._record(
            FinalizationEvent(
                code="finish_candidate_builder_used",
                status="ok",
                message="Built a finish candidate through compatibility synthesis.",
            )
        )
        return materialize_model_turn(
            draft,
            finish_candidate=candidate,
        )

    def _record(self, event: FinalizationEvent) -> None:
        self._events = [*self._events, event][-MAX_FINALIZATION_EVENTS:]


def _nonempty(value: str | None) -> bool:
    return bool(value and value.strip())


__all__ = [
    "CompatibilitySynthesisRunner",
    "FinalizationEvent",
    "FinishCandidateBuildError",
    "FinishCandidateBuilder",
    "MAX_FINALIZATION_EVENTS",
    "SynthesisRunResult",
]

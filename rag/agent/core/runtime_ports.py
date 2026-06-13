from __future__ import annotations

from collections.abc import Awaitable
from typing import TYPE_CHECKING, Protocol

from rag.agent.core.definition import AgentDefinition
from rag.agent.memory.models import InjectedContext

if TYPE_CHECKING:
    from rag.agent.loop.state import LoopState
    from rag.agent.state import ThinkOutput


class ToolDecisionProvider(Protocol):
    """Compatibility port for callers that still provide legacy decisions."""

    def decide(
        self,
        state: LoopState,
        *,
        definition: AgentDefinition,
        budget_remaining: int,
        context: InjectedContext,
    ) -> (
        ThinkOutput
        | dict[str, object]
        | Awaitable[ThinkOutput | dict[str, object]]
    ): ...


class RetrievalHintProvider(Protocol):
    """Metadata-only retrieval hint port; it cannot select a loop branch."""

    def hint(
        self,
        state: LoopState,
    ) -> (
        dict[str, object]
        | Awaitable[dict[str, object]]
    ): ...


__all__ = [
    "RetrievalHintProvider",
    "ToolDecisionProvider",
]

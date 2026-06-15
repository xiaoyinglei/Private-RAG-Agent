from __future__ import annotations

from collections.abc import Awaitable
from typing import Protocol

from typing_extensions import TypedDict

from rag.agent.loop.state import LoopState
from rag.schema.query import RetrievalSignals


class RetrievalHintUpdate(TypedDict):
    decision_reason: str
    retrieval_signals: RetrievalSignals
    retrieval_signals_debug: dict[str, object]


class RetrievalHintProvider(Protocol):
    """Metadata-only retrieval hint port; it cannot select a loop branch."""

    def hint(
        self,
        state: LoopState,
    ) -> RetrievalHintUpdate | Awaitable[RetrievalHintUpdate]: ...


__all__ = [
    "RetrievalHintProvider",
    "RetrievalHintUpdate",
]

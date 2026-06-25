"""Provider-neutral typed messages for the agent loop.

These types are the internal protocol between AgentLoop, ModelTurnProvider,
and checkpoint storage.  Provider adapters (OpenAI, Anthropic, fallback)
translate to/from these types at the boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field

# ── Stop reason ──


class StopReason(StrEnum):
    """Normalized stop reason.  AgentLoop control flow depends *only* on this."""

    TOOL_USE = "tool_use"
    END_TURN = "end_turn"
    MAX_TOKENS = "max_tokens"


# ── Tool call ──


@dataclass(frozen=True)
class ToolCall:
    """A single tool invocation requested by the model.

    ``id`` is model-provided or fallback-generated (``tc_<hex>``).
    ``input`` is already parsed into a dict — never a JSON string.
    """

    id: str
    name: str
    input: dict[str, Any]

    @classmethod
    def create(cls, name: str, input: dict[str, Any]) -> ToolCall:
        return cls(id=f"tc_{uuid4().hex[:12]}", name=name, input=input)


# ── Model message ──


@dataclass(frozen=True)
class ModelMessage:
    """A single message in the agent conversation.

    Roles:
    - ``system``: generated each turn by AgentMessageAssembler, never stored
    - ``user``: task or user input
    - ``assistant``: model response, optionally with tool_calls
    - ``tool``: tool result, requires ``tool_call_id``
    """

    role: Literal["system", "user", "assistant", "tool"]
    content: str
    tool_calls: tuple[ToolCall, ...] = ()
    tool_call_id: str | None = None  # required when role="tool"


# ── Tool use result ──


class ToolUseResult(BaseModel):
    """Model output after a turn.  The loop kernel uses ``stop_reason``
    (the normalized enum) for control flow; ``raw_stop_reason`` is
    debug-only and must NOT enter checkpoints."""

    tool_calls: list[ToolCall] = Field(default_factory=list)
    text: str = ""
    stop_reason: StopReason
    raw_stop_reason: str  # debug / logging only


__all__ = [
    "ModelMessage",
    "StopReason",
    "ToolCall",
    "ToolUseResult",
]

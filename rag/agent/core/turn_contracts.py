from __future__ import annotations

from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field

from rag.agent.planning import PlanUpdate


class ToolCallPlan(BaseModel):
    tool_call_id: str
    tool_name: str
    arguments: dict[str, object]

    @classmethod
    def create(
        cls,
        tool_name: str,
        arguments: dict[str, object],
    ) -> ToolCallPlan:
        return cls(
            tool_call_id=f"tc_{uuid4().hex[:12]}",
            tool_name=tool_name,
            arguments=arguments,
        )


class ThinkOutput(BaseModel):
    """Compatibility model normalized into ModelTurnDraft at the loop boundary."""

    action: Literal["execute", "synthesize", "pause"]
    tool_calls: list[ToolCallPlan] = Field(default_factory=list)
    thought: str
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    stop_reason: str | None = None
    needs_user_input: str | None = None
    plan_update: PlanUpdate | None = None


__all__ = ["ThinkOutput", "ToolCallPlan"]

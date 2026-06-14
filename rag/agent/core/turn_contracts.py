from __future__ import annotations

from uuid import uuid4

from pydantic import BaseModel


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


__all__ = ["ToolCallPlan"]

"""Minimal typed model for PlanTracker -- replaces StructuredObservation in planner path."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ToolExecutionObservation(BaseModel):
    """Generic execution status that PlanTracker can read via getattr.

    Replaces the dict-based hack that earlier PR2 plan versions would have used.
    PlanTracker uses getattr(observation, "tool_call_id") etc. -- this model
    provides those attributes as typed fields.
    """

    tool_call_id: str
    tool_name: str
    status: str  # "ok" | "error"
    related_step_ids: list[str] = Field(default_factory=list)
    metadata: dict[str, object] = Field(default_factory=dict)


__all__ = ["ToolExecutionObservation"]

"""Trace records for model request and tool execution boundaries."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ModelRequestTrace(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: str
    model: str
    visible_tools: list[str] = Field(default_factory=list)
    hidden_tools: list[str] = Field(default_factory=list)
    schema_bytes: int = 0
    tool_choice: str | dict[str, Any] | None = None
    latency_ms: float = 0.0


class ToolExecutionTrace(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tool_call_id: str
    tool_name: str
    status: str
    recoverable: bool
    error_code: str | None = None
    latency_ms: float = 0.0

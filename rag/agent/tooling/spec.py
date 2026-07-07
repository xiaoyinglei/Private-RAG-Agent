"""Claude-like tool boundary data types."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ToolDomain(StrEnum):
    WORKSPACE = "workspace"
    EXECUTION = "execution"
    KNOWLEDGE = "knowledge"
    DISCOVERY = "discovery"
    MCP = "mcp"


class ToolRisk(StrEnum):
    READ = "read"
    WRITE = "write"
    EXECUTE = "execute"
    NETWORK = "network"
    DESTRUCTIVE = "destructive"


class ToolExposure(StrEnum):
    NORMAL = "normal"
    DEFERRED = "deferred"
    INTERNAL = "internal"


class ToolCall(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    arguments: dict[str, Any] = Field(default_factory=dict)


class ToolResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tool_call_id: str
    tool_name: str
    ok: bool
    content: str
    data: dict[str, Any] = Field(default_factory=dict)
    recoverable: bool = True
    error_code: str | None = None


class ToolSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    input_schema: dict[str, Any]
    domain: ToolDomain
    risk: ToolRisk
    exposure: ToolExposure = ToolExposure.NORMAL
    timeout_seconds: float = Field(gt=0)
    output_limit_chars: int = Field(default=50_000, gt=0)

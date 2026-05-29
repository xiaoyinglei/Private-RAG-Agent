from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, Field, model_validator


@dataclass(frozen=True)
class ToolPermissions:
    read_db: bool = False
    write_db: bool = False
    read_object_store: bool = False
    embed: bool = False
    generate: bool = False
    external_network: bool = False
    kg_mutation: bool = False
    user_data: bool = False
    read_fs: bool = False
    write_fs: bool = False
    execute_code: bool = False


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_model: type[BaseModel]
    output_model: type[BaseModel]
    error_model: type[BaseModel]
    permissions: ToolPermissions
    timeout_seconds: float
    max_retries: int = 0
    idempotent: bool = False
    token_budget_cost: int = 0
    requires_confirmation: bool = False
    audit_log: bool = False


class ToolError(BaseModel):
    code: str
    message: str
    retryable: bool
    detail: dict[str, object] = Field(default_factory=dict)


class ToolResult(BaseModel):
    tool_call_id: str
    tool_name: str
    status: Literal["ok", "error"]
    output: BaseModel | None = None
    error: ToolError | None = None
    latency_ms: float
    token_used: int = 0
    retry_count: int = 0

    @model_validator(mode="after")
    def _check_exclusivity(self) -> ToolResult:
        if self.status == "ok":
            if self.output is None:
                raise ValueError("output is required when status='ok'")
            if self.error is not None:
                raise ValueError("error must be None when status='ok'")
        if self.status == "error":
            if self.error is None:
                raise ValueError("error is required when status='error'")
            if self.output is not None:
                raise ValueError("output must be None when status='error'")
        return self

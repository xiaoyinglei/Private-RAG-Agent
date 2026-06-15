from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Literal

from pydantic import (
    BaseModel,
    Field,
    SerializationInfo,
    field_serializer,
    field_validator,
    model_validator,
)


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
    concurrency_safe: bool = False
    is_read_only: bool = False
    work_budget_cost: int = 0
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
    work_units_used: int = 0
    # Legacy checkpoint field. New tool execution records work_units_used instead.
    token_used: int = 0
    retry_count: int = 0

    @field_validator("output", mode="before")
    @classmethod
    def _restore_typed_output(cls, value: object) -> object:
        if not isinstance(value, dict) or value.get("__rag_model_payload__") is not True:
            return value
        module = value.get("module")
        name = value.get("name")
        data = value.get("data")
        if not isinstance(module, str) or not isinstance(name, str):
            raise ValueError("typed tool output payload is missing module/name")
        model_cls = getattr(importlib.import_module(module), name)
        if not isinstance(model_cls, type) or not issubclass(model_cls, BaseModel):
            raise ValueError(f"typed tool output is not a Pydantic model: {module}.{name}")
        return model_cls.model_validate(data)

    @field_serializer("output")
    def _serialize_typed_output(
        self,
        output: BaseModel | None,
        info: SerializationInfo,
    ) -> object:
        if output is None:
            return None
        if info.mode == "json":
            return output.model_dump(mode="json")
        return {
            "__rag_model_payload__": True,
            "module": output.__class__.__module__,
            "name": output.__class__.__name__,
            "data": output.model_dump(mode="json"),
        }

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

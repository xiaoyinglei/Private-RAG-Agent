from __future__ import annotations

import math
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import Literal

type JsonValue = (
    str
    | int
    | float
    | bool
    | None
    | tuple[JsonValue, ...]
    | Mapping[str, JsonValue]
)


def _freeze_json(value: object, *, path: str) -> JsonValue:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} must contain only finite JSON numbers")
        return value
    if isinstance(value, Mapping):
        frozen: dict[str, JsonValue] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} keys must be strings")
            frozen[key] = _freeze_json(item, path=f"{path}.{key}")
        return MappingProxyType(frozen)
    if isinstance(value, (list, tuple)):
        return tuple(
            _freeze_json(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        )
    raise TypeError(f"{path} must contain only JSON-compatible values")


def _freeze_mapping(value: Mapping[str, object], *, path: str) -> Mapping[str, JsonValue]:
    frozen = _freeze_json(value, path=path)
    if not isinstance(frozen, Mapping):
        raise TypeError(f"{path} must be a mapping")
    return frozen


def _require_non_empty_string(value: object, *, field_name: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    if not value:
        raise ValueError(f"{field_name} must not be empty")


class ToolEffect(StrEnum):
    READ_WORKSPACE = "read_workspace"
    WRITE_WORKSPACE = "write_workspace"
    EXECUTE_PROCESS = "execute_process"
    NETWORK = "network"
    DESTRUCTIVE = "destructive"


class CancellationMode(StrEnum):
    COOPERATIVE = "cooperative"
    MANAGED_PROCESS = "managed_process"
    REMOTE_BEST_EFFORT = "remote_best_effort"
    NOT_CANCELLABLE = "not_cancellable"


class InterruptBehavior(StrEnum):
    CANCEL = "cancel"
    FINISH_CURRENT = "finish_current"


@dataclass(frozen=True, slots=True)
class ToolCallOrigin:
    request_id: str
    toolset_revision: str
    exposed_tool_names: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_non_empty_string(self.request_id, field_name="origin request_id")
        _require_non_empty_string(
            self.toolset_revision,
            field_name="origin toolset_revision",
        )
        if not isinstance(self.exposed_tool_names, (list, tuple)):
            raise TypeError("origin exposed_tool_names must be a list or tuple")
        exposed_tool_names = tuple(self.exposed_tool_names)
        for name in exposed_tool_names:
            _require_non_empty_string(name, field_name="origin exposed tool name")
        object.__setattr__(self, "exposed_tool_names", exposed_tool_names)


@dataclass(frozen=True, slots=True)
class ToolDefinition:
    """Immutable model-facing projection of a tool."""

    name: str
    description: str
    input_schema: Mapping[str, JsonValue]

    def __post_init__(self) -> None:
        _require_non_empty_string(self.name, field_name="tool name")
        _require_non_empty_string(self.description, field_name="tool description")
        object.__setattr__(
            self,
            "input_schema",
            _freeze_mapping(self.input_schema, path="input_schema"),
        )


@dataclass(frozen=True, slots=True)
class ToolCall:
    tool_call_id: str
    tool_name: str
    arguments: Mapping[str, JsonValue]
    origin: ToolCallOrigin

    def __post_init__(self) -> None:
        _require_non_empty_string(self.tool_call_id, field_name="tool_call_id")
        _require_non_empty_string(self.tool_name, field_name="tool_name")
        if not isinstance(self.origin, ToolCallOrigin):
            raise TypeError("origin must be a ToolCallOrigin")
        object.__setattr__(
            self,
            "arguments",
            _freeze_mapping(self.arguments, path="arguments"),
        )


@dataclass(frozen=True, slots=True)
class ToolContentBlock:
    """One model-visible block with an immutable JSON-compatible payload."""

    type: Literal["text", "image", "resource"]
    data: Mapping[str, JsonValue]

    def __post_init__(self) -> None:
        if self.type not in {"text", "image", "resource"}:
            raise ValueError("tool content block type must be text, image, or resource")
        object.__setattr__(
            self,
            "data",
            _freeze_mapping(self.data, path="content block data"),
        )


@dataclass(frozen=True, slots=True)
class ArtifactReference:
    """Stable reference to an artifact kept outside model content."""

    artifact_id: str
    media_type: str | None = None
    name: str | None = None

    def __post_init__(self) -> None:
        _require_non_empty_string(self.artifact_id, field_name="artifact_id")
        if self.media_type is not None and not isinstance(self.media_type, str):
            raise TypeError("media_type must be a string when provided")
        if self.name is not None and not isinstance(self.name, str):
            raise TypeError("name must be a string when provided")


@dataclass(frozen=True, slots=True)
class ToolResult:
    """Tool outcome with model-visible content separated from runtime metadata."""

    tool_call_id: str
    tool_name: str
    content: tuple[ToolContentBlock, ...] = ()
    structured_content: JsonValue | None = None
    is_error: bool = False
    error_code: str | None = None
    error_message: str | None = None
    retryable: bool = False
    truncated: bool = False
    metadata: Mapping[str, JsonValue] = field(default_factory=dict)
    attachments: tuple[ArtifactReference, ...] = ()

    def __post_init__(self) -> None:
        _require_non_empty_string(self.tool_call_id, field_name="tool_call_id")
        _require_non_empty_string(self.tool_name, field_name="tool_name")
        if self.is_error and not self.error_code:
            raise ValueError("error_code is required when is_error=True")
        if not self.is_error and (
            self.error_code is not None or self.error_message is not None or self.retryable
        ):
            raise ValueError("error fields require is_error=True")
        if any(not isinstance(block, ToolContentBlock) for block in self.content):
            raise TypeError("content must contain ToolContentBlock values")
        if any(not isinstance(item, ArtifactReference) for item in self.attachments):
            raise TypeError("attachments must contain ArtifactReference values")

        object.__setattr__(self, "content", tuple(self.content))
        object.__setattr__(self, "attachments", tuple(self.attachments))
        if self.structured_content is not None:
            object.__setattr__(
                self,
                "structured_content",
                _freeze_json(self.structured_content, path="structured_content"),
            )
        object.__setattr__(
            self,
            "metadata",
            _freeze_mapping(self.metadata, path="metadata"),
        )


type ValidateInput = Callable[[Mapping[str, JsonValue]], Mapping[str, JsonValue]]
type ToolRunner = Callable[[Mapping[str, JsonValue]], object | Awaitable[object]]
type NormalizeOutput = Callable[[object], ToolResult]
type ResolveToolUse = Callable[[Mapping[str, JsonValue]], frozenset[ToolEffect]]


_LOCAL_SIDE_EFFECTS = frozenset(
    {
        ToolEffect.WRITE_WORKSPACE,
        ToolEffect.EXECUTE_PROCESS,
        ToolEffect.DESTRUCTIVE,
    }
)


@dataclass(frozen=True, slots=True)
class Tool:
    definition: ToolDefinition
    validate_input: ValidateInput
    run: ToolRunner
    normalize_output: NormalizeOutput
    output_schema: Mapping[str, JsonValue] | None
    static_effects: frozenset[ToolEffect]
    resolve_use: ResolveToolUse
    execution_revision: str
    idempotent: bool
    concurrency_safe: bool
    cancellation_mode: CancellationMode
    interrupt_behavior: InterruptBehavior
    timeout_seconds: float
    max_model_output_bytes: int

    def __post_init__(self) -> None:
        if not isinstance(self.definition, ToolDefinition):
            raise TypeError("definition must be a ToolDefinition")
        for field_name in ("validate_input", "run", "normalize_output", "resolve_use"):
            if not callable(getattr(self, field_name)):
                raise TypeError(f"{field_name} must be callable")

        static_effects = frozenset(self.static_effects)
        if any(not isinstance(effect, ToolEffect) for effect in static_effects):
            raise TypeError("static_effects must contain ToolEffect values")
        object.__setattr__(self, "static_effects", static_effects)
        if (
            static_effects & _LOCAL_SIDE_EFFECTS
            and self.cancellation_mode is CancellationMode.NOT_CANCELLABLE
        ):
            raise ValueError(
                "local side-effecting tools cannot use cancellation mode not_cancellable"
            )

        _require_non_empty_string(
            self.execution_revision,
            field_name="execution_revision",
        )
        if not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be a positive finite number")
        if not isinstance(self.max_model_output_bytes, int) or isinstance(
            self.max_model_output_bytes, bool
        ):
            raise ValueError("max_model_output_bytes must be a positive integer")
        if self.max_model_output_bytes <= 0:
            raise ValueError("max_model_output_bytes must be a positive integer")
        if self.output_schema is not None:
            object.__setattr__(
                self,
                "output_schema",
                _freeze_mapping(self.output_schema, path="output_schema"),
            )

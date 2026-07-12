from __future__ import annotations

import math
from collections.abc import Awaitable, Callable, Iterable, Mapping
from dataclasses import dataclass, field, is_dataclass
from enum import StrEnum
from numbers import Real
from types import MappingProxyType
from typing import Literal, get_args, get_origin

from jsonschema import exceptions as jsonschema_exceptions  # type: ignore[import-untyped]
from jsonschema.protocols import (  # type: ignore[import-untyped]
    Validator as JsonSchemaValidator,
)
from jsonschema.validators import validator_for  # type: ignore[import-untyped]
from pydantic import AliasChoices, AliasPath, BaseModel
from pydantic import ValidationError as PydanticValidationError
from pydantic.fields import FieldInfo
from referencing import Registry
from referencing.exceptions import Unresolvable
from typing_extensions import is_typeddict

type JsonValue = (
    str
    | int
    | float
    | bool
    | None
    | tuple[JsonValue, ...]
    | Mapping[str, JsonValue]
)

type _MutableJsonValue = (
    str
    | int
    | float
    | bool
    | None
    | list[_MutableJsonValue]
    | dict[str, _MutableJsonValue]
)


_MAX_VALIDATION_PATH_LENGTH = 256
_MAX_VALIDATION_MESSAGE_LENGTH = 512


def _bounded_text(value: str, *, limit: int) -> str:
    value = " ".join(value.split())
    if len(value) <= limit:
        return value
    return f"{value[: limit - 1]}…"


class ToolValidationError(ValueError):
    """Bounded, caller-safe details for one tool schema validation failure."""

    path: str
    message: str

    def __init__(self, *, path: str, message: str) -> None:
        self.path = _bounded_text(path, limit=_MAX_VALIDATION_PATH_LENGTH)
        self.message = _bounded_text(
            message,
            limit=_MAX_VALIDATION_MESSAGE_LENGTH,
        )
        super().__init__(f"{self.path}: {self.message}")


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


def _thaw_json(value: JsonValue) -> _MutableJsonValue:
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def _canonical_json(value: object, *, subject: str) -> JsonValue:
    try:
        return _freeze_json(value, path=subject)
    except (TypeError, ValueError):
        raise ToolValidationError(
            path="$",
            message=f"{subject} must contain only finite JSON-compatible values",
        ) from None


def _canonical_mapping(
    value: Mapping[str, object],
    *,
    subject: str,
) -> Mapping[str, JsonValue]:
    canonical = _canonical_json(value, subject=subject)
    if not isinstance(canonical, Mapping):
        raise ToolValidationError(path="$", message=f"{subject} must be an object")
    return canonical


def _require_non_empty_string(value: object, *, field_name: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    if not value:
        raise ValueError(f"{field_name} must not be empty")


def _require_exact_bool(value: object, *, field_name: str) -> None:
    if type(value) is not bool:
        raise TypeError(f"{field_name} must be a bool")


def _validate_error_fields(
    *,
    is_error: object,
    error_code: object,
    error_message: object,
    retryable: object,
) -> None:
    _require_exact_bool(is_error, field_name="is_error")
    _require_exact_bool(retryable, field_name="retryable")
    if error_code is not None:
        _require_non_empty_string(error_code, field_name="error_code")
    if error_message is not None and not isinstance(error_message, str):
        raise TypeError("error_message must be a string when provided")
    if is_error and error_code is None:
        raise ValueError("error_code is required when is_error=True")
    if not is_error and (
        error_code is not None or error_message is not None or retryable
    ):
        raise ValueError("error fields require is_error=True")


class ToolEffect(StrEnum):
    READ_WORKSPACE = "read_workspace"
    WRITE_WORKSPACE = "write_workspace"
    EXECUTE_PROCESS = "execute_process"
    NETWORK = "network"
    DESTRUCTIVE = "destructive"


@dataclass(frozen=True, slots=True)
class ToolTarget:
    kind: str
    value: str

    def __post_init__(self) -> None:
        _require_non_empty_string(self.kind, field_name="target kind")
        _require_non_empty_string(self.value, field_name="target value")


@dataclass(frozen=True, slots=True)
class ResolvedToolUse:
    effects: frozenset[ToolEffect]
    targets: tuple[ToolTarget, ...]

    def __post_init__(self) -> None:
        effects = frozenset(self.effects)
        if any(not isinstance(effect, ToolEffect) for effect in effects):
            raise TypeError("effects must contain ToolEffect values")
        targets = tuple(self.targets)
        if any(not isinstance(target, ToolTarget) for target in targets):
            raise TypeError("targets must contain ToolTarget values")
        object.__setattr__(self, "effects", effects)
        object.__setattr__(self, "targets", targets)


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


def _freeze_content(
    content: tuple[ToolContentBlock, ...],
) -> tuple[ToolContentBlock, ...]:
    content = tuple(content)
    if any(not isinstance(block, ToolContentBlock) for block in content):
        raise TypeError("content must contain ToolContentBlock values")
    return content


def _freeze_attachments(
    attachments: tuple[ArtifactReference, ...],
) -> tuple[ArtifactReference, ...]:
    attachments = tuple(attachments)
    if any(not isinstance(item, ArtifactReference) for item in attachments):
        raise TypeError("attachments must contain ArtifactReference values")
    return attachments


@dataclass(frozen=True, slots=True)
class NormalizedToolOutput:
    """Identity-free canonical output produced by a tool adapter."""

    content: tuple[ToolContentBlock, ...] = ()
    structured_content: JsonValue | None = None
    is_error: bool = False
    error_code: str | None = None
    error_message: str | None = None
    retryable: bool = False
    metadata: Mapping[str, JsonValue] = field(default_factory=dict)
    attachments: tuple[ArtifactReference, ...] = ()

    def __post_init__(self) -> None:
        _validate_error_fields(
            is_error=self.is_error,
            error_code=self.error_code,
            error_message=self.error_message,
            retryable=self.retryable,
        )
        object.__setattr__(self, "content", _freeze_content(self.content))
        object.__setattr__(
            self,
            "attachments",
            _freeze_attachments(self.attachments),
        )
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
        _validate_error_fields(
            is_error=self.is_error,
            error_code=self.error_code,
            error_message=self.error_message,
            retryable=self.retryable,
        )
        _require_exact_bool(self.truncated, field_name="truncated")
        object.__setattr__(self, "content", _freeze_content(self.content))
        object.__setattr__(
            self,
            "attachments",
            _freeze_attachments(self.attachments),
        )
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
type NormalizeOutput = Callable[[object], NormalizedToolOutput]
type ResolveToolUse = Callable[[Mapping[str, JsonValue]], ResolvedToolUse]


def _json_path(parts: Iterable[str | int]) -> str:
    path = "$"
    for part in parts:
        path += f"[{part}]" if isinstance(part, int) else f".{part}"
    return _bounded_text(path, limit=_MAX_VALIDATION_PATH_LENGTH)


def _jsonschema_error_path(error: jsonschema_exceptions.ValidationError) -> str:
    parts = list(error.absolute_path)
    if error.validator == "required" and isinstance(error.instance, Mapping):
        for name in error.validator_value:
            if isinstance(name, str) and name not in error.instance:
                parts.append(name)
                break
    elif (
        error.validator == "additionalProperties"
        and isinstance(error.instance, Mapping)
        and not error.schema.get("patternProperties")
    ):
        properties = error.schema.get("properties", {})
        if isinstance(properties, Mapping):
            for name in error.instance:
                if name not in properties:
                    parts.append(name)
                    break
    return _json_path(parts)


def _tool_error_from_jsonschema_validation(
    error: jsonschema_exceptions.ValidationError,
) -> ToolValidationError:
    validator = error.validator if isinstance(error.validator, str) else "validation"
    return ToolValidationError(
        path=_jsonschema_error_path(error),
        message=f"{validator}: validation failed",
    )


def _tool_error_from_jsonschema_schema(
    error: jsonschema_exceptions.SchemaError,
) -> ToolValidationError:
    validator = error.validator if isinstance(error.validator, str) else "schema"
    return ToolValidationError(
        path=_json_path(error.absolute_path),
        message=f"{validator}: {error.message}",
    )


def _close_pydantic_object_schemas(value: _MutableJsonValue) -> None:
    if isinstance(value, list):
        for item in value:
            _close_pydantic_object_schemas(item)
        return
    if not isinstance(value, dict):
        return
    if (
        value.get("type") == "object"
        and "properties" in value
        and "additionalProperties" not in value
    ):
        value["additionalProperties"] = False
    for item in value.values():
        _close_pydantic_object_schemas(item)


def _json_schema_validator(
    schema: Mapping[str, JsonValue],
) -> JsonSchemaValidator:
    canonical_schema = _canonical_mapping(schema, subject="schema")
    schema_copy = _thaw_json(canonical_schema)
    if not isinstance(schema_copy, dict):
        raise ToolValidationError(path="$", message="schema must be an object")

    if "$schema" in schema_copy:
        validator_type = validator_for(schema_copy, default=None)
        if validator_type is None:
            raise ToolValidationError(
                path="$.$schema",
                message="unsupported JSON Schema dialect",
            )
    else:
        validator_type = validator_for(schema_copy)
    try:
        validator_type.check_schema(schema_copy)
    except jsonschema_exceptions.SchemaError as error:
        raise _tool_error_from_jsonschema_schema(error) from None
    return validator_type(schema_copy, registry=Registry())


def _validate_with_json_schema(
    validator: JsonSchemaValidator,
    canonical_value: JsonValue,
) -> None:
    instance = _thaw_json(canonical_value)
    try:
        error = next(validator.iter_errors(instance), None)
    except Unresolvable:
        raise ToolValidationError(
            path="$",
            message="schema reference could not be resolved locally",
        ) from None
    if error is not None:
        raise _tool_error_from_jsonschema_validation(error)


def _tool_error_from_pydantic(error: PydanticValidationError) -> ToolValidationError:
    details = error.errors(
        include_url=False,
        include_context=False,
        include_input=False,
    )
    if not details:
        return ToolValidationError(path="$", message="input validation failed")
    first = details[0]
    error_type = first.get("type", "validation")
    return ToolValidationError(
        path=_json_path(first.get("loc", ())),
        message=f"{error_type}: validation failed",
    )


def _field_input_paths(
    model: type[BaseModel],
    field_name: str,
    field: FieldInfo,
) -> tuple[tuple[str | int, ...], ...]:
    paths: list[tuple[str | int, ...]] = []
    validation_alias = field.validation_alias
    has_alias = validation_alias is not None or field.alias is not None
    if model.model_config.get("validate_by_alias", True):
        if isinstance(validation_alias, str):
            paths.append((validation_alias,))
        elif field.alias is not None:
            paths.append((field.alias,))
    if not has_alias or model.model_config.get("validate_by_name", False):
        paths.append((field_name,))
    return tuple(paths)


def _value_at_input_path(
    value: object,
    path: tuple[str | int, ...],
) -> tuple[bool, object]:
    current = value
    for part in path:
        if isinstance(part, str) and isinstance(current, Mapping):
            if part not in current:
                return False, None
            current = current[part]
        elif isinstance(part, int) and isinstance(current, (list, tuple)):
            if part < 0 or part >= len(current):
                return False, None
            current = current[part]
        else:
            return False, None
    return True, current


def _reject_ignored_extras_in_value(
    validated: object,
    original: object,
    *,
    path: tuple[str | int, ...],
) -> None:
    if isinstance(validated, BaseModel) and isinstance(original, Mapping):
        _reject_ignored_model_extras(validated, original, path=path)
        return
    if isinstance(validated, (list, tuple)) and isinstance(original, (list, tuple)):
        for index, (validated_item, original_item) in enumerate(
            zip(validated, original, strict=False)
        ):
            _reject_ignored_extras_in_value(
                validated_item,
                original_item,
                path=(*path, index),
            )
        return
    if isinstance(validated, Mapping) and isinstance(original, Mapping):
        if len(validated) != len(original):
            raise ToolValidationError(
                path=_json_path(path),
                message=(
                    "mapping keys changed cardinality during validation; "
                    "normalized keys must remain unique"
                ),
            )
        for (original_key, original_item), validated_item in zip(
            original.items(),
            validated.values(),
            strict=False,
        ):
            _reject_ignored_extras_in_value(
                validated_item,
                original_item,
                path=(*path, original_key),
            )


def _reject_ignored_model_extras(
    validated: BaseModel,
    original: Mapping[str, object],
    *,
    path: tuple[str | int, ...],
) -> None:
    model = type(validated)
    if validated.model_extra:
        canonical_names = set(model.model_fields) | set(model.model_computed_fields)
        for key in original:
            if key in validated.model_extra and key in canonical_names:
                raise ToolValidationError(
                    path=_json_path((*path, key)),
                    message="extra key collides with canonical field name",
                )
    consumed_keys: set[str] = set()
    for field_name, model_field in model.model_fields.items():
        for input_path in _field_input_paths(model, field_name, model_field):
            found, original_value = _value_at_input_path(original, input_path)
            if not found:
                continue
            top_level_key = input_path[0]
            if isinstance(top_level_key, str):
                consumed_keys.add(top_level_key)
            _reject_ignored_extras_in_value(
                getattr(validated, field_name),
                original_value,
                path=(*path, *input_path),
            )
            break

    extra_policy = model.model_config.get("extra")
    if extra_policy in (None, "ignore"):
        for key in original:
            if key not in consumed_keys:
                raise ToolValidationError(
                    path=_json_path((*path, key)),
                    message="extra_forbidden: Extra inputs are not permitted",
                )
    elif extra_policy == "allow" and validated.model_extra:
        for key, extra_value in validated.model_extra.items():
            if key in original:
                _reject_ignored_extras_in_value(
                    extra_value,
                    original[key],
                    path=(*path, key),
                )


def _audit_pydantic_annotation(
    annotation: object,
    *,
    path: tuple[str | int, ...],
    visited: set[type[BaseModel]],
) -> None:
    if is_dataclass(annotation):
        raise ToolValidationError(
            path=_json_path(path),
            message="unsupported Pydantic input shape: dataclass",
        )
    if is_typeddict(annotation):
        raise ToolValidationError(
            path=_json_path(path),
            message="unsupported Pydantic input shape: TypedDict",
        )
    if get_origin(annotation) in (set, frozenset):
        raise ToolValidationError(
            path=_json_path(path),
            message="unsupported Pydantic input shape: set or frozenset",
        )
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        _audit_pydantic_model(annotation, path=path, visited=visited)
        return
    for argument in get_args(annotation):
        _audit_pydantic_annotation(argument, path=path, visited=visited)


def _audit_pydantic_model(
    model: type[BaseModel],
    *,
    path: tuple[str | int, ...] = (),
    visited: set[type[BaseModel]] | None = None,
) -> None:
    if visited is None:
        visited = set()
    if getattr(model, "__pydantic_root_model__", False):
        raise ToolValidationError(
            path=_json_path(path),
            message="unsupported Pydantic input shape: RootModel",
        )
    if model in visited:
        return
    visited.add(model)
    for field_name, model_field in model.model_fields.items():
        validation_alias = model_field.validation_alias
        if isinstance(validation_alias, (AliasPath, AliasChoices)):
            raise ToolValidationError(
                path=_json_path((*path, field_name)),
                message=(
                    "unsupported validation alias: complex alias paths and choices "
                    "cannot be represented faithfully in tool JSON Schema"
                ),
            )
        _audit_pydantic_annotation(
            model_field.annotation,
            path=(*path, field_name),
            visited=visited,
        )


def pydantic_input(
    model: type[BaseModel],
) -> tuple[dict[str, JsonValue], ValidateInput]:
    """Build a closed builtin schema and its canonical Pydantic validator."""

    if not isinstance(model, type) or not issubclass(model, BaseModel):
        raise TypeError("model must be a BaseModel subclass")
    _audit_pydantic_model(model)
    schema: _MutableJsonValue = model.model_json_schema(mode="validation")
    _close_pydantic_object_schemas(schema)
    if not isinstance(schema, dict):
        raise ToolValidationError(path="$", message="Pydantic schema must be an object")
    canonical_schema = _canonical_mapping(schema, subject="schema")
    _json_schema_validator(canonical_schema)

    def validate(arguments: Mapping[str, JsonValue]) -> Mapping[str, JsonValue]:
        try:
            validated = model.model_validate(arguments)
        except PydanticValidationError as error:
            raise _tool_error_from_pydantic(error) from None
        _reject_ignored_model_extras(validated, arguments, path=())
        dumped = validated.model_dump(mode="json")
        return _canonical_mapping(dumped, subject="input")

    return dict(canonical_schema), validate


def json_schema_input(schema: Mapping[str, JsonValue]) -> ValidateInput:
    """Build an input validator directly from a complete JSON Schema document."""

    validator = _json_schema_validator(schema)

    def validate(arguments: Mapping[str, JsonValue]) -> Mapping[str, JsonValue]:
        canonical_arguments = _canonical_mapping(arguments, subject="input")
        _validate_with_json_schema(validator, canonical_arguments)
        return canonical_arguments

    return validate


def json_schema_output(
    schema: Mapping[str, JsonValue] | None,
    value: JsonValue,
) -> JsonValue:
    """Canonicalize output and, when declared, validate its complete schema."""

    canonical_output = _canonical_json(value, subject="output")
    if schema is not None:
        _validate_with_json_schema(_json_schema_validator(schema), canonical_output)
    return canonical_output


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
        if not isinstance(self.cancellation_mode, CancellationMode):
            raise TypeError("cancellation_mode must be a CancellationMode")
        if not isinstance(self.interrupt_behavior, InterruptBehavior):
            raise TypeError("interrupt_behavior must be an InterruptBehavior")
        _require_exact_bool(self.idempotent, field_name="idempotent")
        _require_exact_bool(self.concurrency_safe, field_name="concurrency_safe")
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
        if isinstance(self.timeout_seconds, bool) or not isinstance(
            self.timeout_seconds,
            Real,
        ):
            raise TypeError("timeout_seconds must be a real number")
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

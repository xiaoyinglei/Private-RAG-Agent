from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, Literal

import pytest
from pydantic import BaseModel, ConfigDict, Field

from rag.agent.tools.tool import (
    JsonValue,
    ToolValidationError,
    json_schema_input,
    json_schema_output,
    pydantic_input,
)


class BuiltinArguments(BaseModel):
    count: int = Field(ge=5, le=10)
    mode: Literal["fast", "safe"]


class ExtensibleArguments(BaseModel):
    model_config = ConfigDict(extra="allow")

    query: str


class AliasedArguments(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    internal_name: str = Field(alias="externalName")


class NestedArguments(BaseModel):
    label: str


class ParentArguments(BaseModel):
    nested: NestedArguments


class ExtensibleNestedArguments(BaseModel):
    model_config = ConfigDict(extra="allow")

    label: str


class ParentWithExtensibleNestedArguments(BaseModel):
    nested: ExtensibleNestedArguments


class ExtensibleParentWithClosedNestedArguments(BaseModel):
    model_config = ConfigDict(extra="allow")

    nested: NestedArguments


def _raw_input_validator() -> Callable[
    [Mapping[str, JsonValue]], Mapping[str, JsonValue]
]:
    return json_schema_input(
        {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "type": "object",
            "properties": {
                "count": {"type": "integer", "minimum": 5, "maximum": 10},
                "mode": {"enum": ["fast", "safe"]},
            },
            "required": ["count", "mode"],
            "additionalProperties": False,
        }
    )


def test_pydantic_input_returns_closed_schema_and_canonical_arguments() -> None:
    schema, validate = pydantic_input(BuiltinArguments)
    original = {"count": 7, "mode": "fast"}

    arguments = validate(original)
    original["count"] = 8

    assert schema["additionalProperties"] is False
    assert schema["required"] == ("count", "mode")
    assert arguments == {"count": 7, "mode": "fast"}
    with pytest.raises(TypeError):
        arguments["count"] = 9  # type: ignore[index]


@pytest.mark.parametrize(
    ("arguments", "path"),
    [
        pytest.param({"mode": "fast"}, "$.count", id="required"),
        pytest.param({"count": 4, "mode": "fast"}, "$.count", id="minimum"),
        pytest.param({"count": 11, "mode": "fast"}, "$.count", id="maximum"),
        pytest.param({"count": 7, "mode": "turbo"}, "$.mode", id="enum"),
        pytest.param(
            {"count": 7, "mode": "fast", "discarded": True},
            "$.discarded",
            id="extra-is-not-silently-dropped",
        ),
    ],
)
def test_pydantic_input_rejects_invalid_arguments(
    arguments: dict[str, Any],
    path: str,
) -> None:
    _, validate = pydantic_input(BuiltinArguments)

    with pytest.raises(ToolValidationError) as error:
        validate(arguments)

    assert error.value.path == path
    assert len(error.value.message) <= 512


def test_pydantic_input_preserves_extras_only_when_model_explicitly_allows_them() -> None:
    schema, validate = pydantic_input(ExtensibleArguments)

    arguments = validate({"query": "status", "limit": 3})

    assert schema["additionalProperties"] is True
    assert arguments == {"query": "status", "limit": 3}


def test_pydantic_input_returns_coerced_canonical_arguments() -> None:
    _, validate = pydantic_input(BuiltinArguments)

    arguments = validate({"count": "7", "mode": "fast"})  # type: ignore[dict-item]

    assert arguments == {"count": 7, "mode": "fast"}


@pytest.mark.parametrize(
    "arguments",
    [
        pytest.param({"externalName": "status"}, id="schema-alias"),
        pytest.param({"internal_name": "status"}, id="python-field-name"),
    ],
)
def test_pydantic_input_accepts_aliases_and_dumps_canonical_field_names(
    arguments: dict[str, str],
) -> None:
    _, validate = pydantic_input(AliasedArguments)

    assert validate(arguments) == {"internal_name": "status"}


def test_pydantic_input_rejects_nested_unknown_arguments() -> None:
    schema, validate = pydantic_input(ParentArguments)

    nested_schema = schema["$defs"]["NestedArguments"]  # type: ignore[index]
    assert nested_schema["additionalProperties"] is False  # type: ignore[index]
    with pytest.raises(ToolValidationError) as error:
        validate({"nested": {"label": "ready", "discarded": True}})  # type: ignore[dict-item]

    assert error.value.path == "$.nested.discarded"


def test_pydantic_input_preserves_explicitly_allowed_nested_extras() -> None:
    schema, validate = pydantic_input(ParentWithExtensibleNestedArguments)

    nested_schema = schema["$defs"]["ExtensibleNestedArguments"]  # type: ignore[index]
    assert nested_schema["additionalProperties"] is True  # type: ignore[index]
    assert validate({"nested": {"label": "ready", "detail": "kept"}}) == {  # type: ignore[dict-item]
        "nested": {"label": "ready", "detail": "kept"}
    }


def test_pydantic_input_rejects_ignored_nested_extras_under_extensible_parent() -> None:
    schema, validate = pydantic_input(ExtensibleParentWithClosedNestedArguments)

    nested_schema = schema["$defs"]["NestedArguments"]  # type: ignore[index]
    assert nested_schema["additionalProperties"] is False  # type: ignore[index]
    with pytest.raises(ToolValidationError) as error:
        validate(
            {
                "nested": {"label": "ready", "discarded": True},
                "root_detail": "kept",
            }  # type: ignore[dict-item]
        )

    assert error.value.path == "$.nested.discarded"


@pytest.mark.parametrize(
    ("arguments", "path"),
    [
        pytest.param({"mode": "fast"}, "$.count", id="required"),
        pytest.param({"count": 4, "mode": "fast"}, "$.count", id="minimum"),
        pytest.param({"count": 11, "mode": "fast"}, "$.count", id="maximum"),
        pytest.param({"count": 7, "mode": "turbo"}, "$.mode", id="enum"),
        pytest.param(
            {"count": 7, "mode": "fast", "discarded": True},
            "$.discarded",
            id="additional-properties",
        ),
    ],
)
def test_raw_json_schema_rejects_complete_constraint_failures(
    arguments: dict[str, Any],
    path: str,
) -> None:
    validate = _raw_input_validator()

    with pytest.raises(ToolValidationError) as error:
        validate(arguments)

    assert error.value.path == path


def test_raw_json_schema_accepts_and_freezes_complete_valid_arguments() -> None:
    closed_schema = {
        "type": "object",
        "properties": {
            "items": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["items"],
        "additionalProperties": False,
    }
    validate = json_schema_input(closed_schema)
    source = {"items": ["one"]}
    arguments = validate(source)  # type: ignore[arg-type]
    source["items"].append("mutated")

    assert arguments == {"items": ("one",)}
    with pytest.raises(TypeError):
        arguments["items"] = ()  # type: ignore[index]


@pytest.mark.parametrize(
    ("keyword", "valid", "invalid"),
    [
        pytest.param(
            "oneOf",
            {"value": 7},
            {"value": True},
            id="oneOf",
        ),
        pytest.param(
            "anyOf",
            {"value": "ready"},
            {"value": False},
            id="anyOf",
        ),
    ],
)
def test_raw_json_schema_preserves_composition_keywords(
    keyword: str,
    valid: dict[str, Any],
    invalid: dict[str, Any],
) -> None:
    schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "value": {
                keyword: [
                    {"type": "integer"},
                    {"type": "string"},
                ]
            }
        },
        "required": ["value"],
    }
    validate = json_schema_input(schema)

    assert validate(valid) == valid
    with pytest.raises(ToolValidationError, match=keyword):
        validate(invalid)


def test_raw_json_schema_resolves_only_local_defs() -> None:
    schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$defs": {
            "positive": {"type": "integer", "minimum": 1},
        },
        "type": "object",
        "properties": {"count": {"$ref": "#/$defs/positive"}},
        "required": ["count"],
    }
    validate = json_schema_input(schema)

    assert validate({"count": 1}) == {"count": 1}
    with pytest.raises(ToolValidationError) as error:
        validate({"count": 0})
    assert error.value.path == "$.count"


@pytest.mark.parametrize(
    "reference",
    [
        "https://example.invalid/external-schema.json",
        "file:///tmp/external-schema.json",
        "custom://schema/argument",
    ],
)
def test_raw_json_schema_never_retrieves_external_references(reference: str) -> None:
    validate = json_schema_input({"$ref": reference})

    with pytest.raises(ToolValidationError) as error:
        validate({})

    assert error.value.path == "$"
    assert error.value.message == "schema reference could not be resolved locally"
    assert reference not in str(error.value)


def test_raw_json_schema_respects_declared_dialect() -> None:
    validate = json_schema_input(
        {
            "$schema": "http://json-schema.org/draft-04/schema#",
            "type": "object",
            "properties": {
                "value": {
                    "type": "number",
                    "minimum": 5,
                    "exclusiveMinimum": True,
                }
            },
            "required": ["value"],
        }
    )

    with pytest.raises(ToolValidationError):
        validate({"value": 5})
    assert validate({"value": 6}) == {"value": 6}


def test_raw_json_schema_is_copied_at_factory_creation() -> None:
    schema: dict[str, Any] = {"type": "object", "required": ["count"]}
    validate = json_schema_input(schema)
    schema["required"].clear()

    with pytest.raises(ToolValidationError):
        validate({})


def test_invalid_json_schema_is_rejected_at_factory_creation() -> None:
    with pytest.raises(ToolValidationError) as error:
        json_schema_input({"type": "not-a-json-schema-type"})

    assert error.value.path == "$.type"
    assert len(error.value.message) <= 512


def test_json_schema_output_validates_and_freezes_output() -> None:
    schema = {
        "type": "object",
        "properties": {"status": {"enum": ["ready"]}},
        "required": ["status"],
        "additionalProperties": False,
    }
    source = {"status": "ready"}

    output = json_schema_output(schema, source)  # type: ignore[arg-type]
    source["status"] = "mutated"

    assert output == {"status": "ready"}
    with pytest.raises(ToolValidationError) as error:
        json_schema_output(schema, {"status": "failed"})  # type: ignore[arg-type]
    assert error.value.path == "$.status"


def test_json_schema_output_without_schema_still_freezes_json() -> None:
    source = {"items": ["one"]}

    output = json_schema_output(None, source)  # type: ignore[arg-type]
    source["items"].append("mutated")

    assert output == {"items": ("one",)}


def test_validation_errors_bound_path_and_message() -> None:
    enormous_name = "x" * 2_000
    validate = json_schema_input(
        {
            "type": "object",
            "properties": {enormous_name: {"enum": [7]}},
            "required": [enormous_name],
        }
    )

    with pytest.raises(ToolValidationError) as error:
        validate({enormous_name: 1})

    assert len(error.value.path) <= 256
    assert len(error.value.message) <= 512
    assert enormous_name not in str(error.value)


def test_regression_complete_schema_rejects_before_runner_could_execute() -> None:
    validate = json_schema_input(
        {
            "type": "object",
            "properties": {
                "count": {"type": "integer", "minimum": 5, "enum": [7]},
            },
            "required": ["count"],
        }
    )
    runner_called = False

    with pytest.raises(ToolValidationError):
        arguments = validate({"count": 1})
        runner_called = True
        assert arguments

    assert runner_called is False

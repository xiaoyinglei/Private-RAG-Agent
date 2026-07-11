from __future__ import annotations

from collections.abc import Callable
from dataclasses import FrozenInstanceError, fields, replace

import pytest

from rag.agent.tools.tool import (
    ArtifactReference,
    CancellationMode,
    InterruptBehavior,
    Tool,
    ToolCall,
    ToolCallOrigin,
    ToolContentBlock,
    ToolDefinition,
    ToolEffect,
    ToolResult,
)


def _normalize_output(output: object) -> ToolResult:
    return ToolResult(
        tool_call_id="fixture-call",
        tool_name="read_text",
        structured_content={"output": str(output)},
    )


def _origin() -> ToolCallOrigin:
    return ToolCallOrigin(
        request_id="request-1",
        toolset_revision="toolset-v3",
        exposed_tool_names=("read_text",),
    )


def _tool(**changes: object) -> Tool:
    tool = Tool(
        definition=ToolDefinition(
            name="read_text",
            description="Read text from the workspace.",
            input_schema={
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        ),
        validate_input=lambda arguments: arguments,
        run=lambda arguments: {"text": arguments["path"]},
        normalize_output=_normalize_output,
        output_schema={
            "type": "object",
            "properties": {"text": {"type": "string"}},
        },
        static_effects=frozenset({ToolEffect.READ_WORKSPACE}),
        resolve_use=lambda _arguments: frozenset(),
        execution_revision="read-text-v1",
        idempotent=True,
        concurrency_safe=True,
        cancellation_mode=CancellationMode.COOPERATIVE,
        interrupt_behavior=InterruptBehavior.CANCEL,
        timeout_seconds=10.0,
        max_model_output_bytes=4096,
    )
    return replace(tool, **changes)


def test_tool_projects_definition_without_runner() -> None:
    schema = {"type": "object", "required": ["path"]}
    tool = _tool(
        definition=ToolDefinition(
            name="read_text",
            description="Read text from the workspace.",
            input_schema=schema,
        )
    )

    schema["required"].append("mutated")

    assert tool.definition.name == "read_text"
    assert tool.definition.input_schema["required"] == ("path",)
    assert not hasattr(tool.definition, "run")
    with pytest.raises(FrozenInstanceError):
        tool.definition.name = "other"  # type: ignore[misc]


def test_tool_rejects_local_side_effect_with_non_cancellable_mode() -> None:
    with pytest.raises(ValueError, match="local side-effecting"):
        _tool(
            static_effects=frozenset({ToolEffect.WRITE_WORKSPACE}),
            cancellation_mode=CancellationMode.NOT_CANCELLABLE,
            interrupt_behavior=InterruptBehavior.FINISH_CURRENT,
        )


def test_tool_call_origin_preserves_checkpoint_evidence() -> None:
    origin = ToolCallOrigin(
        request_id="request-1",
        toolset_revision="toolset-v3",
        exposed_tool_names=("read_text", "list_files"),
    )
    call = ToolCall(
        tool_call_id="call-1",
        tool_name="read_text",
        arguments={"path": "notes.txt"},
        origin=origin,
    )

    assert tuple(field.name for field in fields(origin)) == (
        "request_id",
        "toolset_revision",
        "exposed_tool_names",
    )
    assert call.origin == origin
    with pytest.raises(FrozenInstanceError):
        origin.request_id = "other"  # type: ignore[misc]


@pytest.mark.parametrize("field_name", ["request_id", "toolset_revision"])
def test_tool_call_origin_rejects_mutable_scalar_evidence(field_name: str) -> None:
    values: dict[str, object] = {
        "request_id": "request-1",
        "toolset_revision": "toolset-v3",
        "exposed_tool_names": (),
    }
    values[field_name] = ["mutable"]

    with pytest.raises(TypeError, match=field_name):
        ToolCallOrigin(**values)  # type: ignore[arg-type]


def test_tool_call_origin_freezes_exposed_names_without_splitting_strings() -> None:
    exposed_names = ["read_text"]
    origin = ToolCallOrigin(
        request_id="request-1",
        toolset_revision="toolset-v3",
        exposed_tool_names=exposed_names,  # type: ignore[arg-type]
    )

    exposed_names.append("mutated")

    assert origin.exposed_tool_names == ("read_text",)
    with pytest.raises(TypeError, match="exposed_tool_names"):
        ToolCallOrigin(
            request_id="request-1",
            toolset_revision="toolset-v3",
            exposed_tool_names="read_text",  # type: ignore[arg-type]
        )


def test_tool_call_rejects_non_origin_record() -> None:
    with pytest.raises(TypeError, match="origin"):
        ToolCall(
            tool_call_id="call-1",
            tool_name="read_text",
            arguments={"path": "notes.txt"},
            origin={"request_id": "request-1"},  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("field_name", ["artifact_id", "media_type", "name"])
def test_artifact_reference_rejects_mutable_string_fields(field_name: str) -> None:
    values: dict[str, object] = {
        "artifact_id": "artifact-1",
        "media_type": "text/plain",
        "name": "result.txt",
    }
    values[field_name] = ["mutable"]

    with pytest.raises(TypeError, match=field_name):
        ArtifactReference(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "factory",
    [
        pytest.param(
            lambda: ToolDefinition(
                name=["read_text"],  # type: ignore[arg-type]
                description="Read text.",
                input_schema={},
            ),
            id="definition-name",
        ),
        pytest.param(
            lambda: ToolDefinition(
                name="read_text",
                description=["Read text."],  # type: ignore[arg-type]
                input_schema={},
            ),
            id="definition-description",
        ),
        pytest.param(
            lambda: ToolCall(
                tool_call_id=["call-1"],  # type: ignore[arg-type]
                tool_name="read_text",
                arguments={},
                origin=_origin(),
            ),
            id="call-id",
        ),
        pytest.param(
            lambda: ToolCall(
                tool_call_id="call-1",
                tool_name=["read_text"],  # type: ignore[arg-type]
                arguments={},
                origin=_origin(),
            ),
            id="call-name",
        ),
        pytest.param(
            lambda: ToolResult(
                tool_call_id=["call-1"],  # type: ignore[arg-type]
                tool_name="read_text",
            ),
            id="result-id",
        ),
        pytest.param(
            lambda: ToolResult(
                tool_call_id="call-1",
                tool_name=["read_text"],  # type: ignore[arg-type]
            ),
            id="result-name",
        ),
        pytest.param(
            lambda: _tool(execution_revision=["read-text-v1"]),
            id="execution-revision",
        ),
    ],
)
def test_required_contract_strings_reject_non_string_values(
    factory: Callable[[], object],
) -> None:
    with pytest.raises(TypeError):
        factory()


def test_tool_content_blocks_support_extensible_json_payloads() -> None:
    blocks = (
        ToolContentBlock(type="text", data={"text": "first"}),
        ToolContentBlock(type="image", data={"artifact_id": "image-1"}),
        ToolContentBlock(type="resource", data={"uri": "artifact://result-1"}),
    )

    assert [block.type for block in blocks] == ["text", "image", "resource"]
    assert blocks[1].data == {"artifact_id": "image-1"}
    with pytest.raises(TypeError, match="JSON-compatible"):
        ToolContentBlock(type="image", data={"value": object()})


def test_tool_result_metadata_is_not_model_content() -> None:
    result = ToolResult(
        tool_call_id="call-1",
        tool_name="read_text",
        content=(
            ToolContentBlock(type="text", data={"text": "first"}),
            ToolContentBlock(type="text", data={"text": "second"}),
        ),
        structured_content={"text": "first\nsecond"},
        is_error=False,
        error_code=None,
        error_message=None,
        truncated=False,
        metadata={"trace_id": "runtime-only"},
        attachments=(
            ArtifactReference(
                artifact_id="artifact-1",
                media_type="text/plain",
                name="result.txt",
            ),
        ),
    )

    assert [block.data["text"] for block in result.content] == ["first", "second"]
    assert result.metadata == {"trace_id": "runtime-only"}
    assert all("runtime-only" not in repr(block.data) for block in result.content)
    assert not hasattr(result, "model_content")
    assert result.attachments[0].artifact_id == "artifact-1"

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace

import pytest

from rag.agent.tools.tool import (
    ArtifactReference,
    CancellationMode,
    InterruptBehavior,
    Tool,
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


def test_tool_result_metadata_is_not_model_content() -> None:
    result = ToolResult(
        tool_call_id="call-1",
        tool_name="read_text",
        model_content=(
            ToolContentBlock(type="text", text="first"),
            ToolContentBlock(type="text", text="second"),
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

    assert [block.text for block in result.model_content] == ["first", "second"]
    assert result.metadata == {"trace_id": "runtime-only"}
    assert all("runtime-only" not in (block.text or "") for block in result.model_content)
    assert result.attachments[0].artifact_id == "artifact-1"

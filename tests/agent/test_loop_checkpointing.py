from __future__ import annotations

from collections.abc import Mapping

import pytest
from langgraph.checkpoint.memory import MemorySaver

from rag.agent.core.checkpointing import (
    LangGraphCheckpointStore,
    agent_checkpoint_serde,
)
from rag.agent.core.context import AgentRunConfig
from rag.agent.core.human_input import (
    HumanInputRequest,
    HumanInputRequestIdMismatchError,
    HumanInputResponse,
)
from rag.agent.core.messages import ModelMessage
from rag.agent.core.messages import ToolCall as ModelToolCall
from rag.agent.core.model_request import build_tool_manifest
from rag.agent.core.turn_contracts import ToolCallPlan
from rag.agent.loop.state import LoopPause, create_loop_state
from rag.agent.tools.executor import ExecutionStatus, ToolExecutionRecord
from rag.agent.tools.tool import (
    CancellationMode,
    InterruptBehavior,
    JsonValue,
    NormalizedToolOutput,
    ResolvedToolUse,
    Tool,
    ToolCall,
    ToolCallOrigin,
    ToolDefinition,
    json_schema_input,
)
from rag.schema.runtime import AccessPolicy


def _config(run_id: str) -> AgentRunConfig:
    return AgentRunConfig(
        run_id=run_id,
        thread_id=run_id,
        max_depth=1,
        access_policy=AccessPolicy.default(),
    )


def _tool(*, execution_revision: str = "write-v1") -> Tool:
    schema: Mapping[str, JsonValue] = {
        "type": "object",
        "properties": {"value": {"type": "string"}},
        "additionalProperties": False,
    }
    return Tool(
        definition=ToolDefinition(
            name="remote_write",
            description="Write remotely.",
            input_schema=schema,
        ),
        validate_input=json_schema_input(schema),
        run=lambda _arguments: {},
        normalize_output=lambda _raw: NormalizedToolOutput(),
        output_schema=None,
        static_effects=frozenset(),
        resolve_use=lambda _arguments: ResolvedToolUse(
            effects=frozenset(),
            targets=(),
        ),
        execution_revision=execution_revision,
        idempotent=False,
        concurrency_safe=False,
        cancellation_mode=CancellationMode.REMOTE_BEST_EFFORT,
        interrupt_behavior=InterruptBehavior.CANCEL,
        timeout_seconds=1.0,
        max_model_output_bytes=4096,
    )


def _state(run_id: str):
    tool = _tool()
    origin = ToolCallOrigin(
        request_id="request-a",
        toolset_revision="tools-a",
        exposed_tool_names=("remote_write",),
    )
    call = ToolCall(
        tool_call_id="call-a",
        tool_name="remote_write",
        arguments={"value": "x"},
        origin=origin,
    )
    state = create_loop_state(
        task="Write.",
        run_config=_config(run_id),
        pending_tool_calls=(
            ToolCallPlan(
                tool_call_id=call.tool_call_id,
                tool_name=call.tool_name,
                arguments=dict(call.arguments),
                origin=origin,
            ),
        ),
    )
    state["resident_tool_names"] = ["remote_write"]
    state["canonical_transcript"] = [
        ModelMessage(role="user", content="Write.")
    ]
    state["canonical_tool_calls"] = {call.tool_call_id: call}
    state["tool_manifest"] = build_tool_manifest(
        tools=(tool,),
        resident_tool_names=("remote_write",),
        provider_serializer_revision="provider-v1",
    )
    state["context_revision"] = "context-a"
    state["prompt_revision"] = "prompt-a"
    state["provider_serializer_revision"] = "provider-v1"
    return state, call


@pytest.mark.anyio
async def test_save_load_uses_canonical_codec_and_origin() -> None:
    state, call = _state("checkpoint-roundtrip")
    store = LangGraphCheckpointStore(
        MemorySaver(serde=agent_checkpoint_serde()),
        run_config=state["run_config"],
    )

    await store.save_snapshot(state, reason="scheduled")
    restored = await store.load_latest()

    assert restored is not None
    assert restored["tool_checkpoint"]["format_version"] == 2
    assert restored["canonical_tool_calls"][call.tool_call_id].origin == (
        call.origin
    )
    assert restored["canonical_transcript"] == state["canonical_transcript"]


@pytest.mark.anyio
async def test_save_load_deepcopies_assistant_tool_call_arguments() -> None:
    state, _call = _state("checkpoint-transcript-tool-call")
    state["canonical_transcript"] = [
        ModelMessage(
            role="assistant",
            content="",
            tool_calls=(
                ModelToolCall(
                    id="call-a",
                    name="remote_write",
                    input={"value": "x"},
                ),
            ),
        )
    ]
    store = LangGraphCheckpointStore(
        MemorySaver(serde=agent_checkpoint_serde()),
        run_config=state["run_config"],
    )

    await store.save_snapshot(state, reason="tool_call")
    restored = await store.load_latest()

    assert restored is not None
    assert restored["canonical_transcript"] == state["canonical_transcript"]


@pytest.mark.anyio
async def test_apply_tool_approval_returns_running_state() -> None:
    state, call = _state("checkpoint-approval")
    request = HumanInputRequest(
        request_id="approval-a",
        kind="tool_approval",
        question="Allow?",
    )
    state["status"] = "paused"
    state["approval_request"] = request
    state["pause"] = LoopPause(reason="approval", request=request)
    store = LangGraphCheckpointStore(
        MemorySaver(serde=agent_checkpoint_serde()),
        run_config=state["run_config"],
    )
    await store.save_snapshot(state, reason="approval")

    restored = await store.apply_human_response(
        HumanInputResponse(
            request_id=request.request_id,
            decision="allow_once",
            approved_tool_call_ids=[call.tool_call_id],
        )
    )

    assert restored["status"] == "running"
    assert restored["approved_tool_call_ids"] == [call.tool_call_id]
    assert restored["pause"] is None


@pytest.mark.anyio
async def test_apply_human_response_rejects_wrong_request_id() -> None:
    state, _call = _state("checkpoint-request-id")
    request = HumanInputRequest(
        request_id="approval-a",
        kind="tool_approval",
        question="Allow?",
    )
    state["status"] = "paused"
    state["approval_request"] = request
    state["pause"] = LoopPause(reason="approval", request=request)
    store = LangGraphCheckpointStore(
        MemorySaver(serde=agent_checkpoint_serde()),
        run_config=state["run_config"],
    )
    await store.save_snapshot(state, reason="approval")

    with pytest.raises(HumanInputRequestIdMismatchError):
        await store.apply_human_response(
            HumanInputResponse(request_id="wrong", decision="deny")
        )


@pytest.mark.anyio
async def test_running_non_idempotent_record_requires_reconciliation() -> None:
    state, call = _state("checkpoint-ambiguous")
    state["tool_execution_records"][call.tool_call_id] = ToolExecutionRecord(
        tool_call_id=call.tool_call_id,
        tool_name=call.tool_name,
        operation_id="operation-a",
        arguments_digest="digest-a",
        idempotent=False,
        status=ExecutionStatus.RUNNING,
        attempt_count=1,
    )
    store = LangGraphCheckpointStore(
        MemorySaver(serde=agent_checkpoint_serde()),
        run_config=state["run_config"],
    )
    await store.save_snapshot(state, reason="running")

    restored = await store.load_for_resume()

    assert restored is not None
    assert restored["status"] == "paused"
    assert restored["approval_request"] is not None
    assert restored["approval_request"].kind == "tool_reconciliation"
    assert (
        restored["tool_execution_records"][call.tool_call_id].status
        is ExecutionStatus.UNKNOWN
    )

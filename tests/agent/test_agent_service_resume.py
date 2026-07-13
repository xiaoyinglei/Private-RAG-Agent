from __future__ import annotations

from collections.abc import Mapping

import pytest
from langgraph.checkpoint.memory import MemorySaver

from rag.agent.core.checkpointing import agent_checkpoint_serde
from rag.agent.core.definition import AgentRuntimePolicy
from rag.agent.core.human_input import (
    HumanInputRequestIdMismatchError,
    HumanInputResponse,
)
from rag.agent.core.turn_contracts import ToolCallPlan
from rag.agent.loop.state import LoopState, ModelTurnDraft
from rag.agent.service import AgentRunRequest, AgentService
from rag.agent.tools.registry import ToolRegistry
from rag.agent.tools.tool import (
    CancellationMode,
    InterruptBehavior,
    JsonValue,
    NormalizedToolOutput,
    ResolvedToolUse,
    Tool,
    ToolDefinition,
    ToolEffect,
    json_schema_input,
)


class _FinishProvider:
    async def next_turn(
        self,
        state: LoopState,
        *,
        definition: AgentRuntimePolicy,
        budget_remaining: int,
    ) -> ModelTurnDraft:
        del definition, budget_remaining
        if state["tool_results"]:
            payload = state["tool_results"][-1].structured_content
            if isinstance(payload, Mapping):
                text = payload.get("text")
                if isinstance(text, str):
                    return ModelTurnDraft(
                        action="finish",
                        final_answer=text,
                    )
        return ModelTurnDraft(action="finish", final_answer="done")


def _tool(
    calls: list[str],
    *,
    execution_revision: str = "remote-v1",
) -> Tool:
    schema: Mapping[str, JsonValue] = {
        "type": "object",
        "properties": {"value": {"type": "string"}},
        "required": ["value"],
        "additionalProperties": False,
    }

    def run(arguments: Mapping[str, JsonValue]) -> object:
        value = str(arguments["value"])
        calls.append(value)
        return {"text": f"ran:{value}"}

    return Tool(
        definition=ToolDefinition(
            name="remote_write",
            description="Perform one remote write.",
            input_schema=schema,
        ),
        validate_input=json_schema_input(schema),
        run=run,
        normalize_output=lambda raw: NormalizedToolOutput(
            structured_content=raw  # type: ignore[arg-type]
        ),
        output_schema=None,
        static_effects=frozenset({ToolEffect.NETWORK}),
        resolve_use=lambda _arguments: ResolvedToolUse(
            effects=frozenset({ToolEffect.NETWORK}),
            targets=(),
        ),
        execution_revision=execution_revision,
        idempotent=False,
        concurrency_safe=False,
        cancellation_mode=CancellationMode.COOPERATIVE,
        interrupt_behavior=InterruptBehavior.CANCEL,
        timeout_seconds=1.0,
        max_model_output_bytes=4096,
    )


def _definition() -> AgentRuntimePolicy:
    return AgentRuntimePolicy.test_factory(
        system_prompt="Use the remote tool.",
        allowed_tools=["remote_write"],
        max_iterations=4,
    )


def _service(
    calls: list[str],
    *,
    checkpointer: MemorySaver | None = None,
    execution_revision: str = "remote-v1",
) -> AgentService:
    registry = ToolRegistry()
    registry.register(
        _tool(calls, execution_revision=execution_revision)
    )
    return AgentService(
        definition=_definition(),
        tool_registry=registry,
        model_turn_provider=_FinishProvider(),
        checkpointer=checkpointer,
    )


async def _pause(
    service: AgentService,
    *,
    run_id: str,
    value: str,
):
    call = ToolCallPlan.create("remote_write", {"value": value})
    result = await service.run(
        AgentRunRequest(
            task="Run one remote write.",
            run_id=run_id,
            thread_id=run_id,
            pending_tool_calls=[call],
        )
    )
    assert result.status == "paused"
    assert result.human_input_request is not None
    return call, result


@pytest.mark.anyio
async def test_default_checkpointer_resumes_on_same_service_without_replay() -> None:
    calls: list[str] = []
    service = _service(calls)
    call, paused = await _pause(
        service,
        run_id="resume-default",
        value="once",
    )

    resumed = await service.resume(
        run_id="resume-default",
        response=HumanInputResponse(
            request_id=paused.human_input_request.request_id,
            decision="allow_once",
            approved_tool_call_ids=[call.tool_call_id],
        ),
    )

    assert resumed.status == "done"
    assert resumed.final_answer == "ran:once"
    assert calls == ["once"]
    assert resumed.tool_results[0].is_error is False


@pytest.mark.anyio
async def test_resume_uses_same_codec_across_service_boundary() -> None:
    checkpointer = MemorySaver(serde=agent_checkpoint_serde())
    calls: list[str] = []
    first = _service(calls, checkpointer=checkpointer)
    call, paused = await _pause(
        first,
        run_id="resume-cross-service",
        value="persisted",
    )

    second = _service(calls, checkpointer=checkpointer)
    pending = second.pending_human_input_request(
        run_id="resume-cross-service"
    )
    assert pending.request_id == paused.human_input_request.request_id
    resumed = await second.resume(
        run_id="resume-cross-service",
        response=HumanInputResponse(
            request_id=pending.request_id,
            decision="allow_once",
            approved_tool_call_ids=[call.tool_call_id],
        ),
    )

    assert resumed.status == "done"
    assert calls == ["persisted"]


@pytest.mark.anyio
async def test_changed_pending_tool_definition_requires_reconciliation() -> None:
    checkpointer = MemorySaver(serde=agent_checkpoint_serde())
    calls: list[str] = []
    first = _service(
        calls,
        checkpointer=checkpointer,
        execution_revision="remote-v1",
    )
    _call, paused = await _pause(
        first,
        run_id="resume-drift",
        value="drifted",
    )
    second = _service(
        calls,
        checkpointer=checkpointer,
        execution_revision="remote-v2",
    )

    result = await second.resume(
        run_id="resume-drift",
        response=HumanInputResponse(
            request_id=paused.human_input_request.request_id,
            decision="allow_once",
        ),
    )

    assert result.status == "paused"
    assert result.needs_user_input == "tool_definition_changed"
    assert result.human_input_request.kind == "tool_reconciliation"
    assert calls == []


@pytest.mark.anyio
async def test_resume_rejects_wrong_human_request_id() -> None:
    service = _service([])
    _call, _paused = await _pause(
        service,
        run_id="resume-request-id",
        value="x",
    )

    with pytest.raises(HumanInputRequestIdMismatchError):
        await service.resume(
            run_id="resume-request-id",
            response=HumanInputResponse(
                request_id="wrong",
                decision="deny",
            ),
        )

from __future__ import annotations

import pytest
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel

from rag.agent.core.context import RuntimeRegistry
from rag.agent.core.definition import AgentDefinition
from rag.agent.core.human_input import HumanInputResponse
from rag.agent.service import AgentRunRequest, AgentService
from rag.agent.state import ToolCallPlan
from rag.agent.tools.registry import ToolRegistry
from rag.agent.tools.spec import ToolError, ToolPermissions, ToolSpec


class _WriteInput(BaseModel):
    data: str


class _WriteOutput(BaseModel):
    result: str


def _write_spec() -> ToolSpec:
    return ToolSpec(
        name="write_tool",
        description="write",
        input_model=_WriteInput,
        output_model=_WriteOutput,
        error_model=ToolError,
        permissions=ToolPermissions(write_db=True),
        timeout_seconds=1.0,
    )


def _definition() -> AgentDefinition:
    return AgentDefinition(
        agent_type="resume_test",
        description="Resume test",
        system_prompt="Run tools.",
        allowed_tools=["write_tool"],
        max_iterations=3,
    )


def _service(*, checkpointer: BaseCheckpointSaver, calls: list[str]) -> AgentService:
    registry = ToolRegistry()

    def runner(payload: _WriteInput) -> _WriteOutput:
        calls.append(payload.data)
        return _WriteOutput(result=f"wrote:{payload.data}")

    registry.register(_write_spec(), runner=runner)
    return AgentService(
        definition=_definition(),
        tool_registry=registry,
        checkpointer=checkpointer,
    )


@pytest.mark.anyio
async def test_resume_restores_runtime_handles_from_checkpoint_after_process_boundary() -> None:
    checkpointer = MemorySaver()
    calls: list[str] = []
    service = _service(checkpointer=checkpointer, calls=calls)
    call = ToolCallPlan.create("write_tool", {"data": "persisted"})

    paused = await service.run(
        AgentRunRequest(
            task="run write tool",
            run_id="resume-cross-process",
            thread_id="resume-cross-process",
            pending_tool_calls=[call],
        )
    )

    assert paused.status == "paused"
    assert paused.human_input_request is not None
    RuntimeRegistry.remove("resume-cross-process")

    resumed_service = _service(checkpointer=checkpointer, calls=calls)
    request = resumed_service.pending_human_input_request(run_id="resume-cross-process")
    resumed = await resumed_service.resume(
        run_id="resume-cross-process",
        response=HumanInputResponse(
            request_id=request.request_id,
            decision="allow_once",
            approved_tool_call_ids=[call.tool_call_id],
        ),
    )

    assert resumed.status == "done"
    assert calls == ["persisted"]
    [tool_result] = resumed.tool_results
    assert tool_result.status == "ok"
    assert tool_result.output == _WriteOutput(result="wrote:persisted")
    with pytest.raises(KeyError):
        RuntimeRegistry.get("resume-cross-process")


@pytest.mark.anyio
async def test_sqlite_checkpointer_persists_paused_run_between_service_instances(tmp_path) -> None:
    from rag.agent.core.checkpointing import (
        aclose_agent_checkpointer,
        create_agent_checkpointer,
    )

    checkpoint_db = tmp_path / "agent-checkpoints.sqlite"
    calls: list[str] = []
    checkpointer = create_agent_checkpointer(checkpoint_db)
    service = _service(
        checkpointer=checkpointer,
        calls=calls,
    )
    call = ToolCallPlan.create("write_tool", {"data": "sqlite"})

    paused = await service.run(
        AgentRunRequest(
            task="run write tool",
            run_id="resume-sqlite",
            thread_id="resume-sqlite",
            pending_tool_calls=[call],
        )
    )

    assert paused.status == "paused"
    RuntimeRegistry.remove("resume-sqlite")

    await aclose_agent_checkpointer(checkpointer)
    resumed_checkpointer = create_agent_checkpointer(checkpoint_db)
    resumed_service = _service(
        checkpointer=resumed_checkpointer,
        calls=calls,
    )
    request = await resumed_service.apending_human_input_request(run_id="resume-sqlite")
    resumed = await resumed_service.resume(
        run_id="resume-sqlite",
        response=HumanInputResponse(
            request_id=request.request_id,
            decision="allow_once",
            approved_tool_call_ids=[call.tool_call_id],
        ),
    )

    assert resumed.status == "done"
    assert calls == ["sqlite"]
    await aclose_agent_checkpointer(resumed_checkpointer)

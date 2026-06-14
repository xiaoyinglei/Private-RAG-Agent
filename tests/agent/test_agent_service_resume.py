from __future__ import annotations

import pytest
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel

from rag.agent.builtin.research import RESEARCH_AGENT
from rag.agent.builtin_registry import create_builtin_tool_registry
from rag.agent.compat.goal_contract import GoalDeliverable, GoalSpec
from rag.agent.core.checkpointing import agent_checkpoint_serde
from rag.agent.core.context import RunRegistry
from rag.agent.core.definition import AgentDefinition
from rag.agent.core.human_input import HumanInputResponse
from rag.agent.loop.state import LoopState, ModelTurnDraft
from rag.agent.service import AgentRunRequest, AgentService
from rag.agent.state import ToolCallPlan
from rag.agent.tools.registry import ToolRegistry
from rag.agent.tools.spec import ToolError, ToolPermissions, ToolSpec


class _WriteInput(BaseModel):
    data: str


class _WriteOutput(BaseModel):
    result: str


class _TextGenerator:
    def generate_text(self, *, prompt: str, **kwargs: object) -> str:
        del prompt, kwargs
        return "resume summary"


class _ResolvedFakeModel:
    def __init__(self) -> None:
        self.generator = _TextGenerator()
        self.kwargs: dict[str, object] = {}


class _FakeModelRegistry:
    def resolve_for_node(self, *, node_model: str | None, node_name: str) -> _ResolvedFakeModel:
        del node_model, node_name
        return _ResolvedFakeModel()


class _FinishProvider:
    async def next_turn(
        self,
        state: LoopState,
        *,
        definition: AgentDefinition,
        budget_remaining: int,
    ) -> ModelTurnDraft:
        del definition, budget_remaining
        if state["answer_candidates"]:
            return ModelTurnDraft(
                action="finish",
                final_answer=state["answer_candidates"][-1].text,
            )
        return ModelTurnDraft(
            action="pause",
            pause_reason="No answer candidate is available.",
        )


class _PauseAfterGoalFeedbackProvider:
    async def next_turn(
        self,
        state: LoopState,
        *,
        definition: AgentDefinition,
        budget_remaining: int,
    ) -> ModelTurnDraft:
        del definition, budget_remaining
        if state["stop_hook_feedback"]:
            return ModelTurnDraft(
                action="pause",
                pause_reason="Explicit goal still needs evidence.",
            )
        return ModelTurnDraft(
            action="finish",
            final_answer="unsupported answer",
        )


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


def _service_with_default_checkpointer(*, calls: list[str]) -> AgentService:
    registry = ToolRegistry()

    def runner(payload: _WriteInput) -> _WriteOutput:
        calls.append(payload.data)
        return _WriteOutput(result=f"wrote:{payload.data}")

    registry.register(_write_spec(), runner=runner)
    return AgentService(
        definition=_definition(),
        tool_registry=registry,
    )


@pytest.mark.anyio
async def test_default_checkpointer_supports_resume_on_same_service() -> None:
    calls: list[str] = []
    service = _service_with_default_checkpointer(calls=calls)
    call = ToolCallPlan.create("write_tool", {"data": "default"})

    paused = await service.run(
        AgentRunRequest(
            task="run write tool",
            run_id="resume-default-checkpointer",
            thread_id="resume-default-checkpointer",
            pending_tool_calls=[call],
        )
    )

    assert paused.status == "paused"
    assert paused.human_input_request is not None

    resumed = await service.resume(
        run_id="resume-default-checkpointer",
        response=HumanInputResponse(
            request_id=paused.human_input_request.request_id,
            decision="allow_once",
            approved_tool_call_ids=[call.tool_call_id],
        ),
    )

    assert resumed.status == "done"
    assert calls == ["default"]


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
    RunRegistry.remove("resume-cross-process")

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
        RunRegistry.get("resume-cross-process")


@pytest.mark.anyio
async def test_resume_restores_explicit_goal_hook_after_process_boundary() -> None:
    checkpointer = MemorySaver(serde=agent_checkpoint_serde())
    calls: list[str] = []
    provider = _PauseAfterGoalFeedbackProvider()
    service = _service(checkpointer=checkpointer, calls=calls)
    service._model_turn_provider = provider
    call = ToolCallPlan.create("write_tool", {"data": "goal-bound"})
    goal = GoalSpec(
        original_query="Answer with evidence.",
        deliverables=[
            GoalDeliverable(
                deliverable_id="answer",
                kind="answer",
                acceptance_rule="non_empty_answer",
            ),
            GoalDeliverable(
                deliverable_id="evidence",
                kind="evidence",
                acceptance_rule="traceable_evidence",
            ),
        ],
    )

    paused = await service.run(
        AgentRunRequest(
            task="Answer with evidence.",
            run_id="resume-goal-hook",
            thread_id="resume-goal-hook",
            pending_tool_calls=[call],
            goal_spec=goal,
        )
    )

    assert paused.status == "paused"
    assert paused.human_input_request is not None
    RunRegistry.remove("resume-goal-hook")

    resumed_service = _service(checkpointer=checkpointer, calls=calls)
    resumed_service._model_turn_provider = provider
    resumed = await resumed_service.resume(
        run_id="resume-goal-hook",
        response=HumanInputResponse(
            request_id=paused.human_input_request.request_id,
            decision="allow_once",
            approved_tool_call_ids=[call.tool_call_id],
        ),
    )

    assert resumed.status == "paused"
    assert resumed.needs_user_input == "Explicit goal still needs evidence."
    assert calls == ["goal-bound"]


@pytest.mark.anyio
async def test_resume_preserves_model_backed_llm_tool_runners() -> None:
    service = AgentService(
        definition=RESEARCH_AGENT,
        tool_registry=create_builtin_tool_registry(runners={}),
        model_turn_provider=_FinishProvider(),
        model_registry=_FakeModelRegistry(),  # type: ignore[arg-type]
    )
    write_call = ToolCallPlan.create(
        "write_file",
        {"path": "scratch/note.txt", "content": "ok", "overwrite": True},
    )
    summarize_call = ToolCallPlan.create(
        "llm_summarize",
        {"task": "Summarize note", "context_sections": ["note written"]},
    )

    paused = await service.run(
        AgentRunRequest(
            task="write and summarize",
            run_id="resume-model-llm-tools",
            thread_id="resume-model-llm-tools",
            pending_tool_calls=[write_call, summarize_call],
        )
    )

    assert paused.status == "paused"
    assert paused.human_input_request is not None
    resumed = await service.resume(
        run_id="resume-model-llm-tools",
        workspace_path=paused.workspace_path,
        response=HumanInputResponse(
            request_id=paused.human_input_request.request_id,
            decision="allow_once",
            approved_tool_call_ids=[write_call.tool_call_id],
        ),
    )

    assert resumed.status == "done"
    assert resumed.final_answer == "resume summary"
    assert [result.tool_name for result in resumed.tool_results] == [
        "write_file",
        "llm_summarize",
    ]


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
    RunRegistry.remove("resume-sqlite")

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

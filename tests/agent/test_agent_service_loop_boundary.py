from __future__ import annotations

from typing import cast

import pytest
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel

from rag.agent.core.agent_service_factory import AgentServiceFactory
from rag.agent.core.checkpointing import (
    LangGraphCheckpointStore,
    agent_checkpoint_serde,
)
from rag.agent.core.context import AgentRunConfig, RunRegistry
from rag.agent.core.definition import AgentDefinition
from rag.agent.core.human_input import HumanInputResponse
from rag.agent.core.tool_execution import (
    ToolExecutionRecord,
    tool_arguments_digest,
)
from rag.agent.goal_runtime import GoalDeliverable, GoalSpec
from rag.agent.loop.state import LoopState, ModelTurnDraft, create_loop_state
from rag.agent.service import AgentRunRequest, AgentService
from rag.agent.state import ToolCallPlan
from rag.agent.tools.registry import ToolRegistry
from rag.agent.tools.spec import ToolError, ToolPermissions, ToolSpec
from rag.schema.runtime import AccessPolicy


class _Input(BaseModel):
    value: str


class _Output(BaseModel):
    text: str


class _FinishFromResultsProvider:
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
            action="finish",
            final_answer="direct answer",
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


class _ExplodingGoalProvider:
    def infer(self, state: object) -> object:
        del state
        raise AssertionError("goal controller must not run")


def _definition(*, requires_confirmation: bool = False) -> AgentDefinition:
    del requires_confirmation
    return AgentDefinition(
        agent_type="service_loop",
        description="Service loop boundary",
        system_prompt="Use the loop.",
        allowed_tools=["write_tool"],
        max_iterations=4,
    )


def _registry(
    calls: list[str],
    *,
    requires_confirmation: bool = False,
) -> ToolRegistry:
    registry = ToolRegistry()

    def runner(payload: _Input) -> _Output:
        calls.append(payload.value)
        return _Output(text=f"wrote:{payload.value}")

    registry.register(
        ToolSpec(
            name="write_tool",
            description="Write once.",
            input_model=_Input,
            output_model=_Output,
            error_model=ToolError,
            permissions=ToolPermissions(write_db=requires_confirmation),
            timeout_seconds=1.0,
            idempotent=False,
            requires_confirmation=requires_confirmation,
        ),
        runner=runner,
    )
    return registry


def _config(run_id: str) -> AgentRunConfig:
    return AgentRunConfig(
        run_id=run_id,
        thread_id=run_id,
        budget_total=10_000,
        max_depth=1,
        access_policy=AccessPolicy.default(),
    )


@pytest.mark.anyio
async def test_service_run_invokes_agent_loop_without_compiling_inner_graph(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def fail_compile(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise AssertionError("single-agent service must not compile a graph")

    monkeypatch.setattr(
        "rag.agent.core.compiler.GraphCompiler.compile",
        fail_compile,
    )
    service = AgentService(
        definition=_definition(),
        tool_registry=_registry(calls),
        model_turn_provider=_FinishFromResultsProvider(),
        goal_contract_provider=cast(object, _ExplodingGoalProvider()),
    )
    call = ToolCallPlan.create("write_tool", {"value": "once"})

    result = await service.run(
        AgentRunRequest(
            task="Write once.",
            run_id="service-loop-run",
            thread_id="service-loop-run",
            pending_tool_calls=[call],
        )
    )

    assert result.status == "done"
    assert result.final_answer == "wrote:once"
    assert calls == ["once"]


@pytest.mark.anyio
async def test_service_factory_accepts_loop_model_turn_provider() -> None:
    factory = AgentServiceFactory(
        tool_registry=_registry([]),
        model_turn_provider=_FinishFromResultsProvider(),
    )

    result = await factory.create(_definition()).run(
        AgentRunRequest(
            task="Answer directly.",
            run_id="service-loop-factory",
            thread_id="service-loop-factory",
        )
    )

    assert result.status == "done"
    assert result.final_answer == "direct answer"


@pytest.mark.anyio
async def test_service_resume_uses_loop_checkpoint_and_does_not_replay() -> None:
    calls: list[str] = []
    checkpointer = MemorySaver(serde=agent_checkpoint_serde())
    service = AgentService(
        definition=_definition(requires_confirmation=True),
        tool_registry=_registry(
            calls,
            requires_confirmation=True,
        ),
        model_turn_provider=_FinishFromResultsProvider(),
        checkpointer=checkpointer,
    )
    call = ToolCallPlan.create("write_tool", {"value": "approved"})

    paused = await service.run(
        AgentRunRequest(
            task="Approve one write.",
            run_id="service-loop-resume",
            thread_id="service-loop-resume",
            pending_tool_calls=[call],
        )
    )

    assert paused.status == "paused"
    request = service.pending_human_input_request(
        run_id="service-loop-resume"
    )
    assert request == paused.human_input_request
    assert (
        await service.apending_human_input_request(
            run_id="service-loop-resume"
        )
        == request
    )

    resumed = await service.resume(
        run_id="service-loop-resume",
        response=HumanInputResponse(
            request_id=request.request_id,
            decision="allow_once",
            approved_tool_call_ids=[call.tool_call_id],
        ),
    )

    assert resumed.status == "done"
    assert resumed.final_answer == "wrote:approved"
    assert resumed.human_input_request is None
    assert calls == ["approved"]


@pytest.mark.anyio
async def test_service_exposes_non_idempotent_unknown_as_reconciliation() -> None:
    run_id = "service-loop-reconciliation"
    config = _config(run_id)
    checkpointer = MemorySaver(serde=agent_checkpoint_serde())
    store = LangGraphCheckpointStore(
        checkpointer,
        run_config=config,
    )
    call = ToolCallPlan.create("write_tool", {"value": "unknown"})
    state = create_loop_state(
        task="Recover an ambiguous write.",
        run_config=config,
        pending_tool_calls=[call],
    )
    state["tool_execution_records"][call.tool_call_id] = (
        ToolExecutionRecord(
            tool_call_id=call.tool_call_id,
            tool_name=call.tool_name,
            operation_id="op-ambiguous",
            arguments_digest=tool_arguments_digest(call.arguments),
            idempotent=False,
            status="started",
            attempt_count=1,
        )
    )
    await store.save_snapshot(state, reason="crash_after_started")
    service = AgentService(
        definition=_definition(),
        tool_registry=_registry([]),
        model_turn_provider=_FinishFromResultsProvider(),
        checkpointer=checkpointer,
    )

    request = await service.apending_human_input_request(run_id=run_id)

    assert request.kind == "tool_reconciliation"
    assert request.context["operation_id"] == "op-ambiguous"
    RunRegistry.remove(run_id)


@pytest.mark.anyio
async def test_explicit_goal_spec_is_a_stop_hook_not_default_controller() -> None:
    service = AgentService(
        definition=_definition(),
        tool_registry=_registry([]),
        model_turn_provider=_PauseAfterGoalFeedbackProvider(),
        goal_contract_provider=cast(object, _ExplodingGoalProvider()),
    )
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

    result = await service.run(
        AgentRunRequest(
            task="Answer with evidence.",
            run_id="service-loop-goal-hook",
            thread_id="service-loop-goal-hook",
            goal_spec=goal,
        )
    )

    assert result.status == "paused"
    assert result.needs_user_input == "Explicit goal still needs evidence."

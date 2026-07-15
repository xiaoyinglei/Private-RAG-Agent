from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import replace

import pytest
from langgraph.checkpoint.memory import MemorySaver

from rag.agent.core.checkpointing import (
    LangGraphCheckpointStore,
    agent_checkpoint_serde,
)
from rag.agent.core.definition import AgentRuntimePolicy
from rag.agent.core.human_input import HumanInputRequest, ToolCallSummary
from rag.agent.core.messages import ModelMessage
from rag.agent.core.turn_contracts import ToolCallPlan
from rag.agent.loop.runtime import ModelTurnEnvelope
from rag.agent.loop.state import LoopPause, LoopState, ModelTurnDraft
from rag.agent.service import AgentRunRequest, AgentService
from rag.agent.sessions import (
    RuntimeBinding,
    SessionBusyError,
    SessionStore,
    TurnStatus,
)
from rag.agent.tools.executor import ExecutionStatus, ToolExecutionRecord
from rag.agent.tools.registry import ToolRegistry
from rag.agent.tools.tool import (
    CancellationMode,
    InterruptBehavior,
    JsonValue,
    NormalizedToolOutput,
    ResolvedToolUse,
    Tool,
    ToolDefinition,
    json_schema_input,
)
from rag.agent.workspace import open_workspace


class _PauseProvider:
    async def next_turn(
        self,
        state: LoopState,
        *,
        definition: AgentRuntimePolicy,
        budget_remaining: int,
    ) -> ModelTurnDraft:
        del state, definition, budget_remaining
        return ModelTurnDraft(
            action="pause",
            pause_reason="Need more input.",
        )


class _FinishProvider:
    def __init__(self, answer: str = "done") -> None:
        self.answer = answer
        self.observed: list[LoopState] = []

    async def next_turn(
        self,
        state: LoopState,
        *,
        definition: AgentRuntimePolicy,
        budget_remaining: int,
    ) -> ModelTurnEnvelope:
        del definition, budget_remaining
        self.observed.append(deepcopy(state))
        return ModelTurnEnvelope(
            draft=ModelTurnDraft(
                action="finish",
                final_answer=self.answer,
            ),
            assistant_message=ModelMessage(
                role="assistant",
                content=self.answer,
            ),
        )


def _definition() -> AgentRuntimePolicy:
    return AgentRuntimePolicy.test_factory(
        system_prompt="Resume safely.",
        allowed_tools=[],
        max_iterations=3,
    )


def _service(
    tmp_path,
    *,
    store: SessionStore,
    checkpointer: MemorySaver,
    provider: object,
    registry: ToolRegistry | None = None,
    definition: AgentRuntimePolicy | None = None,
) -> AgentService:
    return AgentService(
        definition=definition or _definition(),
        tool_registry=registry or ToolRegistry(),
        model_turn_provider=provider,
        checkpointer=checkpointer,
        workspace=open_workspace(tmp_path),
        session_store=store,
        runtime_binding=RuntimeBinding(workspace_path=str(tmp_path)),
    )


async def _save_paused_state(
    service: AgentService,
    checkpointer: MemorySaver,
    request: AgentRunRequest,
    human_request: HumanInputRequest,
) -> None:
    state = service.initial_state(request)
    state["status"] = "paused"
    state["approval_request"] = human_request
    state["pause"] = LoopPause(
        reason=human_request.question,
        request=human_request,
    )
    await LangGraphCheckpointStore(
        checkpointer,
        run_config=request.to_run_config(_definition()),
    ).save_snapshot(state, reason="test_pause")


@pytest.mark.anyio
async def test_chat_persists_paused_turn_and_blocks_new_message(
    tmp_path,
) -> None:
    store = SessionStore(tmp_path / "agent.sqlite")
    checkpointer = MemorySaver(serde=agent_checkpoint_serde())
    service = _service(
        tmp_path,
        store=store,
        checkpointer=checkpointer,
        provider=_PauseProvider(),
    )

    paused = await service.chat(AgentRunRequest(task="first"))

    assert paused.status == "paused"
    assert store.get_turn(paused.run_id).status is TurnStatus.PAUSED
    with pytest.raises(SessionBusyError, match="active Turn"):
        await service.chat(
            AgentRunRequest(
                task="must fail",
                session_id=paused.session_id,
            )
        )


@pytest.mark.anyio
async def test_resume_fails_loud_when_turn_checkpoint_is_missing(
    tmp_path,
) -> None:
    store = SessionStore(tmp_path / "agent.sqlite")
    checkpointer = MemorySaver(serde=agent_checkpoint_serde())
    session = store.create_session(RuntimeBinding(workspace_path=str(tmp_path)))
    turn = store.begin_turn(session.session_id, "missing checkpoint")
    store.mark_paused(turn.turn_id)
    service = _service(
        tmp_path,
        store=store,
        checkpointer=checkpointer,
        provider=_FinishProvider(),
    )

    with pytest.raises(KeyError, match="No checkpoint found"):
        await service.resume_turn(
            turn_id=turn.turn_id,
            action="continue",
        )

    assert store.get_turn(turn.turn_id).status is TurnStatus.PAUSED


@pytest.mark.anyio
async def test_resume_interrupted_turn_hydrates_full_session_history(
    tmp_path,
) -> None:
    store = SessionStore(tmp_path / "agent.sqlite")
    checkpointer = MemorySaver(serde=agent_checkpoint_serde())
    session = store.create_session(RuntimeBinding(workspace_path=str(tmp_path)))
    first = store.begin_turn(session.session_id, "remember alpha")
    store.sync_turn_messages(
        first.turn_id,
        (
            ModelMessage(role="user", content="remember alpha"),
            ModelMessage(role="assistant", content="remembered"),
        ),
    )
    store.mark_terminal(first.turn_id, TurnStatus.COMPLETED)
    second = store.begin_turn(
        session.session_id,
        "what did I ask you to remember?",
        lease_owner="dead-worker",
    )
    seed = _service(
        tmp_path,
        store=store,
        checkpointer=checkpointer,
        provider=_FinishProvider(),
    )
    request = AgentRunRequest(
        task="remember alpha",
        session_id=session.session_id,
        run_id=second.turn_id,
        history_messages=[
            ModelMessage(role="assistant", content="remembered"),
            ModelMessage(
                role="user",
                content="what did I ask you to remember?",
            ),
        ],
        turn_messages=[
            ModelMessage(
                role="user",
                content="what did I ask you to remember?",
            )
        ],
    )
    state = seed.initial_state(request)
    await LangGraphCheckpointStore(
        checkpointer,
        run_config=request.to_run_config(_definition()),
    ).save_snapshot(state, reason="process_interrupted")
    store.mark_interrupted(second.turn_id)
    provider = _FinishProvider("alpha")
    restored = _service(
        tmp_path,
        store=store,
        checkpointer=checkpointer,
        provider=provider,
    )

    result = await restored.resume_turn(
        turn_id=second.turn_id,
        action="continue",
        user_input=None,
    )

    assert result.status == "done"
    assert provider.observed[0]["task"] == "remember alpha"
    assert provider.observed[0]["canonical_transcript"] == [
        ModelMessage(role="assistant", content="remembered"),
        ModelMessage(
            role="user",
            content="what did I ask you to remember?",
        ),
    ]
    assert store.history(session.session_id) == (
        ModelMessage(role="user", content="remember alpha"),
        ModelMessage(role="assistant", content="remembered"),
        ModelMessage(
            role="user",
            content="what did I ask you to remember?",
        ),
        ModelMessage(role="assistant", content="alpha"),
    )
    assert store.get_turn(second.turn_id).status is TurnStatus.COMPLETED


@pytest.mark.anyio
async def test_resume_clarification_appends_user_input_to_same_turn(
    tmp_path,
) -> None:
    store = SessionStore(tmp_path / "agent.sqlite")
    checkpointer = MemorySaver(serde=agent_checkpoint_serde())
    session = store.create_session(RuntimeBinding(workspace_path=str(tmp_path)))
    turn = store.begin_turn(session.session_id, "Which target?")
    seed = _service(
        tmp_path,
        store=store,
        checkpointer=checkpointer,
        provider=_FinishProvider(),
    )
    run_request = AgentRunRequest(
        task="Which target?",
        session_id=session.session_id,
        run_id=turn.turn_id,
        turn_messages=[ModelMessage(role="user", content="Which target?")],
    )
    human_request = HumanInputRequest(
        request_id="hir_clarify",
        kind="clarification",
        question="Which target should I use?",
        options=["continue", "abort"],
    )
    await _save_paused_state(
        seed,
        checkpointer,
        run_request,
        human_request,
    )
    store.mark_paused(turn.turn_id)
    provider = _FinishProvider("target accepted")
    restored = _service(
        tmp_path,
        store=store,
        checkpointer=checkpointer,
        provider=provider,
    )

    result = await restored.resume_turn(
        turn_id=turn.turn_id,
        action="continue",
        user_input="Use production.",
    )

    assert result.status == "done"
    assert provider.observed[0]["canonical_transcript"] == [
        ModelMessage(role="user", content="Use production."),
    ]
    assert store.history(session.session_id) == (
        ModelMessage(role="user", content="Which target?"),
        ModelMessage(role="user", content="Use production."),
        ModelMessage(role="assistant", content="target accepted"),
    )


@pytest.mark.anyio
async def test_resume_tool_approval_maps_action_to_pending_calls(
    tmp_path,
) -> None:
    store = SessionStore(tmp_path / "agent.sqlite")
    checkpointer = MemorySaver(serde=agent_checkpoint_serde())
    session = store.create_session(RuntimeBinding(workspace_path=str(tmp_path)))
    turn = store.begin_turn(session.session_id, "Approve it")
    seed = _service(
        tmp_path,
        store=store,
        checkpointer=checkpointer,
        provider=_FinishProvider(),
    )
    run_request = AgentRunRequest(
        task="Approve it",
        session_id=session.session_id,
        run_id=turn.turn_id,
        turn_messages=[ModelMessage(role="user", content="Approve it")],
    )
    human_request = HumanInputRequest(
        request_id="hir_approve",
        kind="tool_approval",
        question="Approve write?",
        tool_calls=[
            ToolCallSummary(
                tool_call_id="call_write",
                tool_name="write_file",
                args_preview="path='result.txt'",
            )
        ],
        options=["allow_once", "deny", "abort"],
    )
    await _save_paused_state(
        seed,
        checkpointer,
        run_request,
        human_request,
    )
    store.mark_paused(turn.turn_id)
    provider = _FinishProvider("approved")
    restored = _service(
        tmp_path,
        store=store,
        checkpointer=checkpointer,
        provider=provider,
    )

    await restored.resume_turn(
        turn_id=turn.turn_id,
        action="allow_once",
        user_input=None,
    )

    assert provider.observed[0]["approved_tool_call_ids"] == ["call_write"]


@pytest.mark.anyio
async def test_outcome_unknown_reconciliation_never_replays_side_effect(
    tmp_path,
) -> None:
    runner_calls = 0
    schema: Mapping[str, JsonValue] = {
        "type": "object",
        "properties": {"value": {"type": "string"}},
        "required": ["value"],
        "additionalProperties": False,
    }

    def runner(_arguments: Mapping[str, JsonValue]) -> object:
        nonlocal runner_calls
        runner_calls += 1
        return {"value": "must not run"}

    tool = Tool(
        definition=ToolDefinition(
            name="remote_write",
            description="Perform a non-idempotent write.",
            input_schema=schema,
        ),
        validate_input=json_schema_input(schema),
        run=runner,
        normalize_output=lambda _raw: NormalizedToolOutput(),
        output_schema=None,
        static_effects=frozenset(),
        resolve_use=lambda _arguments: ResolvedToolUse(
            effects=frozenset(),
            targets=(),
        ),
        execution_revision="remote-write-v1",
        idempotent=False,
        concurrency_safe=False,
        cancellation_mode=CancellationMode.REMOTE_BEST_EFFORT,
        interrupt_behavior=InterruptBehavior.CANCEL,
        timeout_seconds=1.0,
        max_model_output_bytes=4096,
    )
    registry = ToolRegistry()
    registry.register(tool)
    definition = AgentRuntimePolicy.test_factory(
        system_prompt="Reconcile safely.",
        allowed_tools=["remote_write"],
        max_iterations=3,
    )
    store = SessionStore(tmp_path / "agent.sqlite")
    checkpointer = MemorySaver(serde=agent_checkpoint_serde())
    session = store.create_session(RuntimeBinding(workspace_path=str(tmp_path)))
    turn = store.begin_turn(session.session_id, "Reconcile write")
    seed = _service(
        tmp_path,
        store=store,
        checkpointer=checkpointer,
        provider=_FinishProvider(),
        registry=registry,
        definition=definition,
    )
    plan = ToolCallPlan.create("remote_write", {"value": "once"})
    run_request = AgentRunRequest(
        task="Reconcile write",
        session_id=session.session_id,
        run_id=turn.turn_id,
        pending_tool_calls=[plan],
        turn_messages=[
            ModelMessage(role="user", content="Reconcile write")
        ],
    )
    state = seed.initial_state(run_request)
    canonical = state["canonical_tool_calls"][plan.tool_call_id]
    state["tool_execution_records"][plan.tool_call_id] = replace(
        ToolExecutionRecord.prepare(canonical, tool),
        status=ExecutionStatus.OUTCOME_UNKNOWN,
        attempt_count=1,
        error_code="interrupted_outcome_unknown",
        requires_reconciliation=True,
    )
    await LangGraphCheckpointStore(
        checkpointer,
        run_config=run_request.to_run_config(definition),
    ).save_snapshot(state, reason="crash_after_started")
    store.mark_interrupted(turn.turn_id)
    provider = _FinishProvider("reconciled")
    restored = _service(
        tmp_path,
        store=store,
        checkpointer=checkpointer,
        provider=provider,
        registry=registry,
        definition=definition,
    )

    result = await restored.resume_turn(
        turn_id=turn.turn_id,
        action="mark_completed",
        user_input=None,
    )

    assert result.status == "done"
    assert runner_calls == 0
    assert result.tool_results[0].metadata["reconciled"] is True

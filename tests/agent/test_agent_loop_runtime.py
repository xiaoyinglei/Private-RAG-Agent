from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from rag.agent.core.context import AgentRunConfig, RunRegistry
from rag.agent.core.definition import AgentDefinition
from rag.agent.core.finalization import FinishCandidateBuilder
from rag.agent.core.tool_execution import (
    ToolExecutionRecord,
    ToolExecutionService,
    tool_arguments_digest,
)
from rag.agent.loop.runtime import (
    AgentLoop,
    LoopEventSink,
    ModelTurnEnvelope,
)
from rag.agent.loop.state import (
    LoopState,
    LoopTransition,
    ModelTurnDraft,
    create_loop_state,
)
from rag.agent.loop.stop_hooks import (
    StopHookBinding,
    StopHookRunner,
    StopVerdict,
)
from rag.agent.memory.compactor import LoopCompactionResult, LoopContextCompactor
from rag.agent.memory.models import MemoryPolicy
from rag.agent.state import ToolCallPlan
from rag.agent.tools.registry import ToolExecutionContext, ToolRegistry
from rag.agent.tools.spec import ToolError, ToolPermissions, ToolSpec
from rag.schema.runtime import AccessPolicy


class _Input(BaseModel):
    value: str


class _Output(BaseModel):
    text: str


class _SequenceProvider:
    def __init__(
        self,
        turns: list[ModelTurnDraft | ModelTurnEnvelope | Exception],
    ) -> None:
        self._turns = turns
        self.seen_states: list[LoopState] = []

    async def next_turn(
        self,
        state: LoopState,
        *,
        definition: AgentDefinition,
        budget_remaining: int,
    ) -> ModelTurnDraft | ModelTurnEnvelope:
        del definition, budget_remaining
        self.seen_states.append(deepcopy(state))
        if not self._turns:
            raise AssertionError("provider called after scripted turns")
        turn = self._turns.pop(0)
        if isinstance(turn, Exception):
            raise turn
        return turn


@dataclass
class _Checkpoint:
    durable: bool = True
    snapshots: list[tuple[str, LoopState]] = field(default_factory=list)
    execution_records: list[ToolExecutionRecord] = field(default_factory=list)

    async def save_snapshot(
        self,
        state: LoopState,
        *,
        reason: str,
    ) -> None:
        self.snapshots.append((reason, deepcopy(state)))

    async def write_execution_record(
        self,
        record: ToolExecutionRecord,
    ) -> None:
        self.execution_records.append(record.model_copy(deep=True))


@dataclass
class _Events(LoopEventSink):
    transitions: list[LoopTransition] = field(default_factory=list)

    async def emit(self, transition: LoopTransition) -> None:
        self.transitions.append(transition.model_copy(deep=True))


class _NoCompaction:
    def prepare(self, state: LoopState) -> LoopCompactionResult:
        del state
        return LoopCompactionResult(changed=False)


class _SequenceHook:
    def __init__(self, verdicts: list[StopVerdict]) -> None:
        self._verdicts = verdicts

    async def evaluate(
        self,
        *,
        state: LoopState,
        candidate: str,
    ) -> StopVerdict:
        del state, candidate
        return self._verdicts.pop(0)


def _config(
    run_id: str,
    *,
    memory_policy: MemoryPolicy | None = None,
) -> AgentRunConfig:
    RunRegistry.remove(run_id)
    return AgentRunConfig(
        run_id=run_id,
        thread_id=run_id,
        budget_total=20_000,
        max_depth=2,
        access_policy=AccessPolicy.default(),
        memory_policy=memory_policy or MemoryPolicy(),
    )


def _definition(
    *,
    allowed_tools: list[str] | None = None,
    max_iterations: int = 10,
) -> AgentDefinition:
    return AgentDefinition(
        agent_type="research",
        description="Loop runtime test",
        system_prompt="Use trusted tools and finish with a candidate.",
        allowed_tools=allowed_tools or [],
        max_iterations=max_iterations,
    )


def _spec(
    name: str,
    *,
    idempotent: bool = True,
    requires_confirmation: bool = False,
    max_retries: int = 0,
) -> ToolSpec:
    return ToolSpec(
        name=name,
        description="Loop runtime tool",
        input_model=_Input,
        output_model=_Output,
        error_model=ToolError,
        permissions=ToolPermissions(),
        timeout_seconds=1.0,
        max_retries=max_retries,
        idempotent=idempotent,
        concurrency_safe=idempotent,
        requires_confirmation=requires_confirmation,
    )


def _accepting_stop_runner() -> StopHookRunner:
    return StopHookRunner(hooks=(), max_blocks=3)


def _loop(
    *,
    definition: AgentDefinition,
    provider: _SequenceProvider,
    tool_runner: ToolExecutionService,
    checkpoint: _Checkpoint,
    events: _Events | None = None,
    context_manager: object | None = None,
    stop_runner: StopHookRunner | None = None,
    max_model_retries: int = 1,
) -> AgentLoop:
    return AgentLoop(
        definition=definition,
        model_provider=provider,
        context_manager=context_manager or _NoCompaction(),
        tool_runner=tool_runner,
        checkpoint_store=checkpoint,
        stop_hook_runner=stop_runner or _accepting_stop_runner(),
        finish_candidate_builder=FinishCandidateBuilder(),
        event_sink=events or _Events(),
        max_model_retries=max_model_retries,
    )


@pytest.mark.anyio
async def test_model_tool_result_next_turn_and_finish() -> None:
    config = _config("loop-basic")
    call = ToolCallPlan.create("echo", {"value": "hello"})
    provider = _SequenceProvider(
        [
            ModelTurnDraft(action="execute", tool_calls=(call,)),
            ModelTurnDraft(action="finish", final_answer="Final answer."),
        ]
    )
    registry = ToolRegistry()
    registry.register(
        _spec("echo"),
        runner=lambda payload: _Output(text=payload.value),
    )
    checkpoint = _Checkpoint()
    events = _Events()
    state = create_loop_state(task="Echo and answer.", run_config=config)

    result = await _loop(
        definition=_definition(allowed_tools=["echo"]),
        provider=provider,
        tool_runner=ToolExecutionService(
            tool_registry=registry,
            record_writer=checkpoint,
        ),
        checkpoint=checkpoint,
        events=events,
    ).run(state)

    assert result["status"] == "completed"
    assert result["final_answer"] == "Final answer."
    assert result["iteration"] == 2
    assert len(result["tool_results"]) == 1
    assert len(result["structured_observations"]) == 1
    assert provider.seen_states[1]["tool_results"][0].status == "ok"
    assert [record.status for record in checkpoint.execution_records] == [
        "prepared",
        "started",
        "completed",
    ]
    accepted_execute = next(
        snapshot
        for reason, snapshot in checkpoint.snapshots
        if reason == "model_turn"
        and snapshot["last_model_turn"] is not None
        and snapshot["last_model_turn"].action == "execute"
    )
    assert accepted_execute["pending_tool_calls"] == [call]
    assert "tool_results_recorded" in [reason for reason, _ in checkpoint.snapshots]
    assert events.transitions[-1].reason == "finished"


@pytest.mark.anyio
async def test_multiple_model_tool_turns_run_in_order() -> None:
    config = _config("loop-multiple-tools")
    first = ToolCallPlan.create("echo", {"value": "one"})
    second = ToolCallPlan.create("echo", {"value": "two"})
    provider = _SequenceProvider(
        [
            ModelTurnDraft(action="execute", tool_calls=(first,)),
            ModelTurnDraft(action="execute", tool_calls=(second,)),
            ModelTurnDraft(action="finish", final_answer="Both complete."),
        ]
    )
    executed: list[str] = []

    def runner(payload: _Input) -> _Output:
        executed.append(payload.value)
        return _Output(text=payload.value)

    registry = ToolRegistry()
    registry.register(_spec("echo"), runner=runner)
    checkpoint = _Checkpoint()

    result = await _loop(
        definition=_definition(allowed_tools=["echo"]),
        provider=provider,
        tool_runner=ToolExecutionService(
            tool_registry=registry,
            record_writer=checkpoint,
        ),
        checkpoint=checkpoint,
    ).run(create_loop_state(task="Run two calls.", run_config=config))

    assert result["status"] == "completed"
    assert executed == ["one", "two"]
    assert result["iteration"] == 3
    assert [
        observation.tool_call_id
        for observation in result["structured_observations"]
    ] == [first.tool_call_id, second.tool_call_id]


@pytest.mark.anyio
async def test_approval_pause_and_resume_invokes_mutation_once() -> None:
    config = _config("loop-approval")
    call = ToolCallPlan.create("write", {"value": "report"})
    provider = _SequenceProvider(
        [
            ModelTurnDraft(action="execute", tool_calls=(call,)),
            ModelTurnDraft(action="finish", final_answer="Written once."),
        ]
    )
    invocations = 0

    def runner(payload: _Input) -> _Output:
        nonlocal invocations
        invocations += 1
        return _Output(text=payload.value)

    registry = ToolRegistry()
    registry.register(
        _spec(
            "write",
            idempotent=False,
            requires_confirmation=True,
        ),
        runner=runner,
    )
    checkpoint = _Checkpoint()
    loop = _loop(
        definition=_definition(allowed_tools=["write"]),
        provider=provider,
        tool_runner=ToolExecutionService(
            tool_registry=registry,
            record_writer=checkpoint,
        ),
        checkpoint=checkpoint,
    )
    state = create_loop_state(task="Write a report.", run_config=config)

    paused = await loop.run(state)

    assert paused["status"] == "paused"
    assert paused["pause"] is not None
    assert paused["pause"].request is not None
    assert paused["pause"].request.kind == "tool_approval"
    assert invocations == 0

    paused["status"] = "running"
    paused["approved_tool_call_ids"] = [call.tool_call_id]
    paused["approval_request"] = None
    paused["pause"] = None
    completed = await loop.run(paused)

    assert completed["status"] == "completed"
    assert invocations == 1
    assert completed["tool_execution_records"][call.tool_call_id].status == "completed"


@pytest.mark.anyio
async def test_completed_record_is_removed_from_pending_without_replay() -> None:
    config = _config("loop-completed-record")
    call = ToolCallPlan.create("echo", {"value": "already done"})
    record = ToolExecutionRecord(
        tool_call_id=call.tool_call_id,
        tool_name=call.tool_name,
        operation_id="op-completed",
        arguments_digest=tool_arguments_digest(call.arguments),
        idempotent=True,
        status="completed",
        attempt_count=1,
    )
    invocations = 0

    def runner(payload: _Input) -> _Output:
        nonlocal invocations
        invocations += 1
        return _Output(text=payload.value)

    registry = ToolRegistry()
    registry.register(_spec("echo"), runner=runner)
    checkpoint = _Checkpoint()
    state = create_loop_state(
        task="Resume after completion.",
        run_config=config,
        pending_tool_calls=[call],
    )
    state["tool_execution_records"][call.tool_call_id] = record

    result = await _loop(
        definition=_definition(allowed_tools=["echo"]),
        provider=_SequenceProvider(
            [ModelTurnDraft(action="finish", final_answer="No replay.")]
        ),
        tool_runner=ToolExecutionService(
            tool_registry=registry,
            record_writer=checkpoint,
        ),
        checkpoint=checkpoint,
    ).run(state)

    assert result["status"] == "completed"
    assert result["pending_tool_calls"] == []
    assert invocations == 0
    recorded = [
        snapshot
        for reason, snapshot in checkpoint.snapshots
        if reason == "tool_results_recorded"
    ][0]
    assert recorded["latest_transition"] is not None
    assert recorded["latest_transition"].detail[
        "skipped_completed_tool_call_ids"
    ] == [call.tool_call_id]


@pytest.mark.anyio
async def test_idempotent_started_recovery_reuses_operation_id() -> None:
    config = _config("loop-idempotent-recovery")
    call = ToolCallPlan.create("read", {"value": "data"})
    record = ToolExecutionRecord(
        tool_call_id=call.tool_call_id,
        tool_name=call.tool_name,
        operation_id="op-original",
        arguments_digest=tool_arguments_digest(call.arguments),
        idempotent=True,
        status="started",
        attempt_count=1,
    )
    seen_operation_ids: list[str | None] = []

    def runner(
        payload: _Input,
        context: ToolExecutionContext,
    ) -> _Output:
        seen_operation_ids.append(context.operation_id)
        return _Output(text=payload.value)

    registry = ToolRegistry()
    registry.register(_spec("read", idempotent=True))
    registry.register_contextual_runner("read", runner)
    checkpoint = _Checkpoint()
    state = create_loop_state(
        task="Resume the read.",
        run_config=config,
        pending_tool_calls=[call],
    )
    state["tool_execution_records"][call.tool_call_id] = record

    result = await _loop(
        definition=_definition(allowed_tools=["read"]),
        provider=_SequenceProvider(
            [ModelTurnDraft(action="finish", final_answer="Recovered.")]
        ),
        tool_runner=ToolExecutionService(
            tool_registry=registry,
            record_writer=checkpoint,
        ),
        checkpoint=checkpoint,
    ).run(state)

    assert result["status"] == "completed"
    assert seen_operation_ids == ["op-original"]


@pytest.mark.anyio
async def test_non_idempotent_started_recovery_pauses_for_reconciliation() -> None:
    config = _config("loop-unknown-recovery")
    call = ToolCallPlan.create("write", {"value": "data"})
    record = ToolExecutionRecord(
        tool_call_id=call.tool_call_id,
        tool_name=call.tool_name,
        operation_id="op-unknown",
        arguments_digest=tool_arguments_digest(call.arguments),
        idempotent=False,
        status="started",
        attempt_count=1,
    )
    invocations = 0

    def runner(payload: _Input) -> _Output:
        nonlocal invocations
        invocations += 1
        return _Output(text=payload.value)

    registry = ToolRegistry()
    registry.register(_spec("write", idempotent=False), runner=runner)
    checkpoint = _Checkpoint()
    state = create_loop_state(
        task="Recover a write.",
        run_config=config,
        pending_tool_calls=[call],
    )
    state["tool_execution_records"][call.tool_call_id] = record

    result = await _loop(
        definition=_definition(allowed_tools=["write"]),
        provider=_SequenceProvider([]),
        tool_runner=ToolExecutionService(
            tool_registry=registry,
            record_writer=checkpoint,
        ),
        checkpoint=checkpoint,
    ).run(state)

    assert result["status"] == "paused"
    assert result["pause"] is not None
    assert result["pause"].request is not None
    assert result["pause"].request.kind == "tool_reconciliation"
    assert result["tool_execution_records"][call.tool_call_id].status == "unknown"
    assert invocations == 0


@pytest.mark.anyio
async def test_explicit_model_pause_is_typed_and_checkpointed() -> None:
    config = _config("loop-model-pause")
    checkpoint = _Checkpoint()
    state = create_loop_state(task="Choose a source.", run_config=config)

    result = await _loop(
        definition=_definition(),
        provider=_SequenceProvider(
            [
                ModelTurnDraft(
                    action="pause",
                    pause_reason="Choose source A or B.",
                )
            ]
        ),
        tool_runner=ToolExecutionService(tool_registry=ToolRegistry()),
        checkpoint=checkpoint,
    ).run(state)

    assert result["status"] == "paused"
    assert result["pause"] is not None
    assert result["pause"].reason == "Choose source A or B."
    assert checkpoint.snapshots[-1][0] == "model_pause"


@pytest.mark.anyio
async def test_stop_hook_block_feedback_then_accept() -> None:
    config = _config("loop-stop-block")
    hook = _SequenceHook(
        [
            StopVerdict(
                action="block",
                code="needs_citation",
                message="Add a citation.",
            ),
            StopVerdict(action="accept", code="accepted"),
        ]
    )
    provider = _SequenceProvider(
        [
            ModelTurnDraft(action="finish", final_answer="Draft."),
            ModelTurnDraft(action="finish", final_answer="Draft with citation [1]."),
        ]
    )
    checkpoint = _Checkpoint()

    result = await _loop(
        definition=_definition(),
        provider=provider,
        tool_runner=ToolExecutionService(tool_registry=ToolRegistry()),
        checkpoint=checkpoint,
        stop_runner=StopHookRunner(
            hooks=(
                StopHookBinding(
                    name="citation",
                    hook=hook,
                    critical=True,
                ),
            ),
            max_blocks=3,
        ),
    ).run(create_loop_state(task="Answer with citation.", run_config=config))

    assert result["status"] == "completed"
    assert result["final_answer"] == "Draft with citation [1]."
    assert result["stop_hook_feedback"][0].code == "needs_citation"
    assert provider.seen_states[1]["stop_hook_feedback"][0].message == "Add a citation."
    assert "stop_hook_blocked" in [
        snapshot["latest_transition"].reason
        for _, snapshot in checkpoint.snapshots
        if snapshot["latest_transition"] is not None
    ]


@pytest.mark.anyio
async def test_stop_hook_halt_fails_visibly() -> None:
    config = _config("loop-stop-halt")
    hook = _SequenceHook(
        [
            StopVerdict(
                action="halt",
                code="unsafe_finish",
                message="Candidate violates a hard requirement.",
            )
        ]
    )

    result = await _loop(
        definition=_definition(),
        provider=_SequenceProvider(
            [ModelTurnDraft(action="finish", final_answer="Unsafe.")]
        ),
        tool_runner=ToolExecutionService(tool_registry=ToolRegistry()),
        checkpoint=_Checkpoint(),
        stop_runner=StopHookRunner(
            hooks=(
                StopHookBinding(
                    name="safety",
                    hook=hook,
                    critical=True,
                ),
            ),
            max_blocks=3,
        ),
    ).run(create_loop_state(task="Finish safely.", run_config=config))

    assert result["status"] == "failed"
    assert result["terminal"] is not None
    assert result["terminal"].stop_reason == "unsafe_finish"
    assert result["terminal"].error == "Candidate violates a hard requirement."


@pytest.mark.anyio
async def test_max_iterations_terminates_blocked_finish_loop() -> None:
    config = _config("loop-max-iterations")
    hook = _SequenceHook(
        [
            StopVerdict(action="block", code="not_ready", message="Continue.")
            for _ in range(3)
        ]
    )
    provider = _SequenceProvider(
        [
            ModelTurnDraft(action="finish", final_answer="Draft one."),
            ModelTurnDraft(action="finish", final_answer="Draft two."),
        ]
    )

    result = await _loop(
        definition=_definition(max_iterations=2),
        provider=provider,
        tool_runner=ToolExecutionService(tool_registry=ToolRegistry()),
        checkpoint=_Checkpoint(),
        stop_runner=StopHookRunner(
            hooks=(
                StopHookBinding(
                    name="readiness",
                    hook=hook,
                    critical=True,
                ),
            ),
            max_blocks=10,
        ),
    ).run(create_loop_state(task="Keep trying.", run_config=config))

    assert result["status"] == "failed"
    assert result["terminal"] is not None
    assert result["terminal"].stop_reason == "max_iterations"
    assert result["latest_transition"] is not None
    assert result["latest_transition"].reason == "max_iterations"


@pytest.mark.anyio
async def test_model_retry_and_provider_fallback_are_emitted() -> None:
    config = _config("loop-model-retry")
    events = _Events()
    provider = _SequenceProvider(
        [
            RuntimeError("temporary model failure"),
            ModelTurnEnvelope(
                draft=ModelTurnDraft(
                    action="finish",
                    final_answer="Fallback answer.",
                ),
                transitions=(
                    LoopTransition(
                        reason="fallback",
                        iteration=0,
                        detail={"provider": "backup"},
                    ),
                ),
            ),
        ]
    )

    result = await _loop(
        definition=_definition(),
        provider=provider,
        tool_runner=ToolExecutionService(tool_registry=ToolRegistry()),
        checkpoint=_Checkpoint(),
        events=events,
        max_model_retries=1,
    ).run(create_loop_state(task="Use fallback.", run_config=config))

    assert result["status"] == "completed"
    reasons = [transition.reason for transition in events.transitions]
    assert "retry" in reasons
    assert "fallback" in reasons
    assert reasons[-1] == "finished"


@pytest.mark.anyio
async def test_tool_retry_transition_is_emitted() -> None:
    config = _config("loop-tool-retry")
    call = ToolCallPlan.create("flaky", {"value": "retry"})
    invocations = 0

    def runner(payload: _Input) -> _Output:
        nonlocal invocations
        invocations += 1
        if invocations == 1:
            raise RuntimeError("temporary tool failure")
        return _Output(text=payload.value)

    registry = ToolRegistry()
    registry.register(
        _spec("flaky", idempotent=True, max_retries=1),
        runner=runner,
    )
    events = _Events()
    checkpoint = _Checkpoint()

    result = await _loop(
        definition=_definition(allowed_tools=["flaky"]),
        provider=_SequenceProvider(
            [
                ModelTurnDraft(action="execute", tool_calls=(call,)),
                ModelTurnDraft(action="finish", final_answer="Retried."),
            ]
        ),
        tool_runner=ToolExecutionService(
            tool_registry=registry,
            record_writer=checkpoint,
        ),
        checkpoint=checkpoint,
        events=events,
    ).run(create_loop_state(task="Retry the tool.", run_config=config))

    assert result["status"] == "completed"
    assert result["tool_results"][0].retry_count == 1
    assert "retry" in [transition.reason for transition in events.transitions]


@pytest.mark.anyio
async def test_context_compaction_runs_before_provider_and_is_checkpointed() -> None:
    config = _config(
        "loop-runtime-compaction",
        memory_policy=MemoryPolicy(
            message_compaction_min_count=3,
            max_message_tail_count=1,
        ),
    )
    provider = _SequenceProvider(
        [ModelTurnDraft(action="finish", final_answer="Compacted.")]
    )
    checkpoint = _Checkpoint()
    state = create_loop_state(
        task="Summarize history.",
        run_config=config,
        messages=[
            HumanMessage(content=f"message {index}", id=f"msg-{index}")
            for index in range(4)
        ],
    )

    result = await _loop(
        definition=_definition(),
        provider=provider,
        tool_runner=ToolExecutionService(tool_registry=ToolRegistry()),
        checkpoint=checkpoint,
        context_manager=LoopContextCompactor(),
    ).run(state)

    assert result["status"] == "completed"
    assert [message.id for message in provider.seen_states[0]["messages"]] == [
        "msg-3"
    ]
    assert "compaction" in [reason for reason, _ in checkpoint.snapshots]


@pytest.mark.anyio
async def test_agent_tool_executes_through_same_loop_boundary() -> None:
    config = _config("loop-child-agent-tool")
    call = ToolCallPlan.create("agent_child", {"value": "delegated"})
    seen_contexts: list[ToolExecutionContext] = []

    def runner(
        payload: _Input,
        context: ToolExecutionContext,
    ) -> _Output:
        seen_contexts.append(context)
        return _Output(text=f"child:{payload.value}")

    registry = ToolRegistry()
    registry.register(_spec("agent_child", idempotent=True))
    registry.register_contextual_runner("agent_child", runner)
    checkpoint = _Checkpoint()

    result = await _loop(
        definition=_definition(allowed_tools=["agent_child"]),
        provider=_SequenceProvider(
            [
                ModelTurnDraft(action="execute", tool_calls=(call,)),
                ModelTurnDraft(action="finish", final_answer="Delegated result."),
            ]
        ),
        tool_runner=ToolExecutionService(
            tool_registry=registry,
            record_writer=checkpoint,
        ),
        checkpoint=checkpoint,
    ).run(create_loop_state(task="Delegate once.", run_config=config))

    assert result["status"] == "completed"
    assert seen_contexts[0].run_config.run_id == config.run_id
    assert seen_contexts[0].state is result
    assert seen_contexts[0].definition is not None
    assert seen_contexts[0].definition.max_depth == 2


def test_loop_runtime_has_no_graph_node_dependency() -> None:
    source = Path("rag/agent/loop/runtime.py").read_text(encoding="utf-8")

    assert "rag.agent.graphs" not in source

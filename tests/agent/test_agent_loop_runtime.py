from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterable, Mapping
from copy import deepcopy
from dataclasses import dataclass, field, replace
from pathlib import Path

import pytest
from langgraph.checkpoint.memory import MemorySaver

from agent_runtime.planning import AgentPlan, GoalCommitment, PlanStep
from rag.agent.core.checkpointing import (
    CheckpointStore,
    LangGraphCheckpointStore,
    agent_checkpoint_serde,
)
from rag.agent.core.context import AgentRunConfig, TurnRegistry
from rag.agent.core.definition import AgentRuntimePolicy
from rag.agent.core.finalization import FinishCandidateBuilder
from rag.agent.core.goal_contract import (
    GoalConstraint,
    GoalPlanContract,
    GoalSpec,
)
from rag.agent.core.human_input import HumanInputResponse
from rag.agent.core.model_request import build_tool_manifest
from rag.agent.core.observations import ObservationBatch
from rag.agent.core.turn_contracts import ToolCallPlan
from rag.agent.loop.runtime import (
    AgentLoop,
    LoopEventSink,
    ModelTurnEnvelope,
    _approval_request,
    _claim_matching_plan_evidence,
    _guard_evidence_driven_progress,
    _guard_repeated_successful_inspections,
)
from rag.agent.loop.state import (
    LoopState,
    LoopTransition,
    ModelTurnDraft,
    create_loop_state,
)
from rag.agent.loop.stop_hooks import StopHookRunner
from rag.agent.memory.compactor import LoopCompactionResult
from rag.agent.memory.models import MemoryPolicy
from rag.agent.skills.catalog import SkillCatalog
from rag.agent.skills.loader import scan_and_load_skills
from rag.agent.skills.runtime import SkillRuntime
from rag.agent.streaming.events import EventType, StreamEvent, text_delta
from rag.agent.tools.builtins.planning import create_update_plan_tool
from rag.agent.tools.executor import (
    ExecutionStatus,
    ToolExecutionRecord,
    ToolExecutor,
)
from rag.agent.tools.integrations.skills import create_invoke_skill_tool
from rag.agent.tools.permissions import ToolExecutionContext
from rag.agent.tools.selection import (
    FindToolMatch,
    FindToolsOutput,
    create_find_tools_tool,
)
from rag.agent.tools.tool import (
    CancellationMode,
    InterruptBehavior,
    JsonValue,
    NormalizedToolOutput,
    ResolvedToolUse,
    Tool,
    ToolCall,
    ToolCallOrigin,
    ToolContentBlock,
    ToolDefinition,
    ToolEffect,
    ToolResult,
    json_schema_input,
)
from rag.providers.llm_gateway import LLMToolCallValidationError


class _SequenceProvider:
    def __init__(
        self,
        turns: list[ModelTurnDraft | ModelTurnEnvelope | Exception],
    ) -> None:
        self._turns = turns
        self.seen_states: list[LoopState] = []
        self.seen_budget_remaining: list[int] = []

    async def next_turn(
        self,
        state: LoopState,
        *,
        definition: AgentRuntimePolicy,
        budget_remaining: int,
    ) -> ModelTurnDraft | ModelTurnEnvelope:
        del definition
        self.seen_states.append(deepcopy(state))
        self.seen_budget_remaining.append(budget_remaining)
        value = self._turns.pop(0)
        if isinstance(value, Exception):
            raise value
        return value


class _SinkAwareProvider:
    def __init__(self) -> None:
        self._stream_sink: object | None = None

    async def next_turn(
        self,
        state: LoopState,
        *,
        definition: AgentRuntimePolicy,
        budget_remaining: int,
    ) -> ModelTurnDraft:
        del definition, budget_remaining
        emit = getattr(self._stream_sink, "emit", None)
        if callable(emit):
            await emit(
                text_delta(
                    "partial",
                    turn_id=state["run_config"].turn_id,
                    iteration=state["iteration"],
                )
            )
        return ModelTurnDraft(action="finish", final_answer="Final answer.")


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
        self.execution_records.append(record)


@dataclass
class _Events(LoopEventSink):
    transitions: list[LoopTransition] = field(default_factory=list)

    async def emit(self, transition: LoopTransition) -> None:
        self.transitions.append(transition.model_copy(deep=True))


class _NoCompaction:
    def prepare(self, state: LoopState) -> LoopCompactionResult:
        del state
        return LoopCompactionResult(changed=False)


def _config(
    run_id: str,
    *,
    budget: int | None = 20_000,
    max_turns: int | None = None,
) -> AgentRunConfig:
    TurnRegistry.remove(run_id)
    return AgentRunConfig(
        turn_id=run_id,
        llm_budget_total=budget,
        max_turns=max_turns,
    )


def _definition(
    names: list[str] | tuple[str, ...] = (),
    *,
    max_iterations: int = 10,
) -> AgentRuntimePolicy:
    return AgentRuntimePolicy.test_factory(
        system_prompt="Use canonical tools.",
        allowed_tools=list(names),
        max_iterations=max_iterations,
    )


def _tool(
    name: str,
    runner: object,
    *,
    schema: Mapping[str, JsonValue] | None = None,
    effects: frozenset[ToolEffect] = frozenset(),
    metadata: Mapping[str, JsonValue] | None = None,
    idempotent: bool = True,
) -> Tool:
    input_schema = schema or {
        "type": "object",
        "properties": {"value": {"type": "string"}},
        "required": ["value"],
        "additionalProperties": False,
    }

    def normalize(raw: object) -> NormalizedToolOutput:
        text = str(raw)
        return NormalizedToolOutput(
            content=(ToolContentBlock(type="text", data={"text": text}),),
            structured_content={"text": text},
            metadata=metadata or {},
        )

    return Tool(
        definition=ToolDefinition(
            name=name,
            description=f"Use {name}.",
            input_schema=input_schema,
        ),
        validate_input=json_schema_input(input_schema),
        run=runner,  # type: ignore[arg-type]
        normalize_output=normalize,
        output_schema=None,
        static_effects=effects,
        resolve_use=lambda _arguments: ResolvedToolUse(
            effects=effects,
            targets=(),
        ),
        execution_revision=f"{name}-v1",
        idempotent=idempotent,
        concurrency_safe=True,
        cancellation_mode=CancellationMode.COOPERATIVE,
        interrupt_behavior=InterruptBehavior.CANCEL,
        timeout_seconds=1.0,
        max_model_output_bytes=4096,
    )


def _loop(
    *,
    provider: object,
    tools: tuple[Tool, ...] = (),
    checkpoint: CheckpointStore | None = None,
    definition: AgentRuntimePolicy | None = None,
    events: _Events | None = None,
    max_model_retries: int = 1,
    skill_runtime: SkillRuntime | None = None,
    goal_spec: GoalSpec | None = None,
) -> AgentLoop:
    snapshot = {tool.definition.name: tool for tool in tools}
    return AgentLoop(
        definition=definition or _definition(tuple(snapshot)),
        model_provider=provider,  # type: ignore[arg-type]
        context_manager=_NoCompaction(),
        tool_executor=ToolExecutor(snapshot),
        registry_snapshot=snapshot,
        execution_context=ToolExecutionContext(),
        checkpoint_store=checkpoint or _Checkpoint(),
        stop_hook_runner=StopHookRunner(hooks=(), max_blocks=3),
        finish_candidate_builder=FinishCandidateBuilder(),
        event_sink=events or _Events(),
        max_model_retries=max_model_retries,
        skill_runtime=skill_runtime,
        goal_spec=goal_spec,
    )


async def _collect(events: AsyncIterable[StreamEvent]) -> list[StreamEvent]:
    return [event async for event in events]


@pytest.mark.anyio
async def test_model_tool_result_next_turn_and_finish() -> None:
    call = ToolCallPlan.create("echo", {"value": "hello"})
    provider = _SequenceProvider(
        [
            ModelTurnDraft(action="execute", tool_calls=(call,)),
            ModelTurnDraft(action="finish", final_answer="Final answer."),
        ]
    )
    checkpoint = _Checkpoint()
    state = create_loop_state(current_message="Echo.", run_config=_config("loop-basic"))
    state["resident_tool_names"] = ["echo"]

    result = await _loop(
        provider=provider,
        tools=(_tool("echo", lambda arguments: arguments["value"]),),
        checkpoint=checkpoint,
    ).run(state)

    assert result["status"] == "completed"
    assert result["finish_state"].final_answer == "Final answer."
    assert result["tool_results"][0].structured_content == {"text": "hello"}
    assert [message.role for message in result["turn_transcript"][-2:]] == [
        "tool",
        "assistant",
    ]
    assert result["turn_transcript"][-1].content == "Final answer."
    assert any(reason == "tool_results_recorded" for reason, _ in checkpoint.snapshots)
    assert [record.status.value for record in checkpoint.execution_records] == [
        "prepared",
        "started",
        "completed",
    ]


@pytest.mark.anyio
async def test_repeated_failure_evidence_is_fingerprinted_in_model_transcript() -> None:
    failure_text = "same failing test output\n" * 100
    attempt = 0

    def fail_with_variable_duration(
        _arguments: Mapping[str, JsonValue],
    ) -> Mapping[str, JsonValue]:
        nonlocal attempt
        attempt += 1
        return {
            "stdout": "",
            "stderr": failure_text,
            "exit_code": 1,
            "duration_ms": attempt,
        }

    def normalize_failure(raw: object) -> NormalizedToolOutput:
        assert isinstance(raw, Mapping)
        return NormalizedToolOutput(
            structured_content=dict(raw),
            is_error=True,
            error_code="command_failed",
            error_message="command exited with status 1",
            retryable=False,
        )

    tool = replace(
        _tool("run_command", fail_with_variable_duration),
        normalize_output=normalize_failure,
    )
    calls = tuple(
        ToolCallPlan.create("run_command", {"value": command})
        for command in ("pytest -q", "pytest --quiet")
    )
    state = create_loop_state(
        current_message="Run the verification command.",
        run_config=_config("loop-fold-repeated-failure-evidence"),
        pending_tool_calls=calls,
    )
    state["resident_tool_names"] = ["run_command"]

    result = await _loop(
        provider=_SequenceProvider(
            [ModelTurnDraft(action="finish", final_answer="Captured.")]
        ),
        tools=(tool,),
    ).run(state)

    tool_messages = [
        json.loads(message.content)
        for message in result["turn_transcript"]
        if message.role == "tool"
    ]
    assert len(tool_messages) == 2
    assert tool_messages[0]["structured_content"]["stderr"] == failure_text
    repeated = tool_messages[1]["structured_content"]
    assert repeated["repeated_failure"] is True
    assert repeated["original_tool_call_id"] == calls[0].tool_call_id
    assert repeated["repeat_count"] == 2
    assert len(repeated["evidence_fingerprint"]) == 64
    assert failure_text not in result["turn_transcript"][2].content
    assert [
        item.structured_content["stderr"]
        for item in result["tool_results"]
        if isinstance(item.structured_content, Mapping)
    ] == [failure_text, failure_text]


@pytest.mark.anyio
async def test_apply_patch_result_event_exposes_only_cli_diff_details() -> None:
    call = ToolCallPlan.create("apply_patch", {"value": "edit"})
    provider = _SequenceProvider(
        [
            ModelTurnDraft(action="execute", tool_calls=(call,)),
            ModelTurnDraft(action="finish", final_answer="Done."),
        ]
    )
    state = create_loop_state(
        current_message="Edit the file.",
        run_config=_config("loop-patch-diff"),
    )
    state["resident_tool_names"] = ["apply_patch"]
    tool = _tool(
        "apply_patch",
        lambda _arguments: "patched",
        metadata={
            "file_path": "src/example.py",
            "diff": "--- a/src/example.py\n+++ b/src/example.py\n-old\n+new",
            "diff_truncated": False,
            "private_value": "must-not-leak",
        },
    )

    events = await _collect(_loop(provider=provider, tools=(tool,)).run_streaming(state))

    result = next(event for event in events if event.type is EventType.TOOL_USE_RESULT)
    assert result.data["details"] == {
        "file_path": "src/example.py",
        "diff": "--- a/src/example.py\n+++ b/src/example.py\n-old\n+new",
        "diff_truncated": False,
    }


@pytest.mark.anyio
async def test_approval_pause_is_a_checkpointed_human_input_event() -> None:
    tool = _tool(
        "remote_lookup",
        lambda arguments: arguments["value"],
        effects=frozenset({ToolEffect.NETWORK}),
    )
    call = ToolCallPlan.create("remote_lookup", {"value": "public docs"})
    state = create_loop_state(
        current_message="Look this up.",
        run_config=_config("loop-approval-event"),
        pending_tool_calls=(call,),
    )
    state["resident_tool_names"] = ["remote_lookup"]
    checkpoint = _Checkpoint()

    events = await _collect(
        _loop(
            provider=_SequenceProvider([]),
            tools=(tool,),
            checkpoint=checkpoint,
        ).run_streaming(state)
    )

    assert state["status"] == "paused"
    assert state["tool_results"] == []
    assert any(event.type is EventType.HUMAN_INPUT_REQUIRED for event in events)
    assert not any(event.type is EventType.TOOL_USE_ERROR for event in events)
    start = next(event for event in events if event.type is EventType.TOOL_USE_START)
    assert "public docs" in start.data["input_preview"]
    approval_snapshots = [snapshot for reason, snapshot in checkpoint.snapshots if reason == "tool_pause"]
    assert approval_snapshots[-1]["status"] == "paused"


def test_run_command_approval_shows_full_security_context(tmp_path: Path) -> None:
    command = "printf '\x1b[2J'\npython -c \"print('" + ("x" * 300) + "')\""
    call = ToolCall(
        tool_call_id="call_command",
        tool_name="run_command",
        arguments={
            "command": command,
            "working_dir": ".",
            "timeout_seconds": 120.0,
            "network": True,
        },
        origin=ToolCallOrigin(
            request_id="request_command",
            toolset_revision="tools_v1",
            exposed_tool_names=("run_command",),
        ),
    )
    result = ToolResult(
        tool_call_id=call.tool_call_id,
        tool_name=call.tool_name,
        is_error=True,
        error_code="approval_required",
        error_message="approval required for network access",
        retryable=True,
        metadata={
            "approval_id": "call_command::network",
            "approval_scope": "network",
            "cwd": str(tmp_path),
            "network_requested": True,
            "execution_mode": "restricted_sandbox",
        },
    )

    request = _approval_request(result, call)

    summary = request.tool_calls[0]
    assert summary.approval_id == "call_command::network"
    assert json.dumps(command, ensure_ascii=False) in summary.args_preview
    assert "\x1b" not in summary.args_preview
    assert "\\u001b" in summary.args_preview
    assert f"cwd: {json.dumps(str(tmp_path))}" in summary.args_preview
    assert "network: requested (separate approval required)" in summary.args_preview
    assert "execution mode: restricted_sandbox" in summary.args_preview
    assert request.context["approval_scope"] == "network"
    assert "network access" in request.question


@pytest.mark.anyio
async def test_skill_activation_is_checkpointed_with_tool_result(
    tmp_path: Path,
) -> None:
    skill_dir = tmp_path / ".agents" / "skills" / "review"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: review\ndescription: Review code\n---\nReview carefully.\n",
        encoding="utf-8",
    )
    runtime = SkillRuntime(SkillCatalog(scan_and_load_skills(tmp_path, repo_root=tmp_path)))
    invoke_tool = create_invoke_skill_tool(runtime.invoke_skill)
    call = ToolCallPlan.create(
        "invoke_skill",
        {"name": "project:review"},
    )
    provider = _SequenceProvider(
        [
            ModelTurnDraft(action="execute", tool_calls=(call,)),
            ModelTurnDraft(action="finish", final_answer="Reviewed."),
        ]
    )
    checkpoint = _Checkpoint()
    state = create_loop_state(
        current_message="Review this.",
        run_config=_config("loop-skill-activation"),
    )
    state["resident_tool_names"] = ["invoke_skill"]

    result = await _loop(
        provider=provider,
        tools=(invoke_tool,),
        checkpoint=checkpoint,
        skill_runtime=runtime,
    ).run(state)

    assert "project:review" in result["skill_state"].active
    recorded = [snapshot for reason, snapshot in checkpoint.snapshots if reason == "tool_results_recorded"]
    assert recorded
    assert "project:review" in recorded[-1]["skill_state"].active
    assert recorded[-1]["tool_results"][-1].tool_name == "invoke_skill"


@pytest.mark.anyio
async def test_multiple_tool_calls_preserve_model_order() -> None:
    seen: list[str] = []
    calls = (
        ToolCallPlan.create("echo", {"value": "one"}),
        ToolCallPlan.create("echo", {"value": "two"}),
    )
    provider = _SequenceProvider(
        [
            ModelTurnDraft(action="execute", tool_calls=calls),
            ModelTurnDraft(action="finish", final_answer="done"),
        ]
    )
    state = create_loop_state(current_message="Echo twice.", run_config=_config("loop-order"))
    state["resident_tool_names"] = ["echo"]

    await _loop(
        provider=provider,
        tools=(
            _tool(
                "echo",
                lambda arguments: seen.append(str(arguments["value"])) or arguments["value"],
            ),
        ),
    ).run(state)

    assert seen == ["one", "two"]


@pytest.mark.anyio
async def test_loop_passes_remaining_budget_to_provider() -> None:
    config = _config("loop-budget", budget=100)
    handles = TurnRegistry.get_or_create(config)
    assert handles.llm_budget_ledger is not None
    assert await handles.llm_budget_ledger.reserve("seed", 35)
    await handles.llm_budget_ledger.commit("seed", 35)
    provider = _SequenceProvider([ModelTurnDraft(action="finish", final_answer="done")])

    await _loop(provider=provider).run(create_loop_state(current_message="Answer.", run_config=config))

    assert provider.seen_budget_remaining == [65]
    TurnRegistry.remove(config.turn_id)


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("request_max_turns", "definition_max_iterations", "expected_reason"),
    [
        pytest.param(1, 5, "max_turns", id="request-limit"),
        pytest.param(5, 1, "max_iterations", id="definition-limit"),
    ],
)
async def test_effective_turn_limit_stops_before_another_model_turn(
    request_max_turns: int,
    definition_max_iterations: int,
    expected_reason: str,
) -> None:
    calls: list[str] = []
    first_call = ToolCallPlan.create("echo", {"value": "once"})
    provider = _SequenceProvider(
        [
            ModelTurnDraft(action="execute", tool_calls=(first_call,)),
            ModelTurnDraft(action="finish", final_answer="too late"),
        ]
    )
    state = create_loop_state(
        current_message="Use one turn only.",
        run_config=_config(
            f"loop-{expected_reason}",
            max_turns=request_max_turns,
        ),
    )
    state["resident_tool_names"] = ["echo"]

    result = await _loop(
        provider=provider,
        tools=(
            _tool(
                "echo",
                lambda arguments: calls.append(str(arguments["value"])) or arguments["value"],
            ),
        ),
        definition=_definition(
            ("echo",),
            max_iterations=definition_max_iterations,
        ),
    ).run(state)

    assert calls == ["once"]
    assert len(provider.seen_states) == 1
    assert result["iteration"] == 1
    assert result["status"] == "failed"
    assert result["terminal"] is not None
    assert result["terminal"].stop_reason == expected_reason


@pytest.mark.anyio
async def test_max_turns_stream_does_not_announce_an_unstarted_turn() -> None:
    call = ToolCallPlan.create("echo", {"value": "once"})
    provider = _SequenceProvider([ModelTurnDraft(action="execute", tool_calls=(call,))])
    state = create_loop_state(
        current_message="Emit only real model turns.",
        run_config=_config("loop-max-turn-events", max_turns=1),
    )
    state["resident_tool_names"] = ["echo"]

    events = await _collect(
        _loop(
            provider=provider,
            tools=(_tool("echo", lambda arguments: arguments["value"]),),
            definition=_definition(("echo",), max_iterations=5),
        ).run_streaming(state)
    )

    turn_events = [event for event in events if event.type is EventType.TURN_START]
    assert [event.iteration for event in turn_events] == [1]
    assert [event.turn_id for event in turn_events] == ["loop-max-turn-events"]
    assert all(not hasattr(event, "session_id") for event in turn_events)
    assert all(event.sequence > 0 for event in events)
    assert [event.sequence for event in events] == sorted(event.sequence for event in events)
    assert state["status"] == "failed"
    assert state["terminal"] is not None
    assert state["terminal"].stop_reason == "max_turns"


@pytest.mark.anyio
async def test_provider_error_retries_then_fails() -> None:
    provider = _SequenceProvider([RuntimeError("down"), RuntimeError("down")])
    result = await _loop(provider=provider, max_model_retries=1).run(
        create_loop_state(
            current_message="Answer.",
            run_config=_config("loop-provider-error"),
        )
    )

    assert result["status"] == "failed"
    assert result["terminal"] is not None
    assert result["terminal"].stop_reason == "model_provider_failed"


@pytest.mark.anyio
async def test_provider_error_redacts_credential_identifier_from_state() -> None:
    credential_id = "ak-provider-credential-123456"
    provider = _SequenceProvider(
        [RuntimeError(f"429 rate limit for <{credential_id}>")]
    )

    result = await _loop(provider=provider, max_model_retries=0).run(
        create_loop_state(
            current_message="Answer.",
            run_config=_config("loop-provider-secret-redaction"),
        )
    )

    assert result["terminal"] is not None
    assert credential_id not in (result["terminal"].error or "")
    assert "[REDACTED]" in (result["terminal"].error or "")
    assert all(
        credential_id not in item.message
        for item in result["runtime_diagnostics"]
    )
    assert result["latest_transition"] is not None
    assert credential_id not in result["latest_transition"].model_dump_json()


@pytest.mark.anyio
async def test_provider_tool_validation_retry_gives_model_corrective_context() -> None:
    provider = _SequenceProvider(
        [
            LLMToolCallValidationError(
                validation_error=("Tool call validation failed: max_bytes exceeds maximum"),
                failed_generation=('<function=read_file>{"path":"README.md","max_bytes":2000000}</function>'),
            ),
            ModelTurnDraft(action="finish", final_answer="recovered"),
        ]
    )

    result = await _loop(provider=provider, max_model_retries=1).run(
        create_loop_state(
            current_message="Read README.md.",
            run_config=_config("loop-provider-tool-validation"),
        )
    )

    assert result["status"] == "completed"
    retry_transcript = provider.seen_states[1]["turn_transcript"]
    feedback = retry_transcript[-1]
    assert feedback.role == "context"
    assert "model_tool_call_rejected" in feedback.content
    assert "max_bytes exceeds maximum" in feedback.content
    assert 'max_bytes\\":2000000' in feedback.content


@pytest.mark.anyio
async def test_provider_tool_validation_does_not_consume_transport_retry() -> None:
    provider = _SequenceProvider(
        [
            LLMToolCallValidationError(
                validation_error=(
                    "Tool call validation failed: timeout_seconds exceeds maximum"
                ),
                failed_generation=(
                    '<function=run_command>{"command":"pytest -q",'
                    '"timeout_seconds":1000}</function>'
                ),
            ),
            RuntimeError("request timed out"),
            ModelTurnDraft(action="finish", final_answer="recovered"),
        ]
    )

    result = await _loop(provider=provider, max_model_retries=1).run(
        create_loop_state(
            current_message="Run focused tests.",
            run_config=_config("loop-provider-tool-validation-timeout"),
        )
    )

    assert result["status"] == "completed"
    assert result["iteration"] == 3
    assert any(
        item.code == "model_tool_call_rejected"
        for item in result["runtime_diagnostics"]
    )


@pytest.mark.anyio
async def test_run_streaming_injects_sink_and_closes() -> None:
    provider = _SinkAwareProvider()
    events = await asyncio.wait_for(
        _collect(
            _loop(provider=provider).run_streaming(
                create_loop_state(
                    current_message="Stream.",
                    run_config=_config("loop-stream"),
                )
            )
        ),
        timeout=1,
    )

    text_event = next(event for event in events if event.type is EventType.TEXT_DELTA)
    assert text_event.turn_id == "loop-stream"
    assert not hasattr(text_event, "session_id")
    assert text_event.iteration == 1
    assert text_event.sequence > 0
    assert events[-1].type is EventType.LOOP_END


@pytest.mark.anyio
async def test_find_tools_result_and_activation_are_checkpointed_atomically() -> None:
    hidden = _tool("mcp__docs__search", lambda _arguments: "hidden")

    def search(_query: str, _limit: int) -> FindToolsOutput:
        return FindToolsOutput(
            query="documentation",
            matches=(
                FindToolMatch(
                    name=hidden.definition.name,
                    description=hidden.definition.description,
                    score=1.0,
                    matched_terms=("documentation",),
                ),
            ),
            proposed_activation_names=(hidden.definition.name,),
        )

    find_tool = create_find_tools_tool(search)
    snapshot = (find_tool, hidden)
    origin = ToolCallOrigin(
        request_id="request-find",
        toolset_revision="tools-find",
        exposed_tool_names=(find_tool.definition.name,),
    )
    call = ToolCall(
        tool_call_id="call-find",
        tool_name=find_tool.definition.name,
        arguments={"query": "documentation", "limit": 5},
        origin=origin,
    )
    state = create_loop_state(
        current_message="Find documentation.",
        run_config=_config("atomic-tool-activation"),
        pending_tool_calls=(
            ToolCallPlan(
                tool_call_id=call.tool_call_id,
                tool_name=call.tool_name,
                arguments=dict(call.arguments),
                origin=origin,
            ),
        ),
    )
    state["resident_tool_names"] = [find_tool.definition.name]
    state["canonical_tool_calls"] = {call.tool_call_id: call}
    checkpoint = _Checkpoint()

    result = await _loop(
        provider=_SequenceProvider([ModelTurnDraft(action="finish", final_answer="Found it.")]),
        tools=snapshot,
        checkpoint=checkpoint,
    ).run(state)

    assert result["active_tool_names"] == [hidden.definition.name]
    activation_snapshots = [
        snap for _reason, snap in checkpoint.snapshots if hidden.definition.name in snap["active_tool_names"]
    ]
    assert activation_snapshots
    assert all(
        any(item.tool_call_id == call.tool_call_id for item in snap["tool_results"]) for snap in activation_snapshots
    )


def test_first_evidence_batch_is_allowed_before_a_plan_exists() -> None:
    state = create_loop_state(
        current_message="Find the implementation choke point.",
        run_config=_config("loop-first-evidence-batch"),
    )
    origin = ToolCallOrigin(
        request_id="first-evidence-request",
        toolset_revision="first-evidence-tools",
        exposed_tool_names=("search_text", "read_file"),
    )
    calls = (
        ToolCall(
            tool_call_id="search-first",
            tool_name="search_text",
            arguments={"pattern": "AgentLoop", "path": "rag"},
            origin=origin,
        ),
        ToolCall(
            tool_call_id="read-first",
            tool_name="read_file",
            arguments={"path": "rag/agent/loop/runtime.py"},
            origin=origin,
        ),
    )

    executable, blocked = _guard_evidence_driven_progress(state, calls)

    assert executable == calls
    assert blocked == ()


def test_successful_evidence_requires_a_plan_before_more_inspection() -> None:
    state = create_loop_state(
        current_message="Deliver the cross-layer change.",
        run_config=_config("loop-plan-after-evidence"),
    )
    state["tool_results"] = [
        ToolResult(
            tool_call_id="search-success",
            tool_name="search_text",
        )
    ]
    origin = ToolCallOrigin(
        request_id="plan-after-evidence-request",
        toolset_revision="plan-after-evidence-tools",
        exposed_tool_names=("read_file", "apply_patch", "update_plan"),
    )
    read = ToolCall(
        tool_call_id="read-without-plan",
        tool_name="read_file",
        arguments={"path": "rag/agent/loop/runtime.py"},
        origin=origin,
    )
    patch = ToolCall(
        tool_call_id="patch-without-plan",
        tool_name="apply_patch",
        arguments={"file_path": "rag/agent/loop/runtime.py"},
        origin=origin,
    )
    update = ToolCall(
        tool_call_id="plan-now",
        tool_name="update_plan",
        arguments={
            "target_files": ["rag/agent/loop/runtime.py"],
            "hypothesis": (
                "The runtime guard must bind further inspection to an "
                "evidence-backed plan."
            ),
            "remaining_unknowns": [],
            "plan": [
                {
                    "step": "Read the runtime choke point",
                    "status": "in_progress",
                    "expected_tool_names": ["read_file"],
                }
            ]
        },
        origin=origin,
    )

    executable, blocked = _guard_evidence_driven_progress(
        state,
        (read, patch, update),
    )

    assert executable == (patch, update)
    assert [result.tool_call_id for result in blocked] == [read.tool_call_id]
    assert blocked[0].error_code == "planning_required"
    assert blocked[0].retryable is False


def test_empty_search_does_not_claim_that_evidence_was_found() -> None:
    state = create_loop_state(
        current_message="Try a different symbol after an empty search.",
        run_config=_config("loop-empty-search-is-not-evidence"),
    )
    state["tool_results"] = [
        ToolResult(
            tool_call_id="search-empty",
            tool_name="search_text",
            structured_content={
                "matches": [],
                "total_matches": 0,
                "truncated": False,
            },
        )
    ]
    origin = ToolCallOrigin(
        request_id="empty-search-request",
        toolset_revision="empty-search-tools",
        exposed_tool_names=("search_text",),
    )
    changed_query = ToolCall(
        tool_call_id="search-different-symbol",
        tool_name="search_text",
        arguments={"pattern": "AgentService", "path": "."},
        origin=origin,
    )

    executable, blocked = _guard_evidence_driven_progress(
        state,
        (changed_query,),
    )

    assert executable == (changed_query,)
    assert blocked == ()


def test_known_search_locator_allows_one_direct_read_followup() -> None:
    state = create_loop_state(
        current_message="Read the implementation found by search.",
        run_config=_config("loop-known-locator-followup"),
    )
    state["plan_state"].agent_plan = AgentPlan(
        objective="Find the implementation.",
        status="needs_replan",
        active_step_id="step_task",
        steps=[PlanStep(step_id="step_task", title="Work on the task.")],
    )
    state["memory_state"].known_locators = [
        {
            "source_tool": "search_text",
            "path": "rag/agent/loop/runtime.py",
            "line_number": 718,
        }
    ]
    state["tool_results"] = [
        ToolResult(
            tool_call_id="search-known-path",
            tool_name="search_text",
        )
    ]
    origin = ToolCallOrigin(
        request_id="known-locator-request",
        toolset_revision="known-locator-tools",
        exposed_tool_names=("read_file",),
    )
    known = ToolCall(
        tool_call_id="read-known-path",
        tool_name="read_file",
        arguments={"path": "rag/agent/loop/runtime.py"},
        origin=origin,
    )
    unrelated = ToolCall(
        tool_call_id="read-unrelated-path",
        tool_name="read_file",
        arguments={"path": "rag/agent/service.py"},
        origin=origin,
    )

    executable, blocked = _guard_evidence_driven_progress(
        state,
        (known, unrelated),
    )

    assert executable == (known,)
    assert [result.tool_call_id for result in blocked] == [
        unrelated.tool_call_id
    ]


def test_verified_workspace_paths_survive_lossy_locator_projection() -> None:
    state = create_loop_state(
        current_message="Keep verified file evidence after context compaction.",
        run_config=replace(
            _config("loop-durable-workspace-evidence"),
            memory_policy=MemoryPolicy(
                reactive_compact_max_evidence=2,
            ),
        ),
    )
    AgentLoop._merge_observations(
        state,
        ObservationBatch(
            locators=[
                {
                    "source_tool": "list_files",
                    "path": "agent_runtime/__init__.py",
                },
                {
                    "source_tool": "list_files",
                    "path": "agent_runtime/result.py",
                },
            ]
        ),
    )
    AgentLoop._merge_observations(
        state,
        ObservationBatch(
            locators=[
                {
                    "source_tool": "list_files",
                    "path": "agent_runtime/runtime/builder.py",
                },
                {
                    "source_tool": "list_files",
                    "path": "agent_runtime/runtime/mcp.py",
                },
            ]
        ),
    )

    assert [
        locator["path"] for locator in state["memory_state"].known_locators
    ] == [
        "agent_runtime/runtime/builder.py",
        "agent_runtime/runtime/mcp.py",
    ]
    assert state["memory_state"].verified_workspace_paths == [
        "agent_runtime/__init__.py",
        "agent_runtime/result.py",
        "agent_runtime/runtime/builder.py",
        "agent_runtime/runtime/mcp.py",
    ]

    state["tool_results"] = [
        ToolResult(
            tool_call_id="list-runtime",
            tool_name="list_files",
        )
    ]
    state["plan_state"].agent_plan = AgentPlan(
        objective="Read a previously verified file.",
        active_step_id="step_read",
        target_files=["agent_runtime/__init__.py"],
        steps=[
            PlanStep(
                step_id="step_read",
                title="Read agent_runtime/__init__.py.",
                status="in_progress",
                expected_tool_names=["read_file"],
            )
        ],
    )
    call = ToolCall(
        tool_call_id="read-evicted-locator",
        tool_name="read_file",
        arguments={"path": "agent_runtime/__init__.py"},
        origin=ToolCallOrigin(
            request_id="durable-evidence-request",
            toolset_revision="durable-evidence-tools",
            exposed_tool_names=("read_file",),
        ),
    )

    executable, blocked = _guard_evidence_driven_progress(state, (call,))

    assert executable == (call,)
    assert blocked == ()


def test_stale_advisory_plan_does_not_block_evidence_backed_read() -> None:
    state = create_loop_state(
        current_message="Use verified evidence even when the plan is stale.",
        run_config=_config("loop-advisory-plan-does-not-authorize"),
    )
    state["memory_state"].known_locators = [
        {
            "source_tool": "list_files",
            "path": "agent_runtime/__init__.py",
        }
    ]
    state["tool_results"] = [
        ToolResult(
            tool_call_id="list-agent-runtime",
            tool_name="list_files",
        )
    ]
    state["plan_state"].agent_plan = AgentPlan(
        objective="Inspect the public runtime.",
        active_step_id="step_list",
        target_files=["agent_runtime/__init__.py"],
        steps=[
            PlanStep(
                step_id="step_list",
                title="Locate the runtime file.",
                status="in_progress",
                expected_tool_names=["list_files"],
            )
        ],
    )
    call = ToolCall(
        tool_call_id="read-grounded-after-list",
        tool_name="read_file",
        arguments={"path": "agent_runtime/__init__.py"},
        origin=ToolCallOrigin(
            request_id="advisory-plan-request",
            toolset_revision="advisory-plan-tools",
            exposed_tool_names=("read_file",),
        ),
    )

    executable, blocked = _guard_evidence_driven_progress(state, (call,))

    assert executable == (call,)
    assert blocked == ()


def test_completed_advisory_plan_does_not_block_grounded_directory_descent() -> None:
    state = create_loop_state(
        current_message="Continue from the verified root listing.",
        run_config=_config("loop-completed-plan-grounded-descent"),
    )
    state["memory_state"].verified_workspace_paths = [
        "agent_runtime",
        "agent_runtime/runtime",
    ]
    state["tool_results"] = [
        ToolResult(
            tool_call_id="list-root",
            tool_name="list_files",
        )
    ]
    state["plan_state"].agent_plan = AgentPlan(
        objective="Locate the runtime implementation.",
        status="active",
        active_step_id=None,
        target_files=[],
        steps=[
            PlanStep(
                step_id="step_list_root",
                title="List the repository root.",
                status="completed",
                expected_tool_names=["list_files"],
                tool_call_ids=["list-root"],
            )
        ],
    )
    call = ToolCall(
        tool_call_id="list-grounded-runtime",
        tool_name="list_files",
        arguments={"path": "agent_runtime/runtime", "limit": 50},
        origin=ToolCallOrigin(
            request_id="completed-plan-request",
            toolset_revision="completed-plan-tools",
            exposed_tool_names=("list_files",),
        ),
    )

    executable, blocked = _guard_evidence_driven_progress(state, (call,))

    assert executable == (call,)
    assert blocked == ()


def test_durable_evidence_survives_a_non_inspection_cycle_boundary() -> None:
    state = create_loop_state(
        current_message="Continue from the verified root listing.",
        run_config=_config("loop-durable-evidence-crosses-cycle-boundary"),
    )
    state["memory_state"].verified_workspace_paths = ["agent_runtime"]
    state["tool_results"] = [
        ToolResult(
            tool_call_id="list-root",
            tool_name="list_files",
        ),
        ToolResult(
            tool_call_id="activate-unrelated-skill",
            tool_name="invoke_skill",
        ),
    ]
    state["plan_state"].agent_plan = AgentPlan(
        objective="Locate the runtime implementation.",
        status="needs_replan",
        active_step_id="step_task",
        steps=[
            PlanStep(
                step_id="step_task",
                title="Inspect the verified runtime directory.",
                status="in_progress",
            )
        ],
    )
    call = ToolCall(
        tool_call_id="list-grounded-runtime",
        tool_name="list_files",
        arguments={"path": "agent_runtime", "limit": 50},
        origin=ToolCallOrigin(
            request_id="durable-cycle-request",
            toolset_revision="durable-cycle-tools",
            exposed_tool_names=("list_files",),
        ),
    )

    executable, blocked = _guard_evidence_driven_progress(state, (call,))

    assert executable == (call,)
    assert blocked == ()


def test_durable_file_evidence_can_be_read_after_a_cycle_boundary() -> None:
    state = create_loop_state(
        current_message="Read the file already verified by the root listing.",
        run_config=_config("loop-durable-file-crosses-cycle-boundary"),
    )
    state["memory_state"].verified_workspace_paths = [
        "agent_runtime/__init__.py"
    ]
    state["tool_results"] = [
        ToolResult(
            tool_call_id="list-root",
            tool_name="list_files",
        ),
        ToolResult(
            tool_call_id="activate-unrelated-skill",
            tool_name="invoke_skill",
        ),
    ]
    state["plan_state"].agent_plan = AgentPlan(
        objective="Inspect the verified runtime file.",
        status="needs_replan",
        active_step_id="step_task",
        steps=[
            PlanStep(
                step_id="step_task",
                title="Inspect the verified runtime file.",
                status="in_progress",
            )
        ],
    )
    call = ToolCall(
        tool_call_id="read-grounded-runtime",
        tool_name="read_file",
        arguments={"path": "agent_runtime/__init__.py"},
        origin=ToolCallOrigin(
            request_id="durable-file-cycle-request",
            toolset_revision="durable-file-cycle-tools",
            exposed_tool_names=("read_file",),
        ),
    )

    executable, blocked = _guard_evidence_driven_progress(state, (call,))

    assert executable == (call,)
    assert blocked == ()


def test_inspection_guard_applies_the_published_default_workspace_path() -> None:
    state = create_loop_state(
        current_message="Search the verified workspace root.",
        run_config=_config("loop-inspection-default-path"),
    )
    state["memory_state"].verified_workspace_paths = ["."]
    state["tool_results"] = [
        ToolResult(
            tool_call_id="list-root",
            tool_name="list_files",
        ),
        ToolResult(
            tool_call_id="activate-unrelated-skill",
            tool_name="invoke_skill",
        ),
    ]
    state["plan_state"].agent_plan = AgentPlan(
        objective="Find the patch implementation.",
        status="needs_replan",
        active_step_id="step_task",
        steps=[
            PlanStep(
                step_id="step_task",
                title="Search the verified workspace.",
                status="in_progress",
            )
        ],
    )
    call = ToolCall(
        tool_call_id="search-default-root",
        tool_name="search_text",
        arguments={"pattern": "apply_patch"},
        origin=ToolCallOrigin(
            request_id="default-path-request",
            toolset_revision="default-path-tools",
            exposed_tool_names=("search_text",),
        ),
    )

    executable, blocked = _guard_evidence_driven_progress(state, (call,))

    assert executable == (call,)
    assert blocked == ()


def test_repeated_source_read_is_blocked_by_repetition_guard() -> None:
    state = create_loop_state(
        current_message="Stop reopening already acquired source.",
        run_config=_config("loop-locator-followup-closed"),
    )
    state["plan_state"].agent_plan = AgentPlan(
        objective="Find the implementation.",
        status="needs_replan",
        active_step_id="step_task",
        steps=[PlanStep(step_id="step_task", title="Work on the task.")],
    )
    state["memory_state"].known_locators = [
        {
            "source_tool": "read_file",
            "path": "rag/agent/loop/runtime.py",
        }
    ]
    state["tool_results"] = [
        ToolResult(
            tool_call_id="search-known-path",
            tool_name="search_text",
        ),
        ToolResult(
            tool_call_id="read-known-path",
            tool_name="read_file",
        ),
    ]
    origin = ToolCallOrigin(
        request_id="closed-locator-request",
        toolset_revision="closed-locator-tools",
        exposed_tool_names=("read_file",),
    )
    previous = ToolCall(
        tool_call_id="read-known-path",
        tool_name="read_file",
        arguments={"path": "rag/agent/loop/runtime.py"},
        origin=origin,
    )
    state["canonical_tool_calls"][previous.tool_call_id] = previous
    repeated = ToolCall(
        tool_call_id="read-known-path-again",
        tool_name="read_file",
        arguments={"path": "rag/agent/loop/runtime.py"},
        origin=origin,
    )

    planned, planning_blocks = _guard_evidence_driven_progress(
        state,
        (repeated,),
    )
    executable, blocked = _guard_repeated_successful_inspections(
        state,
        planned,
    )

    assert planning_blocks == ()
    assert executable == ()
    assert [result.tool_call_id for result in blocked] == [
        repeated.tool_call_id
    ]
    assert blocked[0].error_code == "repeated_inspection"


def test_active_plan_allows_only_expected_inspection_tool() -> None:
    state = create_loop_state(
        current_message="Follow the evidence-backed implementation plan.",
        run_config=_config("loop-plan-tool-binding"),
    )
    state["memory_state"].known_locators = [
        {
            "source_tool": "search_text",
            "path": "rag/agent/loop/runtime.py",
            "line_number": 718,
        }
    ]
    state["plan_state"].agent_plan = AgentPlan(
        objective="Deliver and verify.",
        active_step_id="step_read",
        target_files=["rag/agent/loop/runtime.py"],
        steps=[
            PlanStep(
                step_id="step_read",
                title="Read the exact runtime choke point.",
                status="in_progress",
                expected_tool_names=["read_file"],
            )
        ],
    )
    origin = ToolCallOrigin(
        request_id="plan-tool-binding-request",
        toolset_revision="plan-tool-binding-tools",
        exposed_tool_names=("read_file", "search_text"),
    )
    expected = ToolCall(
        tool_call_id="read-planned",
        tool_name="read_file",
        arguments={"path": "rag/agent/loop/runtime.py"},
        origin=origin,
    )
    drift = ToolCall(
        tool_call_id="search-unplanned",
        tool_name="search_text",
        arguments={"pattern": "Loop", "path": ""},
        origin=origin,
    )

    executable, blocked = _guard_evidence_driven_progress(
        state,
        (expected, drift),
    )

    assert executable == (expected,)
    assert [result.tool_call_id for result in blocked] == [drift.tool_call_id]
    assert blocked[0].error_code == "planning_required"
    assert blocked[0].metadata["expected_tool_names"] == ("read_file",)


def test_active_plan_cannot_turn_a_model_claim_into_grounded_evidence() -> None:
    state = create_loop_state(
        current_message="Do not trust a claimed path.",
        run_config=_config("loop-plan-claim-is-advisory"),
    )
    state["tool_results"] = [
        ToolResult(
            tool_call_id="search-other-file",
            tool_name="search_text",
        )
    ]
    state["memory_state"].known_locators = [
        {
            "source_tool": "search_text",
            "path": "rag/agent/service.py",
            "line_number": 10,
        }
    ]
    state["plan_state"].agent_plan = AgentPlan(
        objective="Deliver and verify.",
        active_step_id="step_read",
        target_files=["rag/agent/loop/runtime.py"],
        hypothesis="The model claims that this file contains the defect.",
        steps=[
            PlanStep(
                step_id="step_read",
                title="Read the claimed runtime file.",
                status="in_progress",
                expected_tool_names=["read_file"],
            )
        ],
    )
    origin = ToolCallOrigin(
        request_id="plan-claim-request",
        toolset_revision="plan-claim-tools",
        exposed_tool_names=("read_file",),
    )
    claimed = ToolCall(
        tool_call_id="read-claimed-path",
        tool_name="read_file",
        arguments={"path": "rag/agent/loop/runtime.py"},
        origin=origin,
    )

    executable, blocked = _guard_evidence_driven_progress(
        state,
        (claimed,),
    )

    assert executable == ()
    assert [result.tool_call_id for result in blocked] == [
        claimed.tool_call_id
    ]
    assert blocked[0].error_code == "planning_evidence_required"
    assert blocked[0].metadata["unverified_path"] == (
        "rag/agent/loop/runtime.py"
    )


def test_active_plan_can_request_one_bounded_discovery_after_a_direct_read() -> None:
    state = create_loop_state(
        current_message="Discover the cross-layer caller.",
        run_config=_config("loop-plan-bounded-discovery"),
    )
    state["tool_results"] = [
        ToolResult(
            tool_call_id="read-entrypoint",
            tool_name="read_file",
        )
    ]
    state["memory_state"].known_locators = [
        {
            "source_tool": "read_file",
            "path": "rag/agent/cli.py",
        }
    ]
    state["plan_state"].agent_plan = AgentPlan(
        objective="Trace the public entrypoint into the runtime.",
        active_step_id="step_search",
        target_files=["rag/agent/cli.py"],
        hypothesis="The CLI delegates the behavior to another runtime layer.",
        steps=[
            PlanStep(
                step_id="step_search",
                title="Search once for the delegated symbol.",
                status="in_progress",
                expected_tool_names=["search_text"],
            )
        ],
    )
    origin = ToolCallOrigin(
        request_id="bounded-discovery-request",
        toolset_revision="bounded-discovery-tools",
        exposed_tool_names=("search_text",),
    )
    discovery = ToolCall(
        tool_call_id="search-cross-layer-symbol",
        tool_name="search_text",
        arguments={"pattern": "AgentService", "path": "."},
        origin=origin,
    )

    executable, blocked = _guard_evidence_driven_progress(
        state,
        (discovery,),
    )

    assert executable == (discovery,)
    assert blocked == ()


@pytest.mark.parametrize(
    "result",
    [
        ToolResult(
            tool_call_id="failed-check",
            tool_name="run_command",
            structured_content={
                "exit_code": 1,
                "timed_out": False,
                "sandbox_error": None,
            },
        ),
        ToolResult(
            tool_call_id="noop-patch",
            tool_name="apply_patch",
        ),
    ],
)
def test_plan_completion_rejects_unsuccessful_delivery_evidence(
    result: ToolResult,
) -> None:
    claimed = _claim_matching_plan_evidence(
        [result],
        expected_tool_names=[result.tool_name],
        used_tool_call_ids=set(),
    )

    assert claimed == []


def test_needs_replan_blocks_further_evidence_but_not_delivery() -> None:
    state = create_loop_state(
        current_message="Recover from plan drift.",
        run_config=_config("loop-needs-replan"),
    )
    state["plan_state"].agent_plan = AgentPlan(
        objective="Deliver and verify.",
        status="needs_replan",
        active_step_id="step_edit",
        steps=[
            PlanStep(
                step_id="step_edit",
                title="Edit the runtime.",
                status="in_progress",
                expected_tool_names=["apply_patch"],
            )
        ],
    )
    origin = ToolCallOrigin(
        request_id="needs-replan-request",
        toolset_revision="needs-replan-tools",
        exposed_tool_names=("read_file", "apply_patch"),
    )
    read = ToolCall(
        tool_call_id="read-after-drift",
        tool_name="read_file",
        arguments={"path": "rag/agent/loop/runtime.py"},
        origin=origin,
    )
    patch = ToolCall(
        tool_call_id="patch-after-drift",
        tool_name="apply_patch",
        arguments={"file_path": "rag/agent/loop/runtime.py"},
        origin=origin,
    )

    executable, blocked = _guard_evidence_driven_progress(
        state,
        (read, patch),
    )

    assert executable == (patch,)
    assert [result.tool_call_id for result in blocked] == [read.tool_call_id]
    assert blocked[0].error_code == "planning_required"
    assert blocked[0].metadata["plan_status"] == "needs_replan"


def test_real_workspace_change_reopens_one_verification_batch() -> None:
    state = create_loop_state(
        current_message="Verify the concrete edit.",
        run_config=_config("loop-verify-after-direct-edit"),
    )
    state["plan_state"].agent_plan = AgentPlan(
        objective="Deliver and verify.",
        status="needs_replan",
        active_step_id="step_task",
        steps=[
            PlanStep(
                step_id="step_task",
                title="Work on the current task.",
                status="in_progress",
            )
        ],
    )
    state["tool_results"] = [
        ToolResult(
            tool_call_id="search-before-edit",
            tool_name="search_text",
        ),
        ToolResult(
            tool_call_id="patch-real-change",
            tool_name="apply_patch",
            metadata={
                "workspace_changed": True,
                "file_path": "src/example.py",
                "before_sha256": "a" * 64,
                "after_sha256": "b" * 64,
            },
        ),
    ]
    origin = ToolCallOrigin(
        request_id="verify-after-edit-request",
        toolset_revision="verify-after-edit-tools",
        exposed_tool_names=("run_command",),
    )
    verify = ToolCall(
        tool_call_id="verify-after-edit",
        tool_name="run_command",
        arguments={"command": "pytest -q"},
        origin=origin,
    )

    executable, blocked = _guard_evidence_driven_progress(state, (verify,))

    assert executable == (verify,)
    assert blocked == ()


@pytest.mark.anyio
async def test_goal_contract_rejects_the_observed_kimi_off_goal_plan() -> None:
    goal_text = (
        "This is an implementation task in the current repository. Modify the "
        "code and run focused tests. Successful patch application should produce "
        "a canonical diff event that flows from the filesystem tool through the "
        "loop and appears once in the public CLI. Preserve existing tool-result "
        "and answer streaming behavior."
    )
    goal = GoalSpec(
        original_query=goal_text,
        constraints=[
            GoalConstraint(
                constraint_id="workspace_change",
                constraint_type="workspace_change",
                expected_value=True,
            )
        ],
    )
    contract = GoalPlanContract.from_goal_spec(goal)
    off_goal_update = ToolCallPlan.create(
        "update_plan",
        {
            "goal_id": contract.goal_id,
            "objective": goal_text,
            "target_files": [],
            "hypothesis": (
                "The specific implementation task is not yet identified. I need "
                "to explore the repository and find any failing test."
            ),
            "remaining_unknowns": [
                "What is the specific implementation task or bug to fix?",
                "Which files need to be modified?",
            ],
            "plan": [
                {
                    "step": (
                        "Explore repository structure and identify the "
                        "implementation task"
                        ),
                        "status": "in_progress",
                        "goal_commitment_ids": [
                            contract.commitment_ids[0]
                        ],
                        "expected_tool_names": ["list_files"],
                    },
                    {
                        "step": "Run tests to find an unrelated failure",
                        "status": "pending",
                        "goal_commitment_ids": [
                            contract.commitment_ids[0]
                        ],
                        "expected_tool_names": ["run_command"],
                    },
            ],
        },
    )
    update_plan = create_update_plan_tool(
        lambda _arguments: {
            "accepted": True,
            "revision": 1,
            "message": "plan updated",
        }
    )
    state = create_loop_state(
        current_message=goal_text,
        run_config=_config("loop-goal-drift-kimi"),
    )
    state["resident_tool_names"] = ["update_plan"]

    result = await _loop(
        provider=_SequenceProvider(
            [
                ModelTurnDraft(action="execute", tool_calls=(off_goal_update,)),
                ModelTurnDraft(
                    action="pause",
                    pause_reason="The immutable goal rejected this plan.",
                ),
            ]
        ),
        tools=(update_plan,),
        goal_spec=goal,
    ).run(state)

    plan = result["plan_state"].agent_plan
    assert plan is not None
    assert plan.objective == goal_text
    assert plan.revision == 0
    assert plan.steps[0].title == "Work on the current task."
    rejected = next(
        item
        for item in result["tool_results"]
        if item.tool_call_id == off_goal_update.tool_call_id
    )
    assert rejected.is_error is True
    assert rejected.error_code == "goal_drift"
    assert any(
        str(issue).startswith("missing_goal_commitment:")
        for issue in rejected.metadata["goal_contract_issues"]
    )
    assert any(
        item.code == "goal_drift"
        and item.component == "goal_plan_contract"
        for item in result["runtime_diagnostics"]
    )


@pytest.mark.anyio
async def test_goal_contract_fails_closed_when_resume_uses_a_different_goal() -> None:
    original = GoalSpec(original_query="Implement canonical diff events.")
    replacement = GoalSpec(original_query="Rewrite unrelated approval output.")
    original_contract = GoalPlanContract.from_goal_spec(original)
    state = create_loop_state(
        current_message=original.original_query,
        run_config=_config("loop-goal-drift-resume"),
    )
    state["plan_state"].agent_plan = AgentPlan(
        goal_id=original_contract.goal_id,
        objective=original.original_query,
        active_step_id="step_implement",
        steps=[
            PlanStep(
                step_id="step_implement",
                title="Implement canonical diff events.",
                status="in_progress",
                expected_tool_names=["apply_patch"],
            )
        ],
    )
    provider = _SequenceProvider(
        [
            ModelTurnDraft(
                action="pause",
                pause_reason="A mismatched checkpoint must never reach the model.",
            )
        ]
    )

    result = await _loop(
        provider=provider,
        goal_spec=replacement,
    ).run(state)

    assert result["status"] == "failed"
    assert result["terminal"] is not None
    assert result["terminal"].stop_reason == "goal_drift"
    assert provider.seen_states == []
    assert any(
        item.code == "goal_drift"
        and item.component == "goal_plan_contract"
        for item in result["runtime_diagnostics"]
    )


@pytest.mark.anyio
async def test_goal_contract_fails_closed_when_checkpoint_rewrites_a_commitment() -> None:
    goal = GoalSpec(original_query="Implement canonical diff events.")
    contract = GoalPlanContract.from_goal_spec(goal)
    rewritten = GoalCommitment(
        commitment_id=contract.commitment_ids[0],
        requirement="Rewrite unrelated approval output.",
    )
    state = create_loop_state(
        current_message=goal.original_query,
        run_config=_config("loop-goal-commitment-rewrite"),
    )
    state["plan_state"].agent_plan = AgentPlan(
        goal_id=contract.goal_id,
        goal_commitments=[rewritten],
        objective=goal.original_query,
        active_step_id="step_rewrite",
        steps=[
            PlanStep(
                step_id="step_rewrite",
                title="Rewrite unrelated approval output.",
                status="in_progress",
                goal_commitment_ids=[rewritten.commitment_id],
                expected_tool_names=["apply_patch"],
            )
        ],
    )
    provider = _SequenceProvider(
        [
            ModelTurnDraft(
                action="pause",
                pause_reason="A rewritten commitment must not reach the model.",
            )
        ]
    )

    result = await _loop(
        provider=provider,
        goal_spec=goal,
    ).run(state)

    assert result["status"] == "failed"
    assert result["terminal"] is not None
    assert result["terminal"].stop_reason == "goal_drift"
    assert provider.seen_states == []
    assert "goal_commitments_mismatch" in (
        result["terminal"].error or ""
    )


@pytest.mark.anyio
async def test_goal_contract_migrates_a_matching_legacy_checkpoint() -> None:
    goal = GoalSpec(original_query="Implement canonical diff events.")
    contract = GoalPlanContract.from_goal_spec(goal)
    state = create_loop_state(
        current_message=goal.original_query,
        run_config=_config("loop-goal-contract-legacy-checkpoint"),
    )
    state["plan_state"].agent_plan = AgentPlan(
        objective=goal.original_query,
        active_step_id="step_implement",
        steps=[
            PlanStep(
                step_id="step_implement",
                title="Implement canonical diff events.",
                status="in_progress",
                expected_tool_names=["apply_patch"],
            )
        ],
    )

    result = await _loop(
        provider=_SequenceProvider(
            [
                ModelTurnDraft(
                    action="pause",
                    pause_reason="Inspect the migrated checkpoint.",
                )
            ]
        ),
        goal_spec=goal,
    ).run(state)

    plan = result["plan_state"].agent_plan
    assert result["status"] == "paused"
    assert plan is not None
    assert plan.goal_id == contract.goal_id
    assert plan.objective == goal.original_query
    assert not any(
        item.code == "goal_drift"
        for item in result["runtime_diagnostics"]
    )


@pytest.mark.anyio
async def test_goal_contract_allows_strategy_changes_for_the_same_goal() -> None:
    goal = GoalSpec(original_query="Implement canonical diff events end to end.")
    contract = GoalPlanContract.from_goal_spec(goal)

    def update(
        *,
        target_file: str,
        hypothesis: str,
        step: str,
        bind_goal: bool = True,
    ) -> ToolCallPlan:
        goal_binding = (
            {
                "goal_id": contract.goal_id,
                "objective": goal.original_query,
            }
            if bind_goal
            else {}
        )
        return ToolCallPlan.create(
            "update_plan",
            {
                **goal_binding,
                "target_files": [target_file],
                "hypothesis": hypothesis,
                "remaining_unknowns": [],
                "plan": [
                    {
                        "step": step,
                        "status": "in_progress",
                        "goal_commitment_ids": list(
                            contract.commitment_ids
                        ),
                        "expected_tool_names": ["apply_patch"],
                    }
                ],
            },
        )

    unbound = update(
        target_file="rag/agent/tools/builtins/filesystem.py",
        hypothesis=(
            "The filesystem patch result must expose the canonical diff payload."
        ),
        step="Add the canonical diff payload at the filesystem boundary",
        bind_goal=False,
    )
    first = update(
        target_file="rag/agent/tools/builtins/filesystem.py",
        hypothesis=(
            "The filesystem patch result must expose the canonical diff payload."
        ),
        step="Add the canonical diff payload at the filesystem boundary",
    )
    second = update(
        target_file="rag/agent/loop/runtime.py",
        hypothesis=(
            "The filesystem payload already exists, so the loop must project it "
            "without duplication."
        ),
        step="Project the existing diff payload once from the loop",
    )
    update_plan = create_update_plan_tool(
        lambda _arguments: {
            "accepted": True,
            "revision": 1,
            "message": "plan updated",
        }
    )
    state = create_loop_state(
        current_message=goal.original_query,
        run_config=_config("loop-goal-strategy-change"),
    )
    state["resident_tool_names"] = ["update_plan"]

    result = await _loop(
        provider=_SequenceProvider(
            [
                ModelTurnDraft(action="execute", tool_calls=(unbound,)),
                ModelTurnDraft(action="execute", tool_calls=(first,)),
                ModelTurnDraft(action="execute", tool_calls=(second,)),
                ModelTurnDraft(
                    action="pause",
                    pause_reason="Inspect the accepted strategy change.",
                ),
            ]
        ),
        tools=(update_plan,),
        goal_spec=goal,
    ).run(state)

    plan = result["plan_state"].agent_plan
    assert plan is not None
    assert plan.goal_id == contract.goal_id
    assert plan.objective == goal.original_query
    assert plan.revision == 2
    assert plan.target_files == ["rag/agent/loop/runtime.py"]
    assert plan.steps[0].title == (
        "Project the existing diff payload once from the loop"
    )
    assert len(
        [
            item
            for item in result["tool_results"]
            if item.tool_name == "update_plan"
            and not item.is_error
        ]
    ) == 2
    assert any(
        item.code == "goal_drift_recovered"
        and item.component == "goal_plan_contract"
        for item in result["runtime_diagnostics"]
    )
    assert all(
        item.error_code in {None, "goal_drift"}
        for item in result["tool_results"]
        if item.tool_name == "update_plan"
    )


@pytest.mark.anyio
async def test_runtime_turns_initial_evidence_into_an_executable_plan_gate() -> None:
    attempts: list[str] = []

    def inspect(arguments: Mapping[str, JsonValue]) -> str:
        value = str(arguments["value"])
        attempts.append(value)
        return value

    inspection_schema: dict[str, JsonValue] = {
        "type": "object",
        "properties": {
            "value": {"type": "string"},
            "path": {"type": "string"},
        },
        "required": ["value", "path"],
        "additionalProperties": False,
    }
    runtime_path = "rag/agent/loop/runtime.py"
    first = ToolCallPlan.create(
        "search_text",
        {"value": "initial evidence", "path": runtime_path},
    )
    unplanned = ToolCallPlan.create(
        "read_file",
        {"value": "unplanned read", "path": "rag/agent/service.py"},
    )
    update = ToolCallPlan.create(
        "update_plan",
        {
            "explanation": "Use the discovered runtime choke point.",
            "target_files": ["rag/agent/loop/runtime.py"],
            "hypothesis": (
                "Binding the read result to a concrete step will prevent "
                "unbounded rediscovery."
            ),
            "remaining_unknowns": [
                "The exact edit required after reading the choke point."
            ],
            "plan": [
                {
                    "step": "Locate the runtime choke point",
                    "status": "completed",
                    "expected_tool_names": ["search_text"],
                },
                {
                    "step": "Read the exact runtime choke point",
                    "status": "in_progress",
                    "expected_tool_names": ["read_file"],
                },
                {
                    "step": "Implement the fix",
                    "status": "pending",
                    "expected_tool_names": ["apply_patch"],
                },
            ],
        },
    )
    planned = ToolCallPlan.create(
        "read_file",
        {"value": "planned read", "path": runtime_path},
    )
    state = create_loop_state(
        current_message="Deliver the cross-layer runtime fix.",
        run_config=_config("loop-evidence-plan-gate"),
        pending_tool_calls=(first,),
    )
    state["resident_tool_names"] = [
        "search_text",
        "read_file",
        "update_plan",
    ]
    state["memory_state"].known_locators = [
        {
            "source_tool": "search_text",
            "path": runtime_path,
            "line_number": 1,
        }
    ]
    update_plan = create_update_plan_tool(
        lambda _arguments: {
            "accepted": True,
            "revision": 1,
            "message": "plan updated",
        }
    )

    result = await _loop(
        provider=_SequenceProvider(
            [
                ModelTurnDraft(action="execute", tool_calls=(unplanned,)),
                ModelTurnDraft(action="execute", tool_calls=(update,)),
                ModelTurnDraft(action="execute", tool_calls=(planned,)),
                ModelTurnDraft(
                    action="pause",
                    pause_reason="Inspect the evidence-bound state.",
                ),
            ]
        ),
        tools=(
            _tool("search_text", inspect, schema=inspection_schema),
            _tool("read_file", inspect, schema=inspection_schema),
            update_plan,
        ),
    ).run(state)

    assert result["status"] == "paused"
    assert attempts == ["initial evidence", "planned read"]
    assert [
        item.error_code for item in result["tool_results"]
    ] == [None, "planning_required", None, None]
    assert result["plan_state"].agent_plan is not None
    assert [
        step.status for step in result["plan_state"].agent_plan.steps
    ] == ["completed", "completed", "pending"]
    assert (
        result["plan_state"].agent_plan.steps[0].tool_call_ids
        == [first.tool_call_id]
    )


def test_repeated_successful_inspection_requires_new_arguments() -> None:
    state = create_loop_state(
        current_message="Find the implementation choke point.",
        run_config=_config("loop-repeated-successful-inspection"),
    )
    origin = ToolCallOrigin(
        request_id="inspection-request",
        toolset_revision="inspection-tools",
        exposed_tool_names=("search_text",),
    )
    previous = ToolCall(
        tool_call_id="search-previous",
        tool_name="search_text",
        arguments={"pattern": "system", "path": ""},
        origin=origin,
    )
    repeated = ToolCall(
        tool_call_id="search-repeated",
        tool_name="search_text",
        arguments={"pattern": "system", "path": ""},
        origin=origin,
    )
    narrowed = ToolCall(
        tool_call_id="search-narrowed",
        tool_name="search_text",
        arguments={"pattern": "system", "path": "rag/providers"},
        origin=origin,
    )
    state["canonical_tool_calls"][previous.tool_call_id] = previous
    state["tool_results"] = [
        ToolResult(
            tool_call_id=previous.tool_call_id,
            tool_name=previous.tool_name,
        )
    ]

    executable, blocked = _guard_repeated_successful_inspections(
        state,
        (repeated, narrowed),
    )

    assert executable == (narrowed,)
    assert len(blocked) == 1
    assert blocked[0].tool_call_id == repeated.tool_call_id
    assert blocked[0].error_code == "repeated_inspection"
    assert blocked[0].retryable is False
    assert blocked[0].metadata["previous_tool_call_id"] == previous.tool_call_id


def test_delivery_action_reopens_same_inspection() -> None:
    state = create_loop_state(
        current_message="Verify the delivered change.",
        run_config=_config("loop-inspection-after-delivery"),
    )
    origin = ToolCallOrigin(
        request_id="inspection-after-delivery-request",
        toolset_revision="inspection-after-delivery-tools",
        exposed_tool_names=("read_file",),
    )
    previous = ToolCall(
        tool_call_id="read-before-patch",
        tool_name="read_file",
        arguments={"path": "src/example.py"},
        origin=origin,
    )
    verification = ToolCall(
        tool_call_id="read-after-patch",
        tool_name="read_file",
        arguments={"path": "src/example.py"},
        origin=origin,
    )
    state["canonical_tool_calls"][previous.tool_call_id] = previous
    state["tool_results"] = [
        ToolResult(
            tool_call_id=previous.tool_call_id,
            tool_name=previous.tool_name,
        ),
        ToolResult(
            tool_call_id="patch",
            tool_name="apply_patch",
            metadata={
                "workspace_changed": True,
                "file_path": "src/example.py",
                "before_sha256": "a" * 64,
                "after_sha256": "b" * 64,
            },
        ),
    ]

    executable, blocked = _guard_repeated_successful_inspections(
        state,
        (verification,),
    )

    assert executable == (verification,)
    assert blocked == ()


@pytest.mark.anyio
async def test_planning_gate_preserves_explicit_refinement_path() -> None:
    attempts: list[str] = []

    def inspect(arguments: Mapping[str, JsonValue]) -> str:
        value = str(arguments["value"])
        attempts.append(value)
        return value

    inspection_schema: dict[str, JsonValue] = {
        "type": "object",
        "properties": {
            "value": {"type": "string"},
            "path": {"type": "string"},
        },
        "required": ["value", "path"],
        "additionalProperties": False,
    }
    runtime_path = "rag/agent/loop/runtime.py"
    first = ToolCallPlan.create(
        "search_text",
        {"value": "broad", "path": runtime_path},
    )
    repeated = ToolCallPlan.create(
        "search_text",
        {"value": "broad", "path": runtime_path},
    )
    update = ToolCallPlan.create(
        "update_plan",
        {
            "explanation": "Refine the broad result at the concrete location.",
            "target_files": ["rag/agent/loop/runtime.py"],
            "hypothesis": (
                "A narrowed search at the known runtime location will resolve "
                "the remaining implementation question."
            ),
            "remaining_unknowns": [
                "The exact guard branch that needs modification."
            ],
            "plan": [
                {
                    "step": "Search the narrowed implementation location",
                    "status": "in_progress",
                    "expected_tool_names": ["search_text"],
                }
            ],
        },
    )
    narrowed = ToolCallPlan.create(
        "search_text",
        {"value": "narrowed", "path": runtime_path},
    )
    state = create_loop_state(
        current_message="Find the exact implementation and deliver the fix.",
        run_config=_config("loop-repeated-inspection-refinement"),
        pending_tool_calls=(first,),
    )
    state["resident_tool_names"] = ["search_text", "update_plan"]
    state["memory_state"].known_locators = [
        {
            "source_tool": "search_text",
            "path": runtime_path,
            "line_number": 1,
        }
    ]
    update_plan = create_update_plan_tool(
        lambda _arguments: {
            "accepted": True,
            "revision": 1,
            "message": "plan updated",
        }
    )

    result = await _loop(
        provider=_SequenceProvider(
                [
                    ModelTurnDraft(action="execute", tool_calls=(repeated,)),
                    ModelTurnDraft(action="execute", tool_calls=(update,)),
                    ModelTurnDraft(action="execute", tool_calls=(narrowed,)),
                    ModelTurnDraft(action="finish", final_answer="Inspected."),
                ]
            ),
            tools=(
                _tool("search_text", inspect, schema=inspection_schema),
                update_plan,
            ),
        ).run(state)

    assert result["status"] == "completed"
    assert attempts == ["broad", "narrowed"]
    assert [
        item.error_code for item in result["tool_results"]
    ] == [None, "repeated_inspection", None, None]


@pytest.mark.anyio
async def test_novel_grounded_inspection_is_not_capped_by_prior_call_count() -> None:
    attempts: list[str] = []
    call = ToolCallPlan.create(
        "read_file",
        {"path": "src/novel.py"},
    )
    state = create_loop_state(
        current_message="Keep following novel grounded evidence.",
        run_config=_config("loop-no-global-inspection-count"),
        pending_tool_calls=(call,),
    )
    state["resident_tool_names"] = ["read_file"]
    state["memory_state"].verified_workspace_paths = ["src/novel.py"]
    state["tool_results"] = [
        ToolResult(tool_call_id=f"seed-{index}", tool_name="read_file")
        for index in range(25)
    ]
    schema: dict[str, JsonValue] = {
        "type": "object",
        "properties": {"path": {"type": "string"}},
        "required": ["path"],
        "additionalProperties": False,
    }

    result = await _loop(
        provider=_SequenceProvider(
            [ModelTurnDraft(action="finish", final_answer="Inspected.")]
        ),
        tools=(
            _tool(
                "read_file",
                lambda arguments: (
                    attempts.append(str(arguments["path"]))
                    or arguments["path"]
                ),
                schema=schema,
            ),
        ),
    ).run(state)

    assert result["status"] == "completed"
    assert attempts == ["src/novel.py"]
    assert result["tool_results"][-1].error_code is None


@pytest.mark.anyio
async def test_repeated_retryable_tool_failure_is_circuited_and_can_recover() -> None:
    attempts: list[str] = []

    def flaky(arguments: Mapping[str, JsonValue]) -> str:
        value = str(arguments["value"])
        attempts.append(value)
        if value == "stuck":
            raise RuntimeError("still stuck")
        return value

    first = ToolCallPlan.create("flaky", {"value": "stuck"})
    second = ToolCallPlan.create("flaky", {"value": "stuck"})
    third = ToolCallPlan.create("flaky", {"value": "stuck"})
    recovery = ToolCallPlan.create("flaky", {"value": "recovered"})
    provider = _SequenceProvider(
        [
            ModelTurnDraft(action="execute", tool_calls=(second,)),
            ModelTurnDraft(action="execute", tool_calls=(third,)),
            ModelTurnDraft(action="execute", tool_calls=(recovery,)),
            ModelTurnDraft(action="finish", final_answer="Recovered."),
        ]
    )
    state = create_loop_state(
        current_message="Recover from a repeated tool failure.",
        run_config=_config("loop-repeated-tool-failure"),
        pending_tool_calls=(first,),
    )
    state["resident_tool_names"] = ["flaky"]

    events = await _collect(_loop(provider=provider, tools=(_tool("flaky", flaky),)).run_streaming(state))

    assert state["status"] == "completed"
    assert attempts == ["stuck", "stuck", "recovered"]
    assert [result.error_code for result in state["tool_results"]] == [
        "runner_failed",
        "runner_failed",
        "repeated_tool_failure",
        None,
    ]
    recovery_event = next(
        event
        for event in events
        if event.type is EventType.RECOVERY and event.data.get("strategy") == "tool_failure_circuit_breaker"
    )
    assert "flaky" in str(recovery_event.data["detail"])


@pytest.mark.anyio
async def test_repeated_tool_failure_circuit_uses_checkpointed_history() -> None:
    attempts: list[str] = []

    def flaky(arguments: Mapping[str, JsonValue]) -> str:
        value = str(arguments["value"])
        attempts.append(value)
        if value == "stuck":
            raise RuntimeError("still stuck")
        return value

    tool = _tool("flaky", flaky)
    first = ToolCallPlan.create("flaky", {"value": "stuck"})
    second = ToolCallPlan.create("flaky", {"value": "stuck"})
    state = create_loop_state(
        current_message="Pause after repeated failures.",
        run_config=_config("loop-repeated-tool-failure-resume"),
        pending_tool_calls=(first,),
    )
    state["resident_tool_names"] = ["flaky"]
    state["tool_manifest"] = build_tool_manifest(
        tools=(tool,),
        resident_tool_names=("flaky",),
        explicit_tool_names=(),
        active_tool_names=(),
        provider_serializer_revision=state["provider_serializer_revision"],
    )
    checkpoint = LangGraphCheckpointStore(
        MemorySaver(serde=agent_checkpoint_serde()),
        run_config=state["run_config"],
    )

    paused = await _loop(
        provider=_SequenceProvider(
            [
                ModelTurnDraft(action="execute", tool_calls=(second,)),
                ModelTurnDraft(action="pause", pause_reason="Resume later."),
            ]
        ),
        tools=(tool,),
        checkpoint=checkpoint,
    ).run(state)

    assert paused["status"] == "paused"
    resumed = await checkpoint.load_latest()
    assert resumed is not None
    resumed["status"] = "running"
    resumed["pause"] = None
    third = ToolCallPlan.create("flaky", {"value": "stuck"})
    recovery = ToolCallPlan.create("flaky", {"value": "recovered"})

    result = await _loop(
        provider=_SequenceProvider(
            [
                ModelTurnDraft(action="execute", tool_calls=(third,)),
                ModelTurnDraft(action="execute", tool_calls=(recovery,)),
                ModelTurnDraft(action="finish", final_answer="Recovered."),
            ]
        ),
        tools=(tool,),
    ).run(resumed)

    assert result["status"] == "completed"
    assert attempts == ["stuck", "stuck", "recovered"]
    assert result["tool_results"][-2].error_code == "repeated_tool_failure"


@pytest.mark.anyio
async def test_repeating_an_open_tool_failure_circuit_fails_fast() -> None:
    attempts = 0

    def always_fails(_arguments: Mapping[str, JsonValue]) -> str:
        nonlocal attempts
        attempts += 1
        raise RuntimeError("still stuck")

    calls = tuple(ToolCallPlan.create("flaky", {"value": "stuck"}) for _ in range(4))
    state = create_loop_state(
        current_message="Stop a repeated failure loop.",
        run_config=_config("loop-repeated-tool-failure-terminal"),
        pending_tool_calls=(calls[0],),
    )
    state["resident_tool_names"] = ["flaky"]

    result = await _loop(
        provider=_SequenceProvider(
            [
                ModelTurnDraft(action="execute", tool_calls=(calls[1],)),
                ModelTurnDraft(action="execute", tool_calls=(calls[2],)),
                ModelTurnDraft(action="execute", tool_calls=(calls[3],)),
            ]
        ),
        tools=(_tool("flaky", always_fails),),
    ).run(state)

    assert attempts == 2
    assert result["status"] == "failed"
    assert result["terminal"] is not None
    assert result["terminal"].stop_reason == "repeated_tool_failure"


@pytest.mark.anyio
async def test_alternating_failed_calls_do_not_evade_the_circuit() -> None:
    attempts: list[str] = []

    def flaky(arguments: Mapping[str, JsonValue]) -> str:
        value = str(arguments["value"])
        attempts.append(value)
        if value != "recovered":
            raise RuntimeError("still stuck")
        return value

    first_a = ToolCallPlan.create("flaky", {"value": "a"})
    first_b = ToolCallPlan.create("flaky", {"value": "b"})
    second_a = ToolCallPlan.create("flaky", {"value": "a"})
    second_b = ToolCallPlan.create("flaky", {"value": "b"})
    third_a = ToolCallPlan.create("flaky", {"value": "a"})
    recovery = ToolCallPlan.create("flaky", {"value": "recovered"})
    state = create_loop_state(
        current_message="Recover without alternating failed calls forever.",
        run_config=_config("loop-alternating-tool-failures"),
        pending_tool_calls=(first_a,),
    )
    state["resident_tool_names"] = ["flaky"]

    result = await _loop(
        provider=_SequenceProvider(
            [
                ModelTurnDraft(action="execute", tool_calls=(first_b,)),
                ModelTurnDraft(action="execute", tool_calls=(second_a,)),
                ModelTurnDraft(action="execute", tool_calls=(second_b,)),
                ModelTurnDraft(action="execute", tool_calls=(third_a,)),
                ModelTurnDraft(action="execute", tool_calls=(recovery,)),
                ModelTurnDraft(action="finish", final_answer="Recovered."),
            ]
        ),
        tools=(_tool("flaky", flaky),),
    ).run(state)

    assert result["status"] == "completed"
    assert attempts == ["a", "b", "a", "b", "recovered"]
    assert result["tool_results"][-2].error_code == "repeated_tool_failure"


@pytest.mark.anyio
async def test_non_retryable_failure_opens_circuit_before_second_attempt() -> None:
    attempts: list[str] = []

    def runner(arguments: Mapping[str, JsonValue]) -> str:
        value = str(arguments["value"])
        attempts.append(value)
        if value == "stuck":
            raise RuntimeError("permanent failure")
        return value

    first = ToolCallPlan.create("non_retryable", {"value": "stuck"})
    repeated = ToolCallPlan.create("non_retryable", {"value": "stuck"})
    recovery = ToolCallPlan.create("non_retryable", {"value": "recovered"})
    state = create_loop_state(
        current_message="Do not retry a permanent failure.",
        run_config=_config("loop-non-retryable-tool-failure"),
        pending_tool_calls=(first,),
    )
    state["resident_tool_names"] = ["non_retryable"]

    result = await _loop(
        provider=_SequenceProvider(
            [
                ModelTurnDraft(action="execute", tool_calls=(repeated,)),
                ModelTurnDraft(action="execute", tool_calls=(recovery,)),
                ModelTurnDraft(action="finish", final_answer="Recovered."),
            ]
        ),
        tools=(
            _tool(
                "non_retryable",
                runner,
                idempotent=False,
            ),
        ),
    ).run(state)

    assert result["status"] == "completed"
    assert attempts == ["stuck", "recovered"]
    assert result["tool_results"][-2].error_code == "repeated_tool_failure"


@pytest.mark.anyio
async def test_same_batch_calls_are_not_preempted_before_retry_outcome() -> None:
    attempts = 0

    def transient(_arguments: Mapping[str, JsonValue]) -> str:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("transient failure")
        return "recovered"

    first = ToolCallPlan.create("transient", {"value": "same"})
    batch = tuple(ToolCallPlan.create("transient", {"value": "same"}) for _ in range(2))
    state = create_loop_state(
        current_message="Allow a successful batch retry.",
        run_config=_config("loop-tool-failure-batch-retry"),
        pending_tool_calls=(first,),
    )
    state["resident_tool_names"] = ["transient"]

    result = await _loop(
        provider=_SequenceProvider(
            [
                ModelTurnDraft(action="execute", tool_calls=batch),
                ModelTurnDraft(action="finish", final_answer="Recovered."),
            ]
        ),
        tools=(_tool("transient", transient),),
    ).run(state)

    assert result["status"] == "completed"
    assert attempts == 3
    assert all(item.error_code != "repeated_tool_failure" for item in result["tool_results"])


@pytest.mark.anyio
async def test_reconciled_execution_record_precedes_failure_circuit() -> None:
    runner_calls = 0

    def must_not_replay(_arguments: Mapping[str, JsonValue]) -> str:
        nonlocal runner_calls
        runner_calls += 1
        return "unexpected replay"

    tool = _tool("remote_write", must_not_replay, idempotent=False)
    plan = ToolCallPlan.create("remote_write", {"value": "once"})
    origin = ToolCallOrigin(
        request_id="request-before-crash",
        toolset_revision="tools-v1",
        exposed_tool_names=("remote_write",),
    )
    call = ToolCall(
        tool_call_id=plan.tool_call_id,
        tool_name=plan.tool_name,
        arguments=plan.arguments,
        origin=origin,
    )
    state = create_loop_state(
        current_message="Recover a non-idempotent tool outcome.",
        run_config=_config("loop-reconciliation-before-circuit"),
        pending_tool_calls=(
            ToolCallPlan(
                tool_call_id=plan.tool_call_id,
                tool_name=plan.tool_name,
                arguments=plan.arguments,
                origin=origin,
            ),
        ),
    )
    state["resident_tool_names"] = ["remote_write"]
    state["tool_manifest"] = build_tool_manifest(
        tools=(tool,),
        resident_tool_names=("remote_write",),
        explicit_tool_names=(),
        active_tool_names=(),
        provider_serializer_revision=state["provider_serializer_revision"],
    )
    state["canonical_tool_calls"] = {call.tool_call_id: call}
    state["tool_execution_records"][call.tool_call_id] = replace(
        ToolExecutionRecord.prepare(call, tool),
        status=ExecutionStatus.OUTCOME_UNKNOWN,
        attempt_count=1,
        error_code="interrupted_outcome_unknown",
        requires_reconciliation=True,
    )
    checkpoint = LangGraphCheckpointStore(
        MemorySaver(serde=agent_checkpoint_serde()),
        run_config=state["run_config"],
    )

    paused = await _loop(
        provider=_SequenceProvider([]),
        tools=(tool,),
        checkpoint=checkpoint,
    ).run(state)

    assert paused["status"] == "paused"
    request = paused["approval_request"]
    assert request is not None
    assert request.kind == "tool_reconciliation"
    resumed = await checkpoint.apply_human_response(
        HumanInputResponse(
            request_id=request.request_id,
            decision="mark_completed",
        )
    )

    result = await _loop(
        provider=_SequenceProvider([ModelTurnDraft(action="finish", final_answer="Recovered.")]),
        tools=(tool,),
        checkpoint=checkpoint,
    ).run(resumed)

    assert result["status"] == "completed"
    assert runner_calls == 0
    assert result["tool_results"][-1].is_error is False
    assert result["tool_results"][-1].metadata["reconciled"] is True

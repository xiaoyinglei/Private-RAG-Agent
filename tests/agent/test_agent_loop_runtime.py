from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterable, Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from rag.agent.core.context import AgentRunConfig, RunRegistry
from rag.agent.core.definition import AgentRuntimePolicy
from rag.agent.core.finalization import FinishCandidateBuilder
from rag.agent.core.turn_contracts import ToolCallPlan
from rag.agent.loop.runtime import (
    AgentLoop,
    LoopEventSink,
    ModelTurnEnvelope,
    _approval_request,
)
from rag.agent.loop.state import (
    LoopState,
    LoopTransition,
    ModelTurnDraft,
    create_loop_state,
)
from rag.agent.loop.stop_hooks import StopHookRunner
from rag.agent.memory.compactor import LoopCompactionResult
from rag.agent.skills.catalog import SkillCatalog
from rag.agent.skills.loader import scan_and_load_skills
from rag.agent.skills.runtime import SkillRuntime
from rag.agent.streaming.events import EventType, StreamEvent, text_delta
from rag.agent.tools.executor import ToolExecutionRecord, ToolExecutor
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
from rag.schema.runtime import AccessPolicy


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
                    run_id=state["run_config"].run_id,
                    turn=state["iteration"],
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


def _config(run_id: str, *, budget: int | None = 20_000) -> AgentRunConfig:
    RunRegistry.remove(run_id)
    return AgentRunConfig(
        run_id=run_id,
        thread_id=run_id,
        llm_budget_total=budget,
        max_depth=1,
        access_policy=AccessPolicy.default(),
    )


def _definition(
    names: list[str] | tuple[str, ...] = (),
    *,
    max_iterations: int = 10,
) -> AgentRuntimePolicy:
    return AgentRuntimePolicy.test_factory(
        agent_type="loop_test",
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
        idempotent=True,
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
    checkpoint: _Checkpoint | None = None,
    definition: AgentRuntimePolicy | None = None,
    events: _Events | None = None,
    max_model_retries: int = 1,
    skill_runtime: SkillRuntime | None = None,
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
    state = create_loop_state(task="Echo.", run_config=_config("loop-basic"))
    state["resident_tool_names"] = ["echo"]

    result = await _loop(
        provider=provider,
        tools=(_tool("echo", lambda arguments: arguments["value"]),),
        checkpoint=checkpoint,
    ).run(state)

    assert result["status"] == "completed"
    assert result["finish_state"].final_answer == "Final answer."
    assert result["tool_results"][0].structured_content == {"text": "hello"}
    assert result["canonical_transcript"][-1].role == "tool"
    assert any(reason == "tool_results_recorded" for reason, _ in checkpoint.snapshots)
    assert [record.status.value for record in checkpoint.execution_records] == [
        "prepared",
        "started",
        "completed",
    ]


@pytest.mark.anyio
async def test_approval_pause_is_a_checkpointed_human_input_event() -> None:
    tool = _tool(
        "remote_lookup",
        lambda arguments: arguments["value"],
        effects=frozenset({ToolEffect.NETWORK}),
    )
    call = ToolCallPlan.create("remote_lookup", {"value": "public docs"})
    state = create_loop_state(
        task="Look this up.",
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
    assert any(
        event.type is EventType.HUMAN_INPUT_REQUIRED for event in events
    )
    assert not any(event.type is EventType.TOOL_USE_ERROR for event in events)
    start = next(event for event in events if event.type is EventType.TOOL_USE_START)
    assert "public docs" in start.data["input_preview"]
    approval_snapshots = [
        snapshot
        for reason, snapshot in checkpoint.snapshots
        if reason == "tool_pause"
    ]
    assert approval_snapshots[-1]["status"] == "paused"


def test_run_command_approval_shows_full_security_context(tmp_path: Path) -> None:
    command = (
        "printf '\x1b[2J'\npython -c \"print('"
        + ("x" * 300)
        + "')\""
    )
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
    runtime = SkillRuntime(
        SkillCatalog(scan_and_load_skills(tmp_path, repo_root=tmp_path))
    )
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
        task="Review this.",
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
    recorded = [
        snapshot
        for reason, snapshot in checkpoint.snapshots
        if reason == "tool_results_recorded"
    ]
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
    state = create_loop_state(task="Echo twice.", run_config=_config("loop-order"))
    state["resident_tool_names"] = ["echo"]

    await _loop(
        provider=provider,
        tools=(
            _tool(
                "echo",
                lambda arguments: seen.append(str(arguments["value"]))
                or arguments["value"],
            ),
        ),
    ).run(state)

    assert seen == ["one", "two"]


@pytest.mark.anyio
async def test_loop_passes_remaining_budget_to_provider() -> None:
    config = _config("loop-budget", budget=100)
    handles = RunRegistry.get_or_create(config)
    assert handles.llm_budget_ledger is not None
    assert await handles.llm_budget_ledger.reserve("seed", 35)
    await handles.llm_budget_ledger.commit("seed", 35)
    provider = _SequenceProvider(
        [ModelTurnDraft(action="finish", final_answer="done")]
    )

    await _loop(provider=provider).run(
        create_loop_state(task="Answer.", run_config=config)
    )

    assert provider.seen_budget_remaining == [65]
    RunRegistry.remove(config.run_id)


@pytest.mark.anyio
async def test_provider_error_retries_then_fails() -> None:
    provider = _SequenceProvider([RuntimeError("down"), RuntimeError("down")])
    result = await _loop(provider=provider, max_model_retries=1).run(
        create_loop_state(
            task="Answer.",
            run_config=_config("loop-provider-error"),
        )
    )

    assert result["status"] == "failed"
    assert result["terminal"] is not None
    assert result["terminal"].stop_reason == "model_provider_failed"


@pytest.mark.anyio
async def test_provider_tool_validation_retry_gives_model_corrective_context() -> None:
    provider = _SequenceProvider(
        [
            LLMToolCallValidationError(
                validation_error=(
                    "Tool call validation failed: max_bytes exceeds maximum"
                ),
                failed_generation=(
                    '<function=read_file>{"path":"README.md",'
                    '"max_bytes":2000000}</function>'
                ),
            ),
            ModelTurnDraft(action="finish", final_answer="recovered"),
        ]
    )

    result = await _loop(provider=provider, max_model_retries=1).run(
        create_loop_state(
            task="Read README.md.",
            run_config=_config("loop-provider-tool-validation"),
        )
    )

    assert result["status"] == "completed"
    retry_transcript = provider.seen_states[1]["canonical_transcript"]
    feedback = retry_transcript[-1]
    assert feedback.role == "context"
    assert "model_tool_call_rejected" in feedback.content
    assert "max_bytes exceeds maximum" in feedback.content
    assert 'max_bytes\\\":2000000' in feedback.content


@pytest.mark.anyio
async def test_run_streaming_injects_sink_and_closes() -> None:
    provider = _SinkAwareProvider()
    events = await asyncio.wait_for(
        _collect(
            _loop(provider=provider).run_streaming(
                create_loop_state(
                    task="Stream.",
                    run_config=_config("loop-stream"),
                )
            )
        ),
        timeout=1,
    )

    assert any(event.type is EventType.TEXT_DELTA for event in events)
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
        task="Find documentation.",
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
        provider=_SequenceProvider(
            [ModelTurnDraft(action="finish", final_answer="Found it.")]
        ),
        tools=snapshot,
        checkpoint=checkpoint,
    ).run(state)

    assert result["active_tool_names"] == [hidden.definition.name]
    activation_snapshots = [
        snap
        for _reason, snap in checkpoint.snapshots
        if hidden.definition.name in snap["active_tool_names"]
    ]
    assert activation_snapshots
    assert all(
        any(item.tool_call_id == call.tool_call_id for item in snap["tool_results"])
        for snap in activation_snapshots
    )

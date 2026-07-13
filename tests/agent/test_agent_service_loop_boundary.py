from __future__ import annotations

from pathlib import Path
from collections.abc import Mapping

import pytest
from langgraph.checkpoint.memory import MemorySaver
from rag.agent.core.agent_service_factory import AgentServiceFactory
from rag.agent.core.checkpointing import (
    CanonicalToolCheckpoint,
    LangGraphCheckpointStore,
    agent_checkpoint_serde,
    encode_tool_checkpoint,
)
from rag.agent.core.context import AgentRunConfig, RunRegistry
from rag.agent.core.definition import AgentRuntimePolicy
from rag.agent.core.goal_contract import GoalDeliverable, GoalSpec
from rag.agent.core.human_input import HumanInputResponse
from rag.agent.core.messages import ModelMessage
from rag.agent.core.model_request import build_tool_manifest
from rag.agent.tools.executor import ExecutionStatus, ToolExecutionRecord
from rag.agent.core.turn_contracts import ToolCallPlan
from rag.agent.loop.state import LoopState, ModelTurnDraft, create_loop_state
from rag.agent.service import AgentRunRequest, AgentService
from rag.agent.tools.registry import ToolRegistry
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
    ToolTarget,
    json_schema_input,
)
from rag.schema.runtime import AccessPolicy


_INPUT_SCHEMA: Mapping[str, JsonValue] = {
    "type": "object",
    "properties": {"value": {"type": "string"}},
    "required": ["value"],
    "additionalProperties": False,
}


class _FinishFromResultsProvider:
    async def next_turn(
        self,
        state: LoopState,
        *,
        definition: AgentRuntimePolicy,
        budget_remaining: int,
    ) -> ModelTurnDraft:
        del definition, budget_remaining
        if state["tool_results"]:
            latest = state["tool_results"][-1]
            if latest.is_error:
                return ModelTurnDraft(action="finish")
            if isinstance(latest.structured_content, Mapping):
                text = latest.structured_content.get("text")
                if isinstance(text, str) and text:
                    return ModelTurnDraft(action="finish", final_answer=text)
        return ModelTurnDraft(
            action="finish",
            final_answer="direct answer",
        )


class _PauseAfterGoalFeedbackProvider:
    async def next_turn(
        self,
        state: LoopState,
        *,
        definition: AgentRuntimePolicy,
        budget_remaining: int,
    ) -> ModelTurnDraft:
        del definition, budget_remaining
        if state["finish_state"].feedback:
            return ModelTurnDraft(
                action="pause",
                pause_reason="Explicit goal still needs evidence.",
            )
        return ModelTurnDraft(
            action="finish",
            final_answer="unsupported answer",
        )


def _definition(*, requires_confirmation: bool = False) -> AgentRuntimePolicy:
    del requires_confirmation
    return AgentRuntimePolicy.test_factory(
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
    execution_revision: str = "write-tool-v1",
) -> ToolRegistry:
    registry = ToolRegistry()

    def runner(payload: Mapping[str, JsonValue]) -> Mapping[str, JsonValue]:
        value = str(payload["value"])
        calls.append(value)
        return {"text": f"wrote:{value}"}

    def normalize(raw: object) -> NormalizedToolOutput:
        assert isinstance(raw, Mapping)
        text = str(raw["text"])
        return NormalizedToolOutput(
            content=(ToolContentBlock(type="text", data={"text": text}),),
            structured_content={"text": text},
        )

    registry.register(
        Tool(
            definition=ToolDefinition(
                name="write_tool",
                description="Write once.",
                input_schema=_INPUT_SCHEMA,
            ),
            validate_input=json_schema_input(_INPUT_SCHEMA),
            run=runner,
            normalize_output=normalize,
            output_schema={
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
                "additionalProperties": False,
            },
            static_effects=frozenset(
                {ToolEffect.WRITE_WORKSPACE} if requires_confirmation else ()
            ),
            resolve_use=lambda _arguments: ResolvedToolUse(
                effects=frozenset(
                    {ToolEffect.WRITE_WORKSPACE} if requires_confirmation else ()
                ),
                targets=(
                    (ToolTarget(kind="workspace_path", value="."),)
                    if requires_confirmation
                    else ()
                ),
            ),
            execution_revision=execution_revision,
            idempotent=not requires_confirmation,
            concurrency_safe=True,
            cancellation_mode=CancellationMode.COOPERATIVE,
            interrupt_behavior=InterruptBehavior.CANCEL,
            timeout_seconds=1.0,
            max_model_output_bytes=4096,
        ),
    )
    return registry


def _config(run_id: str) -> AgentRunConfig:
    return AgentRunConfig(
        run_id=run_id,
        thread_id=run_id,
        llm_budget_total=10_000,
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
    request = service.pending_human_input_request(run_id="service-loop-resume")
    assert request == paused.human_input_request
    assert await service.apending_human_input_request(run_id="service-loop-resume") == request

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
    state["tool_execution_records"][call.tool_call_id] = ToolExecutionRecord(
        tool_call_id=call.tool_call_id,
        tool_name=call.tool_name,
        operation_id="op-ambiguous",
        arguments_digest="legacy-digest-replaced-during-resume",
        idempotent=False,
        status=ExecutionStatus.UNKNOWN,
        attempt_count=1,
        error_code="interrupted_outcome_unknown",
        requires_reconciliation=True,
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
async def test_resume_reconciles_pending_call_before_changed_tool_executes() -> None:
    run_id = "service-loop-manifest-drift"
    config = _config(run_id)
    checkpointer = MemorySaver(serde=agent_checkpoint_serde())
    persisted_registry = _registry([], execution_revision="write-tool-v1")
    persisted_tool = persisted_registry.get("write_tool")
    persisted_manifest = build_tool_manifest(
        tools=(persisted_tool,),
        resident_tool_names=("write_tool",),
        provider_serializer_revision="provider-wire-v1",
    )
    call = ToolCall(
        tool_call_id="call-drift",
        tool_name="write_tool",
        arguments={"value": "do not replay"},
        origin=ToolCallOrigin(
            request_id="request-before-drift",
            toolset_revision=persisted_manifest.toolset_revision,
            exposed_tool_names=("write_tool",),
        ),
    )
    state = create_loop_state(
        task="Resume safely.",
        run_config=config,
        pending_tool_calls=(
            ToolCallPlan(
                tool_call_id=call.tool_call_id,
                tool_name=call.tool_name,
                arguments=dict(call.arguments),
            ),
        ),
    )
    state["tool_checkpoint"] = encode_tool_checkpoint(  # type: ignore[typeddict-unknown-key]
        CanonicalToolCheckpoint(
            context_revision="context-before-drift",
            prompt_revision="prompt-before-drift",
            transcript=(ModelMessage(role="user", content="Resume safely."),),
            manifest=persisted_manifest,
            tool_calls=(call,),
            pending_tool_calls=(call,),
        )
    )
    await LangGraphCheckpointStore(
        checkpointer,
        run_config=config,
    ).save_snapshot(state, reason="before-drift")
    calls: list[str] = []
    service = AgentService(
        definition=_definition(),
        tool_registry=_registry(
            calls,
            execution_revision="write-tool-v2",
        ),
        model_turn_provider=_FinishFromResultsProvider(),
        checkpointer=checkpointer,
    )

    request = await service.apending_human_input_request(run_id=run_id)

    assert request.kind == "tool_reconciliation"
    assert request.context["error_code"] == "tool_definition_changed"
    assert request.context["tool_call_id"] == call.tool_call_id
    assert calls == []
    RunRegistry.remove(run_id)


@pytest.mark.anyio
async def test_explicit_goal_spec_is_a_stop_hook_not_default_controller() -> None:
    service = AgentService(
        definition=_definition(),
        tool_registry=_registry([]),
        model_turn_provider=_PauseAfterGoalFeedbackProvider(),
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


def test_runtime_boundaries_do_not_import_inner_graph_nodes() -> None:
    root = Path(__file__).resolve().parents[2]
    runtime_files = [
        "rag/agent/service.py",
        "rag/agent/core/agent_service_factory.py",
        "rag/agent/core/agent_as_tool.py",
        "rag/agent/core/compiler.py",
        "rag/agent/core/llm_providers.py",
        "rag/agent/builtin/generic.py",
    ]

    offenders = [relative for relative in runtime_files if "rag.agent.graphs.nodes" in (root / relative).read_text()]

    assert offenders == []


def test_runtime_modules_use_loop_state_instead_of_compatibility_state() -> None:
    root = Path(__file__).resolve().parents[2]
    runtime_files = [
        "rag/agent/core/llm_context.py",
        "rag/agent/core/output_finalizer.py",
        "rag/agent/memory/injector.py",
        "rag/agent/tools/registry.py",
    ]

    offenders = [
        relative for relative in runtime_files if "rag.agent.state" in (root / relative).read_text(encoding="utf-8")
    ]

    assert offenders == []


@pytest.mark.anyio
async def test_service_freezes_one_snapshot_and_reuses_one_executor() -> None:
    service = AgentService(
        definition=_definition(),
        tool_registry=_registry([]),
        model_turn_provider=_FinishFromResultsProvider(),
    )

    snapshot = service._tool_snapshot
    executor = service._tool_executor
    result = await service.run(
        AgentRunRequest(
            task="Answer directly.",
            run_id="single-runtime-identity",
            thread_id="single-runtime-identity",
        )
    )

    assert result.status == "done"
    assert service._tool_snapshot is snapshot
    assert service._tool_executor is executor
    assert executor._tools is snapshot


def test_public_runtime_files_do_not_reach_legacy_tool_paths() -> None:
    root = Path(__file__).resolve().parents[2]
    runtime_files = (
        "agent_runtime/agent.py",
        "agent_runtime/runtime/builder.py",
        "rag/agent/cli.py",
        "rag/agent/service.py",
        "rag/agent/loop/runtime.py",
        "rag/agent/core/model_provider_runtime.py",
        "rag/agent/core/llm_providers.py",
        "rag/providers/llm_gateway.py",
    )
    forbidden = (
        "rag.agent.tooling",
        "ToolExecutionService",
        "RuntimeToolRegistryBuilder",
        "ToolSurfaceRequest",
        "ToolSurfacePolicy",
        "DeferredToolStore",
        "resolve_visible_tools",
        "ModelRequestBuilder",
    )

    offenders = {
        relative: tuple(symbol for symbol in forbidden if symbol in (root / relative).read_text())
        for relative in runtime_files
        if any(symbol in (root / relative).read_text() for symbol in forbidden)
    }

    assert offenders == {}


def test_default_runtime_has_no_goal_gap_planning_fields() -> None:
    root = Path(__file__).resolve().parents[2] / "rag" / "agent"
    forbidden = (
        "related_gap_ids",
        "resolved_gaps",
        "produced_gaps",
        "goal_gap_refs_ignored",
    )
    offenders: list[str] = []

    for path in root.rglob("*.py"):
        if "compat" in path.parts or path.name == "stop_hooks.py":
            continue
        source = path.read_text(encoding="utf-8")
        if any(symbol in source for symbol in forbidden):
            offenders.append(str(path.relative_to(root)))

    assert offenders == []

from __future__ import annotations

from collections.abc import Mapping
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from pydantic import BaseModel

from rag.agent.core.context import AgentRunConfig
from rag.agent.core.definition import AgentRuntimePolicy
from rag.agent.core.human_input import HumanInputRequest, ToolCallSummary
from rag.agent.core.llm_context import AgentLLMContextAssembler
from rag.agent.core.llm_providers import (
    LLMLoopModelTurnProvider,
    create_loop_model_turn_provider,
    parse_loop_model_turn,
)
from rag.agent.core.llm_registry import ResolvedModel
from rag.agent.core.messages import (
    StopReason,
    ToolUseResult,
    tool_result_message,
)
from rag.agent.core.messages import (
    ToolCall as ModelToolCall,
)
from rag.agent.core.turn_contracts import ToolCallPlan
from rag.agent.loop.state import (
    LoopState,
    ModelTurnDraft,
    PendingToolCall,
    StopHookFeedback,
    create_loop_state,
)
from rag.agent.memory.compactor import LoopContextCompactor
from rag.agent.memory.injector import ContextBuilder
from rag.agent.memory.models import MemoryPolicy, MemoryRef, MessageBatchPayload
from rag.agent.planning import PlanStep, PlanTracker, PlanUpdate
from rag.agent.tools.tool import (
    CancellationMode,
    InterruptBehavior,
    JsonValue,
    NormalizedToolOutput,
    ResolvedToolUse,
    Tool,
    ToolContentBlock,
    ToolDefinition,
    ToolResult,
    json_schema_input,
)
from rag.assembly.tokenizer import TokenAccountingService, TokenizerContract
from rag.providers.llm_gateway import AgentModelResponse
from rag.schema.llm import LLMCallStage, LLMStageBudget, LLMUsage
from rag.schema.runtime import AccessPolicy


class _RecordingGateway:
    def __init__(self, turn: ToolUseResult | None = None) -> None:
        self.calls: list[dict[str, object]] = []
        self.turn = turn or ToolUseResult(
            text="The policy changed in 2026.",
            tool_calls=[],
            stop_reason=StopReason.END_TURN,
            raw_stop_reason="stop",
        )

    async def agenerate_model_request(self, **kwargs: object) -> AgentModelResponse:
        self.calls.append(dict(kwargs))
        return AgentModelResponse(
            turn=self.turn,
            usage=LLMUsage(
                input_tokens=20,
                output_tokens=4,
                source="provider",
                logical_input_tokens=20,
                uncached_input_tokens=20,
                usage_source="provider",
            ),
            provider_wire_hash="wire-loop-context",
            serializer_revision="provider-wire-v1",
            wire_kind=str(kwargs["provider"]),
        )


class _RecordingMemoryStore:
    def __init__(self) -> None:
        self.records: list[BaseModel] = []

    def write_tool_output(
        self,
        payload: BaseModel,
        *,
        summary: str,
        source_tool_call_id: str | None = None,
        source_tool_name: str | None = None,
        warnings: list[str] | None = None,
    ) -> MemoryRef:
        self.records.append(payload)
        ref_id = f"mem_{len(self.records)}"
        return MemoryRef(
            ref_id=ref_id,
            path=f".agent_memory/records/{ref_id}.json",
            summary=summary,
            source_tool_call_id=source_tool_call_id,
            source_tool_name=source_tool_name,
            warnings=list(warnings or []),
        )


def _definition() -> AgentRuntimePolicy:
    return AgentRuntimePolicy.test_factory(
        agent_type="research",
        description="Research",
        system_prompt="Use tools when they help and preserve citations.",
        allowed_tools=["vector_search", "read_file"],
    )


def _run_config(run_id: str = "loop-context") -> AgentRunConfig:
    return AgentRunConfig(
        run_id=run_id,
        thread_id=run_id,
        llm_budget_total=10_000,
        max_depth=2,
        access_policy=AccessPolicy.default(),
    )


def _state(run_id: str = "loop-context") -> LoopState:
    return create_loop_state(
        task="Explain the policy with sources.",
        run_config=_run_config(run_id),
    )


def _tool(name: str) -> Tool:
    schema: Mapping[str, JsonValue] = {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
        "additionalProperties": False,
    }
    return Tool(
        definition=ToolDefinition(
            name=name,
            description=f"Use {name}.",
            input_schema=schema,
        ),
        validate_input=json_schema_input(schema),
        run=lambda arguments: {"text": str(arguments["query"])},
        normalize_output=lambda raw: NormalizedToolOutput(
            content=(ToolContentBlock(type="text", data={"text": str(raw)}),),
            structured_content={"text": str(raw)},
        ),
        output_schema=None,
        static_effects=frozenset(),
        resolve_use=lambda _arguments: ResolvedToolUse(
            effects=frozenset(),
            targets=(),
        ),
        execution_revision=f"{name}-v1",
        idempotent=True,
        concurrency_safe=True,
        cancellation_mode=CancellationMode.COOPERATIVE,
        interrupt_behavior=InterruptBehavior.CANCEL,
        timeout_seconds=3,
        max_model_output_bytes=4096,
    )


def _provider(
    gateway: _RecordingGateway,
    *,
    names: tuple[str, ...] = ("vector_search", "read_file"),
    supports_native_tools: bool = True,
) -> LLMLoopModelTurnProvider:
    snapshot = {name: _tool(name) for name in names}
    return LLMLoopModelTurnProvider(
        gateway,  # type: ignore[arg-type]
        model="test-model",
        provider="openai-compatible",
        supports_native_tools=supports_native_tools,
        registry_snapshot=snapshot,
        resident_tool_names=names,
    )


def _assembler() -> AgentLLMContextAssembler:
    accounting = TokenAccountingService(
        TokenizerContract(
            embedding_model_name="loop-context",
            tokenizer_model_name="loop-context",
            chunking_tokenizer_model_name="loop-context",
            tokenizer_backend="simple",
            max_context_tokens=8_192,
            prompt_reserved_tokens=256,
            local_files_only=True,
        )
    )
    return AgentLLMContextAssembler(
        token_accounting=accounting,
        stage_budgets={
            LLMCallStage.TOOL_DECISION: LLMStageBudget(
                max_input_tokens=6_000,
                max_output_tokens=1_000,
                safety_margin_tokens=128,
            )
        },
    )


def test_turn_parser_prefers_actual_calls_over_finish_label() -> None:
    call = ToolCallPlan.create("vector_search", {"query": "policy"})

    finish = parse_loop_model_turn(
        {"action": "finish", "final_answer": "Enough evidence."}
    )
    execute = parse_loop_model_turn(
        {
            "action": "finish",
            "final_answer": "Too early.",
            "tool_calls": [call.model_dump()],
        }
    )

    assert finish == ModelTurnDraft(
        action="finish",
        final_answer="Enough evidence.",
    )
    assert execute == ModelTurnDraft(action="execute", tool_calls=(call,))


@pytest.mark.anyio
async def test_loop_provider_builds_one_canonical_request_and_finish() -> None:
    gateway = _RecordingGateway()
    state = _state()
    state["resident_tool_names"] = ["vector_search", "read_file"]

    envelope = await _provider(gateway).next_turn(
        state,
        definition=_definition(),
        budget_remaining=5_000,
    )

    request = gateway.calls[0]["request"]
    assert request is envelope.request
    assert request.exposed_tool_names == ("vector_search", "read_file")
    assert envelope.draft == ModelTurnDraft(
        action="finish",
        final_answer="The policy changed in 2026.",
    )
    assert envelope.model_call_record is not None
    assert envelope.model_call_record.request_id == request.request_id
    assert envelope.model_call_record.provider_wire_hash == "wire-loop-context"
    assert envelope.context_revision is not None
    assert envelope.context_revision.startswith("context_")


@pytest.mark.anyio
async def test_loop_provider_binds_tool_call_to_originating_request() -> None:
    gateway = _RecordingGateway(
        ToolUseResult(
            text="",
            tool_calls=[
                ModelToolCall(
                    id="tc-provider",
                    name="read_file",
                    input={"query": "README.md"},
                )
            ],
            stop_reason=StopReason.TOOL_USE,
            raw_stop_reason="tool_calls",
        )
    )
    state = _state("loop-context-origin")
    state["resident_tool_names"] = ["read_file"]

    envelope = await _provider(gateway, names=("read_file",)).next_turn(
        state,
        definition=_definition(),
        budget_remaining=5_000,
    )

    assert envelope.draft.action == "execute"
    [call] = envelope.draft.tool_calls
    assert call.origin is not None
    assert call.origin.request_id == envelope.request.request_id
    assert call.origin.toolset_revision == envelope.request.toolset_revision
    assert call.origin.exposed_tool_names == ("read_file",)


@pytest.mark.anyio
async def test_loop_provider_selection_is_state_driven_not_task_classified() -> None:
    gateway = _RecordingGateway()
    state = _state("loop-context-selection")
    state["task"] = "Answer exactly with the single word: OK"
    state["resident_tool_names"] = ["read_file"]

    envelope = await _provider(gateway, names=("read_file",)).next_turn(
        state,
        definition=_definition(),
        budget_remaining=5_000,
    )

    assert envelope.request.exposed_tool_names == ("read_file",)


@pytest.mark.anyio
async def test_provider_factory_uses_resolved_wire_capability() -> None:
    gateway = _RecordingGateway()

    class _Registry:
        default_model = "fake"
        fallback_model = "fake"
        generation_config = None

        def resolve_for_node(
            self,
            *,
            node_model: str | None,
            node_name: str,
        ) -> ResolvedModel:
            del node_model, node_name
            return ResolvedModel(
                generator=SimpleNamespace(),
                kwargs={},
                gateway=gateway,
                provider="ollama",
                model="local-model",
                supports_native_tools=False,
            )

    state = _state("loop-context-factory")
    state["resident_tool_names"] = ["read_file"]
    provider = create_loop_model_turn_provider(
        _Registry(),  # type: ignore[arg-type]
        _definition().model_selection,
        registry_snapshot={"read_file": _tool("read_file")},
        resident_tool_names=("read_file",),
    )

    envelope = await provider.next_turn(
        state,
        definition=_definition(),
        budget_remaining=5_000,
    )

    assert envelope.request.settings.model == "local-model"
    assert gateway.calls[0]["provider"] == "ollama"
    assert gateway.calls[0]["supports_native_tools"] is False


def test_loop_context_keeps_approval_and_feedback_without_goal_fields() -> None:
    state = _state()
    call = ToolCallPlan.create("vector_search", {"query": "policy"})
    state["pending_tool_calls"] = [PendingToolCall(plan=call, status="pending")]
    state["approval_request"] = HumanInputRequest(
        request_id="hir_loop",
        kind="tool_approval",
        question="Allow this tool?",
        tool_calls=[
            ToolCallSummary(
                tool_call_id=call.tool_call_id,
                tool_name=call.tool_name,
                args_preview="query='policy'",
            )
        ],
    )
    state["finish_state"].feedback = [
        StopHookFeedback(
            code="citation_required",
            message="Add a traceable citation.",
        )
    ]
    state["plan_state"].agent_plan = PlanTracker().initialize_task(
        task=state["task"]
    )[0]

    context = ContextBuilder(max_context_tokens=4_000).assemble_loop(
        definition=_definition(),
        state=state,
    )

    decisions = context.section("open_decisions").content
    assert call.tool_call_id in decisions
    assert "tool_approval" in decisions
    assert "Add a traceable citation." in decisions
    assert "open_gaps" not in decisions
    assert "goal_spec" not in decisions


def test_loop_context_assembler_uses_focused_loop_entry_point() -> None:
    assembled = _assembler().assemble_loop_turn(
        definition=_definition(),
        state=_state(),
        budget_remaining=5_000,
        output_schema=ModelTurnDraft,
    )

    assert assembled.stage == LLMCallStage.TOOL_DECISION
    assert "open_gaps" not in assembled.prompt
    assert "Use tools when they help" in assembled.prompt


def test_task_plan_is_advisory_and_filters_unsupported_tools() -> None:
    tracker = PlanTracker()
    plan, _ = tracker.initialize_task(task="Inspect the workspace and answer.")

    updated, events = tracker.apply_advisory_update(
        plan,
        PlanUpdate(
            mode="replace",
            steps=[
                PlanStep(
                    step_id="step_inspect",
                    title="Inspect relevant files",
                    expected_tool_names=["vector_search", "delete_everything"],
                )
            ],
        ),
        allowed_tool_names=frozenset({"vector_search"}),
    )

    assert plan.steps[0].title == "Work on the current task."
    assert updated.steps[0].expected_tool_names == ["vector_search"]
    assert "unsupported_tool_names" in events[0].warnings


def test_loop_context_compaction_is_observable_before_model_turn() -> None:
    state = create_loop_state(
        task="Summarize the conversation.",
        run_config=AgentRunConfig(
            run_id="loop-compaction",
            thread_id="loop-compaction",
            llm_budget_total=10_000,
            max_depth=2,
            access_policy=AccessPolicy.default(),
            memory_policy=MemoryPolicy(
                message_compaction_min_count=3,
                max_message_tail_count=1,
            ),
        ),
        messages=[
            HumanMessage(content=f"message {index}", id=f"msg-{index}")
            for index in range(4)
        ],
    )

    result = LoopContextCompactor().prepare(state)

    assert result.changed is True
    assert "messages" in result.channels
    assert [message.id for message in state["messages"]] == ["msg-3"]
    assert state["memory_state"].working_summary is not None
    assert state["latest_transition"] is not None
    assert state["latest_transition"].reason == "compaction"
    assert "memory_unavailable" in state["memory_state"].memory_warnings


def test_loop_context_snips_messages_without_splitting_tool_pairs() -> None:
    tool_call_id = "tc-search"
    store = _RecordingMemoryStore()
    state = create_loop_state(
        task="Summarize the conversation.",
        run_config=AgentRunConfig(
            run_id="loop-snip-compaction",
            thread_id="loop-snip-compaction",
            llm_budget_total=10_000,
            max_depth=2,
            access_policy=AccessPolicy.default(),
            memory_policy=MemoryPolicy(
                message_compaction_min_count=99,
                snip_compact_threshold=4,
                snip_keep_head=1,
                snip_keep_tail=1,
            ),
        ),
        messages=[
            HumanMessage(content="original task", id="msg-head"),
            HumanMessage(content="old detail 1", id="msg-old-1"),
            HumanMessage(content="old detail 2", id="msg-old-2"),
            AIMessage(
                content="",
                id="msg-ai-tool",
                tool_calls=[
                    {
                        "id": tool_call_id,
                        "name": "vector_search",
                        "args": {"query": "policy"},
                    }
                ],
            ),
            ToolMessage(
                content="search result",
                id="msg-tool-result",
                tool_call_id=tool_call_id,
            ),
        ],
    )

    result = LoopContextCompactor(store=store).prepare(state)

    assert result.changed is True
    assert [message.id for message in state["messages"]] == [
        "msg-head",
        "snip_compact_2",
        "msg-ai-tool",
        "msg-tool-result",
    ]
    assert len(store.records) == 1
    stored = store.records[0]
    assert isinstance(stored, MessageBatchPayload)
    assert [message.id for message in stored.messages] == [
        "msg-old-1",
        "msg-old-2",
    ]


def test_compaction_never_reformats_canonical_tool_results() -> None:
    state = _state("loop-canonical-tool-results")
    results = [
        ToolResult(
            tool_call_id=f"tc-{index}",
            tool_name="read_file",
            content=(
                ToolContentBlock(
                    type="text",
                    data={"text": f"fixed content {index}"},
                ),
            ),
            structured_content={"text": f"fixed content {index}"},
        )
        for index in range(3)
    ]
    transcript = [tool_result_message(result) for result in results]
    state["tool_results"] = results
    state["canonical_transcript"] = transcript

    LoopContextCompactor().prepare(state)

    assert state["tool_results"] == results
    assert state["canonical_transcript"] == transcript
    assert [message.content for message in transcript] == [
        message.content for message in state["canonical_transcript"]
    ]

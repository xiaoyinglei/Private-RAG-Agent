from __future__ import annotations

from typing import Any

import pytest
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from rag.agent.core.context import AgentRunConfig
from rag.agent.core.definition import AgentDefinition
from rag.agent.core.human_input import HumanInputRequest, ToolCallSummary
from rag.agent.core.llm_context import AgentLLMContextAssembler
from rag.agent.core.llm_prompts import build_loop_turn_prompt
from rag.agent.core.llm_providers import (
    LLMLoopModelTurnProvider,
    parse_loop_model_turn,
)
from rag.agent.core.observations import (
    AnswerCandidate,
    EvidenceRef,
    StructuredObservation,
)
from rag.agent.loop.state import (
    ModelTurnDraft,
    StopHookFeedback,
    create_loop_state,
)
from rag.agent.memory.compactor import LoopContextCompactor, MemoryCompactor
from rag.agent.memory.injector import ContextBuilder
from rag.agent.memory.models import MemoryPolicy, StateChannelReplacement
from rag.agent.planning import PlanStep, PlanTracker, PlanUpdate
from rag.agent.state import ToolCallPlan
from rag.assembly.tokenizer import TokenAccountingService, TokenizerContract
from rag.schema.llm import LLMCallStage, LLMStageBudget
from rag.schema.runtime import AccessPolicy


class _StubGenerator:
    def __init__(self, responses: list[dict[str, Any]]) -> None:
        self._responses = responses
        self.calls: list[tuple[str, type[BaseModel], dict[str, Any]]] = []

    def generate_structured(
        self,
        *,
        prompt: str,
        schema: type[BaseModel],
        **kwargs: Any,
    ) -> BaseModel:
        self.calls.append((prompt, schema, kwargs))
        return schema.model_validate(self._responses.pop(0))


def _definition() -> AgentDefinition:
    return AgentDefinition(
        agent_type="research",
        description="Research",
        system_prompt="Use tools when they help and preserve citations.",
        allowed_tools=["vector_search", "llm_summarize"],
    )


def _run_config() -> AgentRunConfig:
    return AgentRunConfig(
        run_id="loop-context",
        thread_id="loop-context",
        budget_total=10_000,
        max_depth=2,
        access_policy=AccessPolicy.default(),
    )


def _state() -> dict[str, Any]:
    return create_loop_state(
        task="Explain the policy with sources.",
        run_config=_run_config(),
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


def test_loop_prompt_has_no_goal_gap_completion_authority() -> None:
    state = _state()
    state["agent_plan"] = PlanTracker().initialize_task(task=state["task"])[0]

    prompt = build_loop_turn_prompt(
        state,
        budget_remaining=5_000,
        allowed_tools=_definition().allowed_tools,
    )

    assert '"action": "execute" | "finish" | "pause"' in prompt
    assert "final_answer" in prompt
    assert "actual tool calls take precedence" in prompt
    assert "plan is advisory" in prompt
    assert "open_gaps" not in prompt
    assert "goal checker" not in prompt
    assert "must call llm_summarize" not in prompt


def test_turn_parser_prefers_actual_calls_over_finish_label() -> None:
    call = ToolCallPlan.create("vector_search", {"query": "policy"})

    finish = parse_loop_model_turn(
        {
            "action": "finish",
            "final_answer": "Enough evidence.",
        }
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
async def test_loop_provider_returns_finish_without_satisfaction_authorization() -> None:
    generator = _StubGenerator(
        [
            {
                "action": "finish",
                "final_answer": "The policy changed in 2026.",
                "thought": "Ready to answer.",
            }
        ]
    )
    provider = LLMLoopModelTurnProvider(generator)

    draft = await provider.next_turn(
        _state(),
        definition=_definition(),
        budget_remaining=5_000,
    )

    assert draft == ModelTurnDraft(
        action="finish",
        final_answer="The policy changed in 2026.",
    )
    assert len(generator.calls) == 1
    prompt = generator.calls[0][0]
    assert "open_gaps" not in prompt
    assert "goal checker" not in prompt


def test_loop_context_keeps_approval_and_feedback_without_goal_fields() -> None:
    state = _state()
    call = ToolCallPlan.create("vector_search", {"query": "policy"})
    state["pending_tool_calls"] = [call]
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
    state["stop_hook_feedback"] = [
        StopHookFeedback(
            code="citation_required",
            message="Add a traceable citation.",
        )
    ]
    state["agent_plan"] = PlanTracker().initialize_task(
        task=state["task"],
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
    assert "related_gap_ids" not in context.section("plan").content


def test_loop_context_assembler_uses_focused_loop_entry_point() -> None:
    state = _state()

    assembled = _assembler().assemble_loop_turn(
        definition=_definition(),
        state=state,
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


def test_loop_compaction_pins_pending_approval_and_candidate_evidence() -> None:
    pending = ToolCallPlan(
        tool_call_id="tc_pending",
        tool_name="vector_search",
        arguments={"query": "policy"},
    )
    approval = ToolCallPlan(
        tool_call_id="tc_approval",
        tool_name="write_file",
        arguments={"path": "reports/policy.md"},
    )
    evidence_ref = EvidenceRef(evidence_id="ev_keep", citation_id="cit_keep")
    compactor = MemoryCompactor(
        policy=MemoryPolicy(
            max_structured_observations=2,
            max_answer_candidates=1,
            max_evidence_refs=1,
        ),
        loop_mode=True,
    )
    state: dict[str, Any] = {
        "task": "Explain the policy with sources.",
        "pending_tool_calls": [pending],
        "approval_request": HumanInputRequest(
            request_id="hir_approval",
            kind="tool_approval",
            question="Allow write?",
            tool_calls=[
                ToolCallSummary(
                    tool_call_id=approval.tool_call_id,
                    tool_name=approval.tool_name,
                    args_preview="path='reports/policy.md'",
                )
            ],
        ),
        "structured_observations": [
            StructuredObservation(
                tool_call_id="tc_old",
                tool_name="vector_search",
                status="ok",
                raw_result_ref="tc_old",
            ),
            StructuredObservation(
                tool_call_id=pending.tool_call_id,
                tool_name=pending.tool_name,
                status="ok",
                raw_result_ref=pending.tool_call_id,
            ),
            StructuredObservation(
                tool_call_id=approval.tool_call_id,
                tool_name=approval.tool_name,
                status="ok",
                raw_result_ref=approval.tool_call_id,
            ),
        ],
        "answer_candidates": [
            AnswerCandidate(
                text="Current candidate",
                source_tool_call_id="tc_old",
                evidence_refs=[evidence_ref],
            )
        ],
        "evidence_refs": [
            EvidenceRef(evidence_id="ev_old"),
            evidence_ref,
        ],
    }

    compacted = compactor.compact_update(
        state,
        {"retrieval_signals_debug": {"source": "irrelevant"}},
    )

    [replacement] = compacted["structured_observations"]
    assert isinstance(replacement, StateChannelReplacement)
    observations = _apply_replacement(
        state["structured_observations"],
        compacted["structured_observations"],
    )
    assert {item.tool_call_id for item in observations} == {
        pending.tool_call_id,
        approval.tool_call_id,
    }
    evidence = _apply_replacement(
        state["evidence_refs"],
        compacted["evidence_refs"],
    )
    assert evidence == [evidence_ref]
    assert compacted["retrieval_signals_debug"] == {
        "source": "irrelevant"
    }
    assert compacted["memory_budget"].pinned_item_count >= 3


def test_loop_context_compaction_is_observable_before_model_turn() -> None:
    state = create_loop_state(
        task="Summarize the conversation.",
        run_config=AgentRunConfig(
            run_id="loop-compaction",
            thread_id="loop-compaction",
            budget_total=10_000,
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
    assert state["working_summary"] is not None
    assert state["latest_transition"] is not None
    assert state["latest_transition"].reason == "compaction"
    assert "memory_unavailable" in state["memory_warnings"]


def _apply_replacement(
    current: list[object],
    update: list[object],
) -> list[object]:
    if len(update) == 1 and isinstance(update[0], StateChannelReplacement):
        return list(update[0].items)
    return [*current, *update]

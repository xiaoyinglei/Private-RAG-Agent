from __future__ import annotations

import pytest

from rag.agent.core.context import AgentRunConfig
from rag.agent.core.definition import AgentDefinition
from rag.agent.core.llm_context import (
    AgentLLMContextAssembler,
    AgentLLMContextOverflowError,
)
from rag.agent.goal_runtime import GoalContractHint
from rag.agent.state import AgentState
from rag.schema.llm import LLMCallStage, LLMStageBudget
from rag.schema.runtime import AccessPolicy


class _CharacterTokenAccounting:
    def count(self, text: str) -> int:
        return len(text)

    def clip(
        self,
        text: str,
        token_budget: int,
        *,
        add_ellipsis: bool = False,
    ) -> str:
        clipped = text[: max(token_budget, 0)]
        if add_ellipsis and len(clipped) < len(text) and token_budget >= 4:
            return clipped[: token_budget - 4].rstrip() + " ..."
        return clipped


def _definition() -> AgentDefinition:
    return AgentDefinition(
        agent_type="research",
        description="Research",
        system_prompt="SYSTEM_POLICY",
        allowed_tools=["llm_generate"],
    )


def _state() -> AgentState:
    return {
        "messages": [],
        "evidence": [],
        "citations": [],
        "tool_results": [],
        "task": "Answer the task",
        "retrieval_signals": None,  # type: ignore[typeddict-item]
        "retrieval_signals_debug": None,
        "run_config": AgentRunConfig(
            run_id="context-assembler",
            thread_id="context-assembler",
            budget_total=10_000,
            max_depth=1,
            access_policy=AccessPolicy.default(),
        ),
        "iteration": 0,
        "status": "running",
        "decision_reason": None,
        "stop_reason": None,
        "needs_user_input": None,
        "pending_tool_calls": [],
        "approved_tool_call_ids": [],
        "denied_tool_call_ids": [],
        "user_decision": None,
        "user_message": None,
        "human_input_request": None,
        "human_input_response": None,
        "working_summary": None,
        "extracted_facts": [],
        "context_budget": None,
        "final_answer": None,
        "groundedness_flag": False,
        "insufficient_evidence_flag": False,
        "goal_spec": None,
        "goal_requirements": [],
        "satisfied_requirements": [],
        "open_gaps": [],
        "evidence_refs": [],
        "answer_candidates": [],
        "computation_results": [],
        "structured_observations": [],
        "context_units": [],
        "context_bindings": [],
        "locators": [],
        "asset_refs": [],
        "conflicts": [],
        "no_progress_count": 0,
        "satisfaction_report": None,
        "controller_next": None,
        "agent_plan": None,
        "plan_events": [],
        "memory_refs": [],
        "memory_budget": None,
        "memory_warnings": [],
    }


def _assembler(max_input_tokens: int) -> AgentLLMContextAssembler:
    budgets = {
        stage: LLMStageBudget(
            max_input_tokens=max_input_tokens,
            max_output_tokens=20,
            safety_margin_tokens=0,
        )
        for stage in LLMCallStage
    }
    return AgentLLMContextAssembler(
        token_accounting=_CharacterTokenAccounting(),
        stage_budgets=budgets,
    )


def test_generate_context_uses_model_token_count_for_complete_prompt() -> None:
    assembler = _assembler(max_input_tokens=500)

    assembled = assembler.assemble_generate(
        definition=_definition(),
        state=_state(),
        prompt="Produce an answer",
        context_sections=["REAL_SUPPORT"],
        stage=LLMCallStage.LLM_GENERATE,
    )

    assert "SYSTEM_POLICY" in assembled.prompt
    assert "Produce an answer" in assembled.prompt
    assert "REAL_SUPPORT" in assembled.prompt
    assert assembled.context.context_budget.used_context_tokens == len(
        assembled.prompt
    )


def test_goal_contract_context_preserves_required_task_content() -> None:
    assembler = _assembler(max_input_tokens=3_000)

    assembled = assembler.assemble_goal_contract(
        definition=_definition(),
        state=_state(),
        output_schema=GoalContractHint,
    )

    assert assembled.stage == LLMCallStage.GOAL_CONTRACT
    assert "Answer the task" in assembled.prompt
    assert "sha256=" not in assembled.prompt


def test_required_call_context_overflow_raises_without_hash_replacement() -> None:
    assembler = _assembler(max_input_tokens=80)

    with pytest.raises(AgentLLMContextOverflowError) as exc_info:
        assembler.assemble_generate(
            definition=_definition(),
            state=_state(),
            prompt="Produce an answer",
            context_sections=["REQUIRED_REAL_CONTEXT " * 20],
            stage=LLMCallStage.LLM_GENERATE,
        )

    error = exc_info.value
    assert error.context_budget.overflow is True
    assert "call_context" in error.context_budget.required_truncated
    assert "sha256=" not in str(error)


def test_optional_state_context_can_be_dropped_without_overflow() -> None:
    state = _state()
    state["memory_warnings"] = ["OPTIONAL_WARNING " * 100]
    assembler = _assembler(max_input_tokens=180)

    assembled = assembler.assemble_generate(
        definition=_definition(),
        state=state,
        prompt="Produce an answer",
        context_sections=[],
        stage=LLMCallStage.LLM_GENERATE,
    )

    assert assembled.context.context_budget.overflow is False
    assert "sha256=" not in assembled.prompt

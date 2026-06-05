from __future__ import annotations

from dataclasses import replace

import pytest
from langchain_core.messages import HumanMessage

from rag.agent.core.context import AgentRunConfig, RuntimeRegistry
from rag.agent.core.definition import AgentDefinition
from rag.agent.graphs.nodes.llm_decide import llm_decide_node
from rag.agent.memory.models import InjectedContext, WorkingSummary
from rag.agent.primitive_ops import RunPythonOutput
from rag.agent.state import AgentState, ThinkOutput
from rag.agent.tools.spec import ToolResult
from rag.schema.query import AnswerCitation, EvidenceItem
from rag.schema.runtime import AccessPolicy


def _definition() -> AgentDefinition:
    return AgentDefinition(
        agent_type="research",
        description="Research agent",
        system_prompt="Use evidence.",
        allowed_tools=["search"],
        estimated_token_budget=1000,
    )


def _state(run_id: str) -> AgentState:
    config = AgentRunConfig(
        run_id=run_id,
        thread_id=run_id,
        budget_total=1000,
        max_depth=2,
        access_policy=AccessPolicy.default(),
    )
    RuntimeRegistry.remove(run_id)
    RuntimeRegistry.get_or_create(config)
    return {
        "messages": [HumanMessage(content="Recent message", id="m1")],
        "evidence": [
            EvidenceItem(
                evidence_id="ev1",
                doc_id=1,
                citation_anchor="doc#1",
                text="Authoritative text",
                score=0.9,
            )
        ],
        "citations": [
            AnswerCitation(
                citation_id="cit1",
                evidence_id="ev1",
                record_type="section",
                citation_anchor="doc#1",
            )
        ],
        "tool_results": [],
        "task": "Explain policy",
        "run_config": config,
        "iteration": 0,
        "status": "running",
        "decision_reason": None,
        "stop_reason": None,
        "needs_user_input": None,
        "pending_tool_calls": [],
        "approved_tool_call_ids": [],
        "denied_tool_call_ids": [],
        "user_decision": None,
        "working_summary": WorkingSummary(
            summary="Prior task context",
            covered_message_ids=["old"],
            updated_at="2026-05-08T00:00:00Z",
            token_count=3,
        ),
        "extracted_facts": [],
        "context_budget": None,
        "final_answer": None,
        "groundedness_flag": False,
        "insufficient_evidence_flag": False,
    }


class _ContextAwareDecisionProvider:
    def __init__(self) -> None:
        self.context: InjectedContext | None = None

    async def decide(
        self,
        state: AgentState,
        *,
        definition: AgentDefinition,
        budget_remaining: int,
        context: InjectedContext,
    ) -> ThinkOutput:
        del state, definition, budget_remaining
        self.context = context
        return ThinkOutput(action="synthesize", thought="enough", stop_reason="evidence_sufficient")


class _FailIfCalledDecisionProvider:
    def decide(self, *args: object, **kwargs: object) -> ThinkOutput:
        raise AssertionError("decision provider should not be called when context overflows")


@pytest.mark.anyio
async def test_llm_decide_passes_injected_context_to_tool_decision_provider() -> None:
    provider = _ContextAwareDecisionProvider()
    result = await llm_decide_node(
        _state("eval-context"),
        definition=_definition(),
        decision_provider=provider,
    )

    assert result["status"] == "paused"
    assert result["stop_reason"] == "premature_synthesis"
    assert result["context_budget"].evidence_tokens > 0
    assert provider.context is not None
    names = [section.name for section in provider.context.sections]
    assert names.index("evidence") < names.index("working_memory")
    assert "ev1" in provider.context.section("evidence").content
    RuntimeRegistry.remove("eval-context")


@pytest.mark.anyio
async def test_llm_decide_redirects_premature_synthesis_to_llm_summarize() -> None:
    provider = _ContextAwareDecisionProvider()
    state = _state("eval-premature-summary")
    state["tool_results"] = [
        ToolResult(
            tool_call_id="tc-run",
            tool_name="run_python",
            status="ok",
            output=RunPythonOutput(
                ok=True,
                exit_code=0,
                stdout="Total amount: 40.0\n",
                stderr="",
                stdout_truncated=False,
                stderr_truncated=False,
                duration_ms=10.0,
                generated_files=["reports/summary.txt"],
            ),
            latency_ms=0,
        )
    ]
    result = await llm_decide_node(
        state,
        definition=AgentDefinition(
            agent_type="research",
            description="Research agent",
            system_prompt="Use evidence.",
            allowed_tools=["llm_summarize"],
            estimated_token_budget=1000,
        ),
        decision_provider=provider,
    )

    assert result["status"] == "running"
    assert result["controller_next"] == "execute"
    [call] = result["pending_tool_calls"]
    assert call.tool_name == "llm_summarize"
    assert call.arguments["task"] == "Explain policy"
    assert "Total amount: 40.0" in "\n".join(call.arguments["context_sections"])
    RuntimeRegistry.remove("eval-premature-summary")


@pytest.mark.anyio
async def test_llm_decide_pauses_without_calling_provider_on_context_overflow() -> None:
    state = _state("eval-context-overflow")
    state["run_config"] = replace(state["run_config"], max_context_tokens=1)
    state["task"] = "TASK_RAW " * 200

    result = await llm_decide_node(
        state,
        definition=AgentDefinition(
            agent_type="research",
            description="Research agent",
            system_prompt="SYSTEM_RAW " * 200,
            allowed_tools=["search"],
            estimated_token_budget=1000,
        ),
        decision_provider=_FailIfCalledDecisionProvider(),
    )

    assert result["status"] == "paused"
    assert result["decision_reason"] == "context_overflow"
    assert result["context_budget"].overflow is True
    assert "context_overflow" in result["context_budget"].warnings
    RuntimeRegistry.remove("eval-context-overflow")

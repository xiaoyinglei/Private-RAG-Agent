from __future__ import annotations

import pytest
from langchain_core.messages import HumanMessage

from rag.agent.core.context import AgentRunConfig, RuntimeRegistry
from rag.agent.core.definition import AgentDefinition
from rag.agent.graphs.nodes.llm_decide import llm_decide_node
from rag.agent.memory.models import InjectedContext, WorkingSummary
from rag.agent.state import AgentState, ThinkOutput
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


@pytest.mark.anyio
async def test_llm_decide_passes_injected_context_to_tool_decision_provider() -> None:
    provider = _ContextAwareDecisionProvider()
    result = await llm_decide_node(
        _state("eval-context"),
        definition=_definition(),
        decision_provider=provider,
    )

    assert result["status"] == "done"
    assert result["context_budget"].evidence_tokens > 0
    assert provider.context is not None
    names = [section.name for section in provider.context.sections]
    assert names.index("evidence") < names.index("working_memory")
    assert "ev1" in provider.context.section("evidence").content
    RuntimeRegistry.remove("eval-context")

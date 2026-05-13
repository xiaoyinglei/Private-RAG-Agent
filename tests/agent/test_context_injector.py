from __future__ import annotations

from langchain_core.messages import HumanMessage

from rag.agent.core.context import AgentRunConfig
from rag.agent.core.definition import AgentDefinition
from rag.agent.memory.injector import ContextInjector
from rag.agent.memory.models import ExtractedFact, WorkingSummary
from rag.agent.state import AgentState
from rag.agent.tools.spec import ToolError, ToolResult
from rag.schema.query import AnswerCitation, EvidenceItem
from rag.schema.runtime import AccessPolicy


def _definition() -> AgentDefinition:
    return AgentDefinition(
        agent_type="research",
        description="Research agent",
        system_prompt="System prompt",
        allowed_tools=["search"],
    )


def _state() -> AgentState:
    return {
        "messages": [HumanMessage(content="recent tail", id="h-tail")],
        "evidence": [
            EvidenceItem(
                evidence_id="ev1",
                doc_id=1,
                citation_anchor="doc#1",
                text="Authoritative evidence text",
                score=0.91,
                record_type="section",
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
        "tool_results": [
            ToolResult(
                tool_call_id="tc1",
                tool_name="search",
                status="error",
                error=ToolError(code="tool_not_implemented", message="not wired", retryable=False),
                latency_ms=0,
            )
        ],
        "task": "Explain policy",
        "run_config": AgentRunConfig(
            run_id="ctx",
            thread_id="ctx",
            budget_total=1000,
            max_depth=2,
            access_policy=AccessPolicy.default(),
        ),
        "plan": None,
        "iteration": 0,
        "status": "running",
        "route_reason": None,
        "stop_reason": None,
        "needs_user_input": None,
        "pending_tool_calls": [],
        "approved_tool_call_ids": [],
        "denied_tool_call_ids": [],
        "user_decision": None,
        "next_subtasks": None,
        "working_summary": WorkingSummary(
            summary="Prior working summary",
            covered_message_ids=["h1"],
            updated_at="2026-05-08T00:00:00Z",
            token_count=3,
        ),
        "extracted_facts": [
            ExtractedFact(fact_id="f1", text="Memory fact", evidence_ids=["ev1"]),
        ],
        "context_budget": None,
        "subtask_results": {},
        "terminal_subtasks": set(),
        "successful_subtasks": set(),
        "final_answer": None,
        "groundedness_flag": False,
        "insufficient_evidence_flag": False,
    }


def test_context_sections_follow_spec_order() -> None:
    context = ContextInjector(max_context_tokens=1000).assemble(
        definition=_definition(),
        state=_state(),
    )

    names = [section.name for section in context.sections]
    assert names == ["system", "task", "evidence", "working_memory", "message_tail", "tool_results"]
    assert names.index("evidence") < names.index("working_memory")
    assert "ev1" in context.section("evidence").content
    assert "cit1" in context.section("evidence").content


def test_historical_hints_are_marked_non_authoritative() -> None:
    context = ContextInjector(max_context_tokens=1000).assemble(
        definition=_definition(),
        state=_state(),
        recalled_memories=["Old project preference"],
    )

    historical = context.section("historical_hints")
    assert "historical hints, not authoritative evidence" in historical.content
    assert "Old project preference" in historical.content


def test_budget_keeps_evidence_before_tail() -> None:
    state = _state()
    state["messages"] = [HumanMessage(content="tail " * 200, id="h-tail")]

    context = ContextInjector(max_context_tokens=18).assemble(
        definition=_definition(),
        state=state,
    )

    names = [section.name for section in context.sections]
    assert "system" in names
    assert "task" in names
    assert "evidence" in names
    assert "message_tail" not in names
    assert context.context_budget.evidence_tokens > 0


def test_context_budget_snapshot_counts_sections() -> None:
    context = ContextInjector(max_context_tokens=1000).assemble(
        definition=_definition(),
        state=_state(),
    )

    budget = context.context_budget
    assert budget.max_context_tokens == 1000
    assert budget.system_tokens > 0
    assert budget.evidence_tokens > 0
    assert budget.working_memory_tokens > 0
    assert budget.message_tail_tokens > 0
    assert budget.tool_result_tokens > 0

from __future__ import annotations

from pydantic import BaseModel

from rag.agent.core.context import AgentRunConfig
from rag.agent.state import (
    AgentState,
    ToolCallPlan,
    _merge_citations,
    _merge_evidence,
    _merge_tool_results,
    create_agent_state,
)
from rag.schema.query import AnswerCitation, EvidenceItem
from rag.schema.runtime import AccessPolicy


def _run_config(run_id: str) -> AgentRunConfig:
    return AgentRunConfig(
        run_id=run_id,
        thread_id=run_id,
        budget_total=100,
        max_depth=2,
        access_policy=AccessPolicy.default(),
    )


def test_create_agent_state_populates_every_required_channel() -> None:
    state = create_agent_state(
        task="Inspect architecture",
        run_config=_run_config("state-complete"),
    )

    assert set(state) == set(AgentState.__required_keys__)


def test_create_agent_state_copies_mutable_inputs_and_defaults() -> None:
    pending = [ToolCallPlan.create("vector_search", {"query": "agent loop"})]
    first = create_agent_state(
        task="First",
        run_config=_run_config("state-first"),
        pending_tool_calls=pending,
    )
    second = create_agent_state(
        task="Second",
        run_config=_run_config("state-second"),
    )

    pending.clear()
    first["memory_warnings"].append("bounded")

    assert len(first["pending_tool_calls"]) == 1
    assert second["pending_tool_calls"] == []
    assert second["memory_warnings"] == []


class TestMergeEvidence:
    def test_dedup_by_evidence_id(self) -> None:
        a = EvidenceItem(evidence_id="e1", doc_id=1, citation_anchor="a", text="A", score=0.8)
        b = EvidenceItem(evidence_id="e1", doc_id=1, citation_anchor="a", text="A better", score=0.9)
        merged = _merge_evidence([a], [b])
        assert len(merged) == 1
        assert merged[0].score == 0.9

    def test_conflict_preserves_both(self) -> None:
        a = EvidenceItem(evidence_id="e1", doc_id=1, citation_anchor="a", text="Alpha is good", score=0.8)
        b = EvidenceItem(evidence_id="e1", doc_id=1, citation_anchor="a", text="Alpha is not good", score=0.9)
        merged = _merge_evidence([a], [b])
        assert len(merged) == 2
        assert any("conflict" in item.retrieval_channels for item in merged)


class TestMergeCitations:
    def test_dedup_by_citation_id(self) -> None:
        a = AnswerCitation(citation_id="c1", evidence_id="e1", record_type="section")
        b = AnswerCitation(citation_id="c1", evidence_id="e2", record_type="asset")
        merged = _merge_citations([a], [b])
        assert len(merged) == 1
        assert merged[0].evidence_id == "e2"


class TestMergeToolResults:
    def test_dedup_by_tool_call_id(self) -> None:
        from rag.agent.tools.spec import ToolResult

        class DummyOutput(BaseModel):
            value: str

        r1 = ToolResult(
            tool_call_id="tc1",
            tool_name="search",
            status="ok",
            output=DummyOutput(value="old"),
            latency_ms=10,
        )
        r2 = ToolResult(
            tool_call_id="tc1",
            tool_name="search",
            status="ok",
            output=DummyOutput(value="new"),
            latency_ms=20,
        )
        merged = _merge_tool_results([r1], [r2])
        assert len(merged) == 1
        assert merged[0].output == DummyOutput(value="new")

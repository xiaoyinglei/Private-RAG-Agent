from __future__ import annotations

from pydantic import BaseModel

from rag.agent.state import (
    _merge_citations,
    _merge_evidence,
    _merge_tool_results,
)
from rag.schema.query import AnswerCitation, EvidenceItem


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

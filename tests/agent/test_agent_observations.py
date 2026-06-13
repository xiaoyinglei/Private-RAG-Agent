from __future__ import annotations

from pydantic import BaseModel

from rag.agent.core.observations import (
    ComputationResult,
    ContextUnit,
    EvidenceRef,
    ObservationExtractor,
    StructuredObservation,
)
from rag.agent.tools.asset_tools import AssetAnalyzeOutput
from rag.agent.tools.rag_answer_tools import RAGSearchAnswerOutput
from rag.agent.tools.rag_tools import SearchOutput
from rag.agent.tools.spec import ToolError, ToolResult
from rag.schema.query import AnswerCitation, EvidenceItem


class _TextOutput(BaseModel):
    text: str


def test_neutral_rag_observation_preserves_evidence_and_citations() -> None:
    evidence = EvidenceItem(
        evidence_id="ev-policy",
        doc_id=7,
        citation_anchor="policy#3",
        text="Policy evidence",
        score=0.91,
        page_start=3,
        page_end=4,
        retrieval_channels=["vector", "rerank"],
    )
    citation = AnswerCitation(
        citation_id="cit-policy",
        evidence_id="ev-policy",
        record_type="section",
        citation_anchor="policy#3",
        doc_id=7,
        page_start=3,
        page_end=4,
    )
    result = ToolResult(
        tool_call_id="tc-rag",
        tool_name="rag_search_answer",
        status="ok",
        output=RAGSearchAnswerOutput(
            text="Grounded answer",
            evidence=[evidence],
            citations=[citation],
            groundedness_flag=True,
        ),
        latency_ms=0,
    )

    update = ObservationExtractor().reduce_tool_results({"tool_results": [result]})

    assert update["evidence"] == [evidence]
    assert update["citations"] == [citation]
    assert update["evidence_refs"] == [
        EvidenceRef(
            evidence_id="ev-policy",
            citation_anchor="policy#3",
            doc_id=7,
            source="evidence",
        ),
        EvidenceRef(
            evidence_id="ev-policy",
            citation_id="cit-policy",
            citation_anchor="policy#3",
            doc_id=7,
            source="citation",
        ),
    ]
    assert update["answer_candidates"][0].text == "Grounded answer"


def test_retrieval_observation_preserves_score_rerank_score_and_locator() -> None:
    result = ToolResult(
        tool_call_id="tc-search",
        tool_name="rerank",
        status="ok",
        output=SearchOutput(
            items=[
                {
                    "text": "Ranked evidence",
                    "doc_id": 8,
                    "section_id": 5,
                    "page_start": 2,
                    "page_end": 2,
                    "record_type": "section",
                    "citation_anchor": "doc#5",
                    "evidence_id": "ev-ranked",
                    "score": 0.73,
                    "rerank_score": 0.98,
                    "retrieval_channels": ["vector", "rerank"],
                }
            ]
        ),
        latency_ms=0,
    )

    update = ObservationExtractor().reduce_tool_results({"tool_results": [result]})

    [unit] = update["context_units"]
    assert unit == ContextUnit(
        unit_id="retrieval:ev-ranked",
        unit_type="document_section",
        locator={
            "doc_id": 8,
            "section_id": 5,
            "page_start": 2,
            "page_end": 2,
            "record_type": "section",
            "citation_anchor": "doc#5",
            "evidence_id": "ev-ranked",
            "score": 0.73,
            "rerank_score": 0.98,
            "retrieval_channels": ["vector", "rerank"],
        },
        preview="Ranked evidence",
        content_ref="ev-ranked",
        evidence_refs=[
            EvidenceRef(
                evidence_id="ev-ranked",
                citation_anchor="doc#5",
                doc_id=8,
                source="retrieval",
            )
        ],
        capabilities=["text_extract", "text_synthesize", "quote"],
        metadata={"source_tool": "rerank"},
    )


def test_computation_observation_preserves_expression_and_asset_provenance() -> None:
    result = ToolResult(
        tool_call_id="tc-compute",
        tool_name="asset_analyze",
        status="ok",
        output=AssetAnalyzeOutput(
            asset_id=14,
            asset_type="table",
            sheet_name="Sales",
            operation="dataframe_sql",
            columns=["total"],
            rows=[["15.49"]],
            raw_row_count=1,
            elapsed_ms=1.0,
            truncated=False,
            query="SELECT SUM(amount) AS total FROM sheet",
            markdown="| total |\n|---|\n| 15.49 |",
        ),
        latency_ms=0,
    )

    update = ObservationExtractor().reduce_tool_results({"tool_results": [result]})

    assert update["computation_results"] == [
        ComputationResult(
            source_tool_call_id="tc-compute",
            source_tool_name="asset_analyze",
            operation="dataframe_sql",
            value_preview="| total |\n|---|\n| 15.49 |",
            expression="SELECT SUM(amount) AS total FROM sheet",
            evidence_refs=[EvidenceRef(evidence_id="asset:14", source="asset")],
        )
    ]
    assert update["locators"] == [
        {
            "asset_id": 14,
            "asset_type": "table",
            "sheet_name": "Sales",
            "columns": ["total"],
        }
    ]
    assert update["asset_refs"] == [14]


def test_structured_tool_error_is_visible_without_controller_fields() -> None:
    result = ToolResult(
        tool_call_id="tc-error",
        tool_name="vector_search",
        status="error",
        error=ToolError(
            code="timeout",
            message="retrieval timed out",
            retryable=True,
            detail={"provider": "vector"},
        ),
        latency_ms=100,
    )

    update = ObservationExtractor().reduce_tool_results({"tool_results": [result]})

    assert update["structured_observations"] == [
        StructuredObservation(
            tool_call_id="tc-error",
            tool_name="vector_search",
            status="error",
            error="retrieval timed out",
            raw_result_ref="tc-error",
        )
    ]
    assert update["errors"] == [
        {
            "tool_call_id": "tc-error",
            "tool_name": "vector_search",
            "code": "timeout",
            "message": "retrieval timed out",
            "retryable": True,
            "detail": {"provider": "vector"},
        }
    ]
    assert {
        "satisfied_requirements",
        "open_gaps",
        "no_progress_count",
        "iteration",
        "controller_next",
    }.isdisjoint(update)


def test_neutral_reducer_skips_already_observed_tool_calls() -> None:
    result = ToolResult(
        tool_call_id="tc-text",
        tool_name="llm_summarize",
        status="ok",
        output=_TextOutput(text="Summary"),
        latency_ms=0,
    )
    existing = StructuredObservation(
        tool_call_id="tc-text",
        tool_name="llm_summarize",
        status="ok",
        raw_result_ref="tc-text",
    )

    update = ObservationExtractor().reduce_tool_results(
        {
            "tool_results": [result],
            "structured_observations": [existing],
        }
    )

    assert update == {}

from __future__ import annotations

from rag.agent.report import AgentReportBuilder
from rag.agent.schema import CriticAction, EvidenceAssessment, ReportCitation, SubTask, SubTaskResult
from rag.retrieval.models import ContextEvidence
from rag.schema.core import PartitionKey, StorageTier
from rag.schema.query import AnswerCitation, EvidenceItem


def test_context_evidence_extends_evidence_item_contract() -> None:
    context_evidence = ContextEvidence(
        evidence_id="ev-1",
        doc_id=1,
        citation_anchor="doc#1",
        text="Alpha evidence",
        score=0.9,
        token_count=3,
        selected_token_count=3,
    )

    evidence_item = context_evidence.as_evidence_item()

    assert isinstance(context_evidence, EvidenceItem)
    assert evidence_item == EvidenceItem(
        evidence_id="ev-1",
        doc_id=1,
        citation_anchor="doc#1",
        text="Alpha evidence",
        score=0.9,
    )
    assert "token_count" not in evidence_item.model_dump()


def test_report_citation_reuses_answer_citation_contract() -> None:
    evidence = EvidenceItem(
        evidence_id="ev-1",
        doc_id=1,
        citation_anchor="doc#1",
        text="Alpha evidence",
        score=0.9,
        record_type="section",
        file_name="alpha.md",
    )
    subtask = SubTask(
        subtask_id="task-1",
        objective="Find alpha evidence",
        instruction="Find alpha evidence",
    )
    result = SubTaskResult(
        subtask=subtask,
        findings=["Alpha evidence"],
        evidence=[evidence],
        evidence_assessment=EvidenceAssessment(
            sufficient=True,
            recommended_action=CriticAction.ACCEPT,
        ),
    )

    citations = AgentReportBuilder._citations([result])

    assert len(citations) == 1
    assert isinstance(citations[0], AnswerCitation)
    assert isinstance(citations[0], ReportCitation)
    assert citations[0].evidence_id == "ev-1"
    assert citations[0].chunk_id == "ev-1"
    assert citations[0].record_type == "section"


def test_partition_key_is_storage_tier_alias() -> None:
    assert PartitionKey is StorageTier
    assert PartitionKey.HOT is StorageTier.HOT
    assert PartitionKey.COLD.value == "cold"

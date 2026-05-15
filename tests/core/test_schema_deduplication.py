from __future__ import annotations

from rag.agent.core.task import SubTaskNode, SubTaskResult, SubTaskStatus
from rag.retrieval.models import ContextEvidence
from rag.schema.core import PartitionKey, StorageTier
from rag.schema.query import EvidenceItem


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


def test_subtask_result_reuses_evidence_item_contract() -> None:
    evidence = EvidenceItem(
        evidence_id="ev-1",
        doc_id=1,
        citation_anchor="doc#1",
        text="Alpha evidence",
        score=0.9,
        record_type="section",
        file_name="alpha.md",
    )
    subtask = SubTaskNode(
        subtask_id="task-1",
        agent_type="research",
        prompt="Find alpha evidence",
        priority=1,
    )
    result = SubTaskResult(
        subtask=subtask,
        status=SubTaskStatus.COMPLETED,
        findings=["Alpha evidence"],
        evidence=[evidence],
    )

    assert result.evidence == [evidence]
    assert result.evidence[0].record_type == "section"


def test_partition_key_is_independent_index_layer_enum() -> None:
    assert PartitionKey is not StorageTier
    assert PartitionKey.HOT is not StorageTier.HOT
    assert PartitionKey.HOT.value == StorageTier.HOT.value
    assert PartitionKey.COLD.value == "cold"

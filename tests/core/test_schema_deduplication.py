from __future__ import annotations

from rag.retrieval.models import ContextEvidence
from rag.schema.core import PartitionKey, StorageTier
from rag.schema.query import ArtifactStatus, EvidenceItem, KnowledgeArtifact


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


def test_partition_key_is_independent_index_layer_enum() -> None:
    assert PartitionKey is not StorageTier
    assert PartitionKey.HOT is not StorageTier.HOT
    assert PartitionKey.HOT.value == StorageTier.HOT.value
    assert PartitionKey.COLD.value == "cold"


def test_knowledge_artifact_status_is_backward_compatible_optional_field() -> None:
    artifact = KnowledgeArtifact(
        artifact_id="artifact-1",
        artifact_type="note",
        title="Evidence note",
        supported_evidence_ids=["ev-1"],
        last_reviewed_at="2026-05-17T00:00:00Z",
        body_markdown="Supported by evidence.",
    )
    legacy_artifact = KnowledgeArtifact(
        artifact_id="artifact-2",
        artifact_type="note",
        title="Approved evidence note",
        supported_evidence_ids=["ev-1"],
        status=ArtifactStatus.APPROVED,
        last_reviewed_at="2026-05-17T00:00:00Z",
        body_markdown="Supported by evidence.",
    )

    assert artifact.status is None
    assert legacy_artifact.status is ArtifactStatus.APPROVED

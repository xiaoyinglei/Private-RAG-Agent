from __future__ import annotations

from dataclasses import dataclass, field

from rag.retrieval.synthesis_service import SynthesisService
from rag.schema.core import Document, DocumentType
from rag.schema.query import EvidenceItem
from rag.schema.runtime import AccessPolicy


@dataclass
class _MetadataRepo:
    documents: dict[int, Document] = field(default_factory=dict)

    def get_document(self, doc_id: int) -> Document | None:
        return self.documents.get(doc_id)


def _document(doc_id: int) -> Document:
    return Document(
        doc_id=doc_id,
        source_id=1,
        doc_type=DocumentType.REPORT,
        file_hash=f"hash-{doc_id}",
        version_group_id=doc_id,
        is_active=True,
        index_ready=True,
        embedding_model_id="bge-m3",
    )


def test_synthesis_service_denies_missing_integer_doc_id_without_crashing() -> None:
    service = SynthesisService(metadata_repo=_MetadataRepo())

    filtered = service.filter_evidence(
        evidence=[
            EvidenceItem(
                evidence_id="E1",
                doc_id=99,
                citation_anchor="Missing",
                text="missing evidence",
                score=0.4,
            )
        ],
        access_policy=AccessPolicy.default(),
    )

    assert filtered == []


def test_synthesis_service_allows_visible_new_contract_evidence() -> None:
    service = SynthesisService(metadata_repo=_MetadataRepo(documents={1: _document(1)}))

    filtered = service.filter_evidence(
        evidence=[
            EvidenceItem(
                evidence_id="E1",
                doc_id=1,
                citation_anchor="Allowed",
                text="allowed evidence",
                score=0.9,
            )
        ],
        access_policy=AccessPolicy.default(),
    )

    assert [item.evidence_id for item in filtered] == ["E1"]

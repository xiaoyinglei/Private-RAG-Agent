from __future__ import annotations

import logging
from datetime import UTC, datetime

import pytest

from rag.schema.core import (
    AssetRecord,
    Document,
    ProcessingStateRecord,
    SectionLocatorRecord,
    SectionRecord,
    Source,
    SourceType,
)
from rag.storage.data_contract_service import DataContractService
from rag.storage.index_sync_service import StaleProcessingStateError


class _FakeMetadataRepo:
    def __init__(self) -> None:
        self.existing_document: Document | None = None
        self.existing_source: Source | None = None
        self.saved_sources: list[Source] = []
        self.saved_documents: list[Document] = []
        self.saved_sections: list[SectionRecord] = []
        self.saved_assets: list[AssetRecord] = []
        self.incremented_doc_ids: list[int] = []
        self.index_state_calls: list[tuple[int, bool | None, bool | None, str | None]] = []
        self.index_state_errors: list[str | None] = []
        self.deactivated_doc_ids: list[int] = []
        self.processing_states: dict[int, ProcessingStateRecord] = {}

    def find_document_by_hash(self, file_hash: str):
        return self.existing_document

    def increment_document_reference_count(self, doc_id: int, *, amount: int = 1) -> Document:
        self.incremented_doc_ids.append(doc_id)
        assert self.existing_document is not None
        self.existing_document = self.existing_document.model_copy(
            update={"reference_count": self.existing_document.reference_count + amount}
        )
        return self.existing_document

    def save_source(self, source: Source) -> Source:
        saved = source.model_copy(update={"source_id": 10})
        self.saved_sources.append(saved)
        self.existing_source = saved
        return saved

    def get_source(self, source_id: int) -> Source | None:
        if self.existing_source is not None and self.existing_source.source_id == source_id:
            return self.existing_source
        if self.saved_sources and self.saved_sources[-1].source_id == source_id:
            return self.saved_sources[-1]
        return None

    def save_document(self, document: Document) -> Document:
        saved = document.model_copy(update={"doc_id": 20, "version_group_id": 20})
        self.saved_documents.append(saved)
        self.existing_document = saved
        return saved

    def get_document(self, doc_id: int) -> Document | None:
        if self.existing_document is not None and self.existing_document.doc_id == doc_id:
            return self.existing_document
        if self.saved_documents and self.saved_documents[-1].doc_id == doc_id:
            return self.saved_documents[-1]
        return None

    def save_section(self, section: SectionRecord) -> SectionRecord:
        saved = section.model_copy(update={"section_id": 30})
        self.saved_sections.append(saved)
        return saved

    def list_sections(self, *, doc_id: int | None = None, source_id: int | None = None) -> list[SectionRecord]:
        sections = list(self.saved_sections)
        if doc_id is not None:
            sections = [section for section in sections if section.doc_id == doc_id]
        if source_id is not None:
            sections = [section for section in sections if section.source_id == source_id]
        return sections

    def save_asset(self, asset: AssetRecord) -> AssetRecord:
        saved = asset.model_copy(update={"asset_id": 40})
        self.saved_assets.append(saved)
        return saved

    def list_assets(
        self,
        *,
        doc_id: int | None = None,
        source_id: int | None = None,
        section_id: int | None = None,
    ) -> list[AssetRecord]:
        assets = list(self.saved_assets)
        if doc_id is not None:
            assets = [asset for asset in assets if asset.doc_id == doc_id]
        if source_id is not None:
            assets = [asset for asset in assets if asset.source_id == source_id]
        if section_id is not None:
            assets = [asset for asset in assets if asset.section_id == section_id]
        return assets

    def set_document_index_state(
        self,
        doc_id: int,
        *,
        is_indexed: bool | None = None,
        index_ready: bool | None = None,
        embedding_model_id: str | None = None,
        indexed_at=None,
        last_index_error: str | None = None,
    ) -> Document:
        self.index_state_calls.append((doc_id, is_indexed, index_ready, embedding_model_id))
        self.index_state_errors.append(last_index_error)
        return self.saved_documents[-1] if self.saved_documents else self.existing_document  # type: ignore[return-value]

    def deactivate_document(self, doc_id: int) -> Document:
        self.deactivated_doc_ids.append(doc_id)
        assert self.existing_document is not None
        self.existing_document = self.existing_document.model_copy(update={"is_active": False})
        return self.existing_document

    def save_processing_state(self, record: ProcessingStateRecord) -> ProcessingStateRecord:
        self.processing_states[record.doc_id] = record
        return record

    def get_processing_state(self, doc_id: int) -> ProcessingStateRecord | None:
        return self.processing_states.get(doc_id)

    def list_processing_states(
        self,
        *,
        source_id: int | None = None,
        status: str | None = None,
        stage: str | None = None,
    ) -> list[ProcessingStateRecord]:
        states = list(self.processing_states.values())
        if source_id is not None:
            states = [state for state in states if state.source_id == source_id]
        if status is not None:
            states = [state for state in states if state.status == status]
        if stage is not None:
            states = [state for state in states if state.stage == stage]
        return sorted(states, key=lambda state: (state.updated_at, state.doc_id))


def _section_record(**kwargs) -> SectionRecord:
    byte_start = int(kwargs.setdefault("byte_range_start", 0))
    byte_end = int(kwargs.setdefault("byte_range_end", max(byte_start + 1, 1)))
    char_start = int(kwargs.setdefault("char_range_start", byte_start))
    char_end = int(kwargs.setdefault("char_range_end", byte_end))
    visible_text_key = str(kwargs.setdefault("visible_text_key", f"doc-{kwargs.get('doc_id', 'unknown')}.txt"))
    kwargs.setdefault(
        "raw_locator",
        SectionLocatorRecord(
            visible_text_key=visible_text_key,
            char_range_start=char_start,
            char_range_end=char_end,
            byte_range_start=byte_start,
            byte_range_end=byte_end,
        ),
    )
    return SectionRecord(**kwargs)


class _FakeMilvusRepo:
    def __init__(self) -> None:
        self.upsert_calls: list[tuple[object, list[float], str]] = []
        self.batch_upsert_calls: list[tuple[list[tuple[object, list[float]]], str]] = []
        self.delete_calls: list[str] = []
        self.fail_delete = False
        self.fail_upsert = False
        self.search_calls: list[tuple[list[float], dict[str, object]]] = []

    def upsert_record(self, record, vector, *, embedding_space: str = "default") -> None:
        if self.fail_upsert:
            raise RuntimeError("upsert failed")
        self.upsert_calls.append((record, list(vector), embedding_space))

    def upsert_records(self, items, *, embedding_space: str = "default") -> None:
        if self.fail_upsert:
            raise RuntimeError("upsert failed")
        normalized = [(record, list(vector)) for record, vector in items]
        self.batch_upsert_calls.append((normalized, embedding_space))

    def delete(self, *, expr: str, item_kind=None, embedding_space=None) -> int:
        self.delete_calls.append(expr)
        if self.fail_delete:
            raise RuntimeError("delete failed")
        return 1

    def search(self, query_vector, **kwargs):
        self.search_calls.append((list(query_vector), dict(kwargs)))
        return []


class _FakeEmbedder:
    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(text))] for text in texts]


def _build_document(*, doc_id: int = 99, file_hash: str = "hash-1") -> Document:
    now = datetime(2026, 4, 19, tzinfo=UTC)
    return Document(
        doc_id=doc_id,
        source_id=10,
        title="Call 13812345678",

        language="zh",
        authors=["alice"],
        file_hash=file_hash,
        version_group_id=doc_id,
        doc_status="published",
        is_active=True,
        embedding_model_id="bge-m3",
        created_at=now,
        updated_at=now,
    )


def test_data_contract_service_deduplicates_file_hash_and_returns_early() -> None:
    metadata_repo = _FakeMetadataRepo()
    metadata_repo.existing_document = _build_document()
    service = DataContractService(metadata_repo, _FakeMilvusRepo(), embedder=_FakeEmbedder())

    result = service.register_document(
        source=Source(source_type=SourceType.MARKDOWN, location="docs/a.md", content_hash=""),
        document=_build_document(doc_id=0, file_hash=""),
        file_bytes=b"same-file",
    )

    assert result.is_duplicate is True
    assert result.source is None
    assert result.document.doc_id == 99
    assert metadata_repo.incremented_doc_ids == [99]
    assert metadata_repo.saved_sources == []
    assert metadata_repo.saved_documents == []


def test_data_contract_service_skips_embedding_when_not_urgent() -> None:
    metadata_repo = _FakeMetadataRepo()
    milvus_repo = _FakeMilvusRepo()
    document = _build_document(doc_id=20, file_hash="hash-2")
    metadata_repo.saved_documents.append(document)
    service = DataContractService(metadata_repo, milvus_repo, embedder=_FakeEmbedder())

    saved_section = service.save_section(
        document,
        _section_record(
            doc_id=document.doc_id,
            source_id=document.source_id,
            order_index=0,
            section_kind="body",
            content_hash="section-hash",
        ),
        source_type=SourceType.MARKDOWN,
        summary_text="联系我 13812345678",
        is_urgent=False,
    )

    assert saved_section.section_id == 30
    assert metadata_repo.saved_sections[-1].metadata_json["summary_text"].startswith("Semantic Core:")
    assert milvus_repo.upsert_calls == []
    assert metadata_repo.index_state_calls[-1] == (20, False, False, "bge-m3")


def test_data_contract_service_deactivate_document_logs_milvus_failures(caplog: pytest.LogCaptureFixture) -> None:
    metadata_repo = _FakeMetadataRepo()
    metadata_repo.existing_document = _build_document()
    milvus_repo = _FakeMilvusRepo()
    milvus_repo.fail_delete = True
    service = DataContractService(metadata_repo, milvus_repo, logger=logging.getLogger("data-contract-test"))

    with caplog.at_level(logging.ERROR):
        document = service.deactivate_document(99)

    assert document.is_active is False
    assert metadata_repo.deactivated_doc_ids == [99]
    assert milvus_repo.delete_calls == ["doc_id in [99]"]
    assert "failed to delete milvus vectors" in caplog.text


def test_data_contract_service_marks_document_pending_before_visible_index_flip() -> None:
    metadata_repo = _FakeMetadataRepo()
    milvus_repo = _FakeMilvusRepo()
    document = _build_document(doc_id=20, file_hash="hash-2")
    metadata_repo.saved_documents.append(document)
    service = DataContractService(metadata_repo, milvus_repo, embedder=_FakeEmbedder())

    service.save_section(
        document,
        _section_record(
            doc_id=document.doc_id,
            source_id=document.source_id,
            order_index=0,
            section_kind="body",
            content_hash="section-hash",
        ),
        source_type=SourceType.MARKDOWN,
        summary_text="Alpha section summary",
        is_urgent=True,
    )

    assert metadata_repo.index_state_calls[:2] == [
        (20, False, False, "bge-m3"),
        (20, True, True, "bge-m3"),
    ]


def test_data_contract_service_enqueues_background_sync_for_non_urgent_records() -> None:
    metadata_repo = _FakeMetadataRepo()
    milvus_repo = _FakeMilvusRepo()
    document = _build_document(doc_id=20, file_hash="hash-2")
    metadata_repo.saved_documents.append(document)
    metadata_repo.existing_document = document
    service = DataContractService(metadata_repo, milvus_repo, embedder=_FakeEmbedder())

    service.save_section(
        document,
        _section_record(
            doc_id=document.doc_id,
            source_id=document.source_id,
            order_index=0,
            section_kind="body",
            content_hash="section-hash",
        ),
        source_type=SourceType.MARKDOWN,
        summary_text="Alpha section summary",
        is_urgent=False,
    )

    state = metadata_repo.get_processing_state(20)
    assert state is not None
    assert state.stage == "index_sync"
    assert state.status == "pending"
    assert state.metadata_json["operation"] == "upsert_summary"
    assert state.metadata_json["item_kind"] == "section_summary"
    assert isinstance(state.metadata_json["commit_anchor"], str)


def test_data_contract_service_sync_processing_state_rebuilds_vectors_and_flips_ready_state() -> None:
    metadata_repo = _FakeMetadataRepo()
    metadata_repo.existing_source = Source(
        source_id=10,
        source_type=SourceType.MARKDOWN,
        location="docs/a.md",
        content_hash="hash-2",
    )
    document = _build_document(doc_id=20, file_hash="hash-2").model_copy(
        update={
            "metadata_json": {"summary_text": "Document summary"},
            "is_indexed": False,
            "index_ready": False,
        }
    )
    metadata_repo.saved_documents.append(document)
    metadata_repo.existing_document = document
    metadata_repo.saved_sections.append(
        _section_record(
            section_id=30,
            doc_id=20,
            source_id=10,
            order_index=0,
            section_kind="body",
            content_hash="section-hash",
            metadata_json={"summary_text": "Section summary"},
        )
    )
    metadata_repo.saved_assets.append(
        AssetRecord(
            asset_id=40,
            doc_id=20,
            source_id=10,
            section_id=30,
            asset_type="table",
            page_no=1,
            content_hash="asset-hash",
            storage_key="assets/table-1.png",
            metadata_json={"summary_text": "Asset summary"},
        )
    )
    milvus_repo = _FakeMilvusRepo()
    service = DataContractService(metadata_repo, milvus_repo, embedder=_FakeEmbedder())
    state = ProcessingStateRecord(
        doc_id=20,
        source_id=10,
        stage="index_sync",
        status="pending",
        metadata_json={"operation": "upsert_summary", "embedding_space": "default"},
    )

    count = service.sync_processing_state(state)

    assert count == 3
    assert milvus_repo.delete_calls[-1] == "doc_id in [20]"
    assert len(milvus_repo.batch_upsert_calls) == 1
    records, embedding_space = milvus_repo.batch_upsert_calls[0]
    assert embedding_space == "default"
    assert [type(record).__name__ for record, _vector in records] == [
        "DocSummaryRecord",
        "SectionSummaryRecord",
        "AssetSummaryRecord",
    ]
    assert records[0][0].summary_text.startswith("Semantic Core:")
    assert "Fact Anchors:" in records[1][0].summary_text
    assert metadata_repo.index_state_calls[-1] == (20, True, True, "bge-m3")


def test_data_contract_service_keeps_document_non_visible_when_milvus_upsert_fails() -> None:
    metadata_repo = _FakeMetadataRepo()
    milvus_repo = _FakeMilvusRepo()
    milvus_repo.fail_upsert = True
    document = _build_document(doc_id=20, file_hash="hash-2")
    metadata_repo.saved_documents.append(document)
    service = DataContractService(metadata_repo, milvus_repo, embedder=_FakeEmbedder())

    with pytest.raises(RuntimeError, match="upsert failed"):
        service.save_section(
            document,
            _section_record(
                doc_id=document.doc_id,
                source_id=document.source_id,
                order_index=0,
                section_kind="body",
                content_hash="section-hash",
            ),
            source_type=SourceType.MARKDOWN,
            summary_text="Alpha section summary",
            is_urgent=True,
        )

    assert metadata_repo.index_state_calls[-1] == (20, False, False, "bge-m3")
    assert metadata_repo.index_state_errors[-1] == "upsert failed"


def test_data_contract_service_standardizes_summary_contract_fields() -> None:
    metadata_repo = _FakeMetadataRepo()
    milvus_repo = _FakeMilvusRepo()
    document = _build_document(doc_id=20, file_hash="hash-2")
    metadata_repo.saved_documents.append(document)
    service = DataContractService(metadata_repo, milvus_repo, embedder=_FakeEmbedder())

    saved_section = service.save_section(
        document,
        _section_record(
            doc_id=document.doc_id,
            source_id=document.source_id,
            order_index=0,
            toc_path=["Architecture", "Alpha"],
            page_start=2,
            page_end=3,
            anchor="architecture-alpha",
            section_kind="body",
            content_hash="section-hash",
        ),
        source_type=SourceType.MARKDOWN,
        summary_text="Alpha handles ingestion and routing.",
        is_urgent=False,
    )

    assert saved_section.metadata_json["summary_text"].startswith("Semantic Core:")
    contract = saved_section.metadata_json["summary_contract"]
    assert contract["spec_version"] == "v1"
    assert contract["kind"] == "section"
    assert contract["fact_anchors"]
    assert contract["structural_hint"]


def test_data_contract_service_rejects_stale_commit_anchor() -> None:
    metadata_repo = _FakeMetadataRepo()
    milvus_repo = _FakeMilvusRepo()
    document = _build_document(doc_id=20, file_hash="hash-2")
    metadata_repo.saved_documents.append(document)
    metadata_repo.existing_document = document
    service = DataContractService(metadata_repo, milvus_repo, embedder=_FakeEmbedder())

    service.save_section(
        document,
        _section_record(
            doc_id=document.doc_id,
            source_id=document.source_id,
            order_index=0,
            section_kind="body",
            content_hash="section-hash",
        ),
        source_type=SourceType.MARKDOWN,
        summary_text="Alpha section summary",
        is_urgent=False,
    )
    stale_state = metadata_repo.get_processing_state(20)
    assert stale_state is not None
    metadata_repo.save_processing_state(
        stale_state.model_copy(update={"metadata_json": {**stale_state.metadata_json, "commit_anchor": "newer-anchor"}})
    )

    with pytest.raises(StaleProcessingStateError):
        service.sync_processing_state(stale_state)

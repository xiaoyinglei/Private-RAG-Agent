from __future__ import annotations

import pytest

from rag.schema.core import SourceType
from tests.support import make_runtime


def test_runtime_delete_deactivates_document_and_removes_summary_vectors() -> None:
    runtime = make_runtime()
    try:
        inserted = runtime.insert(
            source_type=SourceType.PLAIN_TEXT.value,
            location="memory://alpha-engine",
            owner="user",
            title="Alpha Engine",
            content_text="Alpha Engine processes ingestion requests.",
        )

        before = runtime.query("What does Alpha Engine process?")
        deleted = runtime.delete(location="memory://alpha-engine")
        after = runtime.query("What does Alpha Engine process?")

        document = runtime.stores.metadata_repo.get_document(inserted.doc_id)
        state = runtime.stores.metadata_repo.get_processing_state(inserted.doc_id)
        section_vectors = runtime.stores.vector_repo.count_vectors(item_kind="section_summary")
        doc_vectors = runtime.stores.vector_repo.count_vectors(item_kind="doc_summary")
    finally:
        runtime.close()

    assert before.retrieval.evidence.internal
    assert deleted.deleted_doc_ids == [inserted.doc_id]
    assert deleted.deleted_source_ids == [inserted.source_id]
    assert deleted.deleted_vector_count >= 2
    assert document is not None
    assert document.is_active is False
    assert document.index_ready is False
    assert state is not None
    assert state.stage == "delete"
    assert state.status == "deleted"
    assert section_vectors == 0
    assert doc_vectors == 0
    assert not after.retrieval.evidence.internal


def test_runtime_rebuild_restores_deleted_document_from_source_object_key() -> None:
    runtime = make_runtime()
    try:
        inserted = runtime.insert(
            source_type=SourceType.PLAIN_TEXT.value,
            location="memory://rebuildable-note",
            owner="user",
            title="Rebuildable Note",
            content_text="Gamma Index stores section summary vectors for retrieval.",
        )
        runtime.delete(location="memory://rebuildable-note")

        rebuilt = runtime.rebuild(location="memory://rebuildable-note")
        after = runtime.query("What does Gamma Index store?")

        document = runtime.stores.metadata_repo.get_document(inserted.doc_id)
        state = runtime.stores.metadata_repo.get_processing_state(inserted.doc_id)
        source = runtime.stores.metadata_repo.get_source(inserted.source_id)
        section_vectors = runtime.stores.vector_repo.count_vectors(item_kind="section_summary")
        doc_vectors = runtime.stores.vector_repo.count_vectors(item_kind="doc_summary")
    finally:
        runtime.close()

    assert rebuilt.rebuilt_doc_ids == [inserted.doc_id]
    assert rebuilt.results[0].doc_id == inserted.doc_id
    assert rebuilt.results[0].indexed_object_count >= 2
    assert document is not None
    assert document.is_active is True
    assert document.index_ready is True
    assert state is not None
    assert state.stage == "index"
    assert state.status == "ready"
    assert source is not None
    assert source.object_key
    assert section_vectors == 1
    assert doc_vectors == 1
    assert after.retrieval.evidence.internal


def test_runtime_rebuild_marks_document_failed_when_source_payload_is_missing() -> None:
    runtime = make_runtime()
    try:
        inserted = runtime.insert(
            source_type=SourceType.PLAIN_TEXT.value,
            location="memory://broken-rebuild",
            owner="user",
            content_text="Alpha Engine processes ingestion requests.",
        )
        source = runtime.stores.metadata_repo.get_source(inserted.source_id)
        assert source is not None
        assert source.object_key is not None
        runtime.delete(location="memory://broken-rebuild")
        runtime.stores.object_store.path_for_key(source.object_key).unlink()

        with pytest.raises(ValueError, match="No rebuildable source payload available"):
            runtime.rebuild(location="memory://broken-rebuild")

        document = runtime.stores.metadata_repo.get_document(inserted.doc_id)
        state = runtime.stores.metadata_repo.get_processing_state(inserted.doc_id)
    finally:
        runtime.close()

    assert document is not None
    assert document.index_ready is False
    assert document.last_index_error is not None
    assert "No rebuildable source payload available" in document.last_index_error
    assert state is not None
    assert state.stage == "rebuild"
    assert state.status == "failed"

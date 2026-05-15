from __future__ import annotations

import sys
import types

import pytest

from rag.storage.search_backends.milvus_vector_repo import MilvusVectorRepo


def test_milvus_vector_repo_queries_numeric_section_entries(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(MilvusVectorRepo, "_connect", lambda self: setattr(self, "_connected", True))

    class _Connections:
        def disconnect(self, alias: str) -> None:
            return None

    monkeypatch.setitem(sys.modules, "pymilvus", types.SimpleNamespace(connections=_Connections()))

    class _FakeCollection:
        name = "knowledge_index__section_summary__default"

        def query(self, *, expr: str, output_fields: list[str]) -> list[dict[str, object]]:
            assert expr == "item_id in [7]"
            assert "source_id" in output_fields
            assert "version_group_id" in output_fields
            return [
                {
                    "item_id": 7,
                    "doc_id": 42,
                    "source_id": 9,
                    "section_id": 7,
                    "version_group_id": 42,
                    "version_no": 3,
                    "doc_status": "published",
                    "is_active": True,
                    "embedding_model_id": "bge-m3",
                    "summary_text": "Alpha section",
                    "metadata_json": "{\"section_id\": 7, \"source_id\": 9}",
                    "embedding": [0.1, 0.2],
                }
            ]

        def release(self) -> None:
            return None

    repo = MilvusVectorRepo("http://127.0.0.1:19530")
    fake_collection = _FakeCollection()
    monkeypatch.setattr(repo, "_has_collection", lambda **kwargs: True)
    monkeypatch.setattr(repo, "_collection", lambda **kwargs: fake_collection)
    repo._collections[fake_collection.name] = fake_collection
    try:
        entry = repo.get_entry("7", item_kind="section_summary")
        assert entry is not None
        assert entry.doc_id == 42
        assert entry.metadata["section_id"] == "7"
        assert entry.text == "Alpha section"
        assert entry.metadata["source_id"] == "9"
    finally:
        repo.close()


def test_milvus_vector_repo_deletes_numeric_documents(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(MilvusVectorRepo, "_connect", lambda self: setattr(self, "_connected", True))

    class _Connections:
        def disconnect(self, alias: str) -> None:
            return None

    monkeypatch.setitem(sys.modules, "pymilvus", types.SimpleNamespace(connections=_Connections()))

    class _FakeCollection:
        def __init__(self) -> None:
            self.expr: str | None = None
            self.flush_count = 0

        def delete(self, expr: str):
            self.expr = expr
            return types.SimpleNamespace(delete_count=2)

        def flush(self) -> None:
            self.flush_count += 1

        def release(self) -> None:
            return None

    repo = MilvusVectorRepo("http://127.0.0.1:19530")
    fake_collection = _FakeCollection()
    monkeypatch.setattr(repo, "_iter_target_collections", lambda **kwargs: [fake_collection])
    try:
        deleted = repo.delete_for_documents(["42"], item_kind="section_summary")
        assert deleted == 2
        assert fake_collection.expr == "doc_id in [42]"
        assert fake_collection.flush_count == 1
    finally:
        repo.close()

from __future__ import annotations

import asyncio
import sys
import types
from datetime import UTC, datetime

import pytest

from rag.schema.core import PartitionKey, SectionSummaryRecord, SourceType
from rag.storage.search_backends.milvus_vector_repo import MilvusVectorRepo


def test_milvus_vector_repo_creates_missing_database(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[tuple[str, str, str | None]] = []

    class _Connections:
        def connect(self, *, alias: str, uri: str, token: str = "", db_name: str = "default", **kwargs) -> None:
            events.append(("connect", alias, db_name))

        def disconnect(self, alias: str) -> None:
            events.append(("disconnect", alias, None))

    class _Db:
        def list_database(self, *, using: str = "default", timeout=None) -> list[str]:
            events.append(("list_database", using, None))
            return ["default"]

        def create_database(self, db_name: str, *, using: str = "default", timeout=None, **kwargs) -> None:
            events.append(("create_database", using, db_name))

    fake_pymilvus = types.SimpleNamespace(connections=_Connections(), db=_Db())
    monkeypatch.setitem(sys.modules, "pymilvus", fake_pymilvus)

    repo = MilvusVectorRepo("http://127.0.0.1:19530", db_name="rag_v1", collection_prefix="summary_index")
    try:
        assert ("list_database", f"{repo._alias}_bootstrap", None) in events
        assert ("create_database", f"{repo._alias}_bootstrap", "rag_v1") in events
        assert ("connect", repo._alias, "rag_v1") in events
    finally:
        repo.close()


def test_milvus_vector_repo_upserts_v1_section_summary_with_partition(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(MilvusVectorRepo, "_connect", lambda self: setattr(self, "_connected", True))

    class _Connections:
        def disconnect(self, alias: str) -> None:
            return None

    monkeypatch.setitem(sys.modules, "pymilvus", types.SimpleNamespace(connections=_Connections()))

    class _FakeCollection:
        name = "summary_index__section_summary__default"

        def __init__(self) -> None:
            self.upsert_calls: list[tuple[list[dict[str, object]], str | None]] = []
            self.flush_count = 0

        def upsert(self, rows: list[dict[str, object]], partition_name: str | None = None) -> None:
            self.upsert_calls.append(([dict(row) for row in rows], partition_name))

        def flush(self) -> None:
            self.flush_count += 1

        def release(self) -> None:
            return None

    repo = MilvusVectorRepo("http://127.0.0.1:19530")
    fake_collection = _FakeCollection()
    monkeypatch.setattr(repo, "_collection", lambda **kwargs: fake_collection)
    monkeypatch.setattr(repo, "_UPSERT_BUFFER_SIZE", 1)
    repo._collections[fake_collection.name] = fake_collection
    record = SectionSummaryRecord(
        section_id=7,
        doc_id=42,
        source_id=9,
        version_group_id=42,
        version_no=3,
        doc_status="published",
        effective_date=datetime(2026, 4, 18, tzinfo=UTC),
        updated_at=datetime(2026, 4, 18, 12, 0, tzinfo=UTC),
        is_active=False,
        tenant_id="tenant-a",
        department_id="dept-a",
        auth_tag="internal",
        source_type=SourceType.MARKDOWN,
        embedding_model_id="bge-m3",
        partition_key=PartitionKey.COLD,
        page_start=1,
        page_end=2,
        section_kind="body",
        toc_path=["A", "B"],
        summary_text="Section summary",
        metadata_json={"rank": 1},
    )
    try:
        repo.upsert_record(record, [0.1, 0.2])
        repo._flush_dirty_collections()
        rows, partition_name = fake_collection.upsert_calls[0]
        assert partition_name == "cold"
        assert rows[0]["item_id"] == 7
        assert rows[0]["doc_id"] == 42
        assert rows[0]["source_id"] == 9
        assert rows[0]["version_group_id"] == 42
        assert rows[0]["section_id"] == 7
        assert rows[0]["toc_path_text"] == "A / B"
        assert rows[0]["embedding_model_id"] == "bge-m3"
        assert rows[0]["index_ready"] is True
        assert rows[0]["external_sparse_embedding"] == {}
        assert fake_collection.flush_count == 1
    finally:
        repo.close()


def test_milvus_vector_repo_sparse_schema_uses_non_nullable_external_field(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(MilvusVectorRepo, "_connect", lambda self: setattr(self, "_connected", True))

    class _Connections:
        def disconnect(self, alias: str) -> None:
            return None

    class _DataType:
        INT64 = "INT64"
        INT32 = "INT32"
        VARCHAR = "VARCHAR"
        BOOL = "BOOL"
        FLOAT_VECTOR = "FLOAT_VECTOR"
        SPARSE_FLOAT_VECTOR = "SPARSE_FLOAT_VECTOR"

    class _FieldSchema:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

    monkeypatch.setitem(sys.modules, "pymilvus", types.SimpleNamespace(connections=_Connections()))

    repo = MilvusVectorRepo("http://127.0.0.1:19530")
    try:
        fields = repo._collection_fields(
            item_kind="section_summary",
            dimension=2,
            data_type=_DataType,
            field_schema=_FieldSchema,
        )
    finally:
        repo.close()

    external_sparse = next(field for field in fields if field.kwargs["name"] == "external_sparse_embedding")
    assert external_sparse.kwargs["dtype"] == _DataType.SPARSE_FLOAT_VECTOR
    assert "nullable" not in external_sparse.kwargs


def test_milvus_vector_repo_deduplicates_primary_keys_within_one_flush(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(MilvusVectorRepo, "_connect", lambda self: setattr(self, "_connected", True))

    class _Connections:
        def disconnect(self, alias: str) -> None:
            return None

    monkeypatch.setitem(sys.modules, "pymilvus", types.SimpleNamespace(connections=_Connections()))

    class _FakeCollection:
        name = "summary_index__section_summary__default"

        def __init__(self) -> None:
            self.upsert_calls: list[tuple[list[dict[str, object]], str | None]] = []

        def upsert(self, rows: list[dict[str, object]], partition_name: str | None = None) -> None:
            item_ids = [row["item_id"] for row in rows]
            assert len(item_ids) == len(set(item_ids))
            self.upsert_calls.append(([dict(row) for row in rows], partition_name))

        def flush(self) -> None:
            return None

        def release(self) -> None:
            return None

    repo = MilvusVectorRepo("http://127.0.0.1:19530")
    fake_collection = _FakeCollection()
    monkeypatch.setattr(repo, "_collection", lambda **kwargs: fake_collection)
    repo._collections[fake_collection.name] = fake_collection

    record = SectionSummaryRecord(
        section_id=7,
        doc_id=42,
        source_id=9,
        version_group_id=42,
        version_no=3,
        doc_status="published",
        effective_date=datetime(2026, 4, 18, tzinfo=UTC),
        updated_at=datetime(2026, 4, 18, 12, 0, tzinfo=UTC),
        is_active=True,
        source_type=SourceType.MARKDOWN,
        embedding_model_id="bge-m3",
        partition_key=PartitionKey.HOT,
        page_start=1,
        page_end=2,
        section_kind="body",
        toc_path=["A", "B"],
        summary_text="old summary",
        metadata_json={},
    )
    try:
        repo.upsert_records(
            [
                (record, [0.1, 0.2]),
                (record.model_copy(update={"summary_text": "new summary"}), [0.3, 0.4]),
            ]
        )
        assert len(fake_collection.upsert_calls) == 1
        rows, partition_name = fake_collection.upsert_calls[0]
        assert partition_name == "hot"
        assert len(rows) == 1
        assert rows[0]["item_id"] == 7
        assert rows[0]["summary_text"] == "new summary"
    finally:
        repo.close()


def test_milvus_vector_repo_search_injects_system_guardrail(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(MilvusVectorRepo, "_connect", lambda self: setattr(self, "_connected", True))

    class _Connections:
        def disconnect(self, alias: str) -> None:
            return None

    monkeypatch.setitem(sys.modules, "pymilvus", types.SimpleNamespace(connections=_Connections()))

    class _FakeCollection:
        name = "summary_index__section_summary__default"

        def search(self, **kwargs):
            assert (
                kwargs["expr"]
                == '(is_active == true and index_ready == true) and (doc_id in [42]) and (tenant_id == "tenant-a")'
            )
            return [[]]

        def release(self) -> None:
            return None

    repo = MilvusVectorRepo("http://127.0.0.1:19530")
    fake_collection = _FakeCollection()
    monkeypatch.setattr(repo, "_has_collection", lambda **kwargs: True)
    monkeypatch.setattr(repo, "_collection", lambda **kwargs: fake_collection)
    repo._collections[fake_collection.name] = fake_collection
    try:
        results = repo.search([0.1, 0.2], doc_ids=["42"], expr='tenant_id == "tenant-a"')
        assert results == []
    finally:
        repo.close()


def test_milvus_vector_repo_hybrid_search_passes_expr_to_every_ann_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(MilvusVectorRepo, "_connect", lambda self: setattr(self, "_connected", True))

    class _Connections:
        def disconnect(self, alias: str) -> None:
            return None

    class _AnnSearchRequest:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

    class _RRFRanker:
        pass

    monkeypatch.setitem(
        sys.modules,
        "pymilvus",
        types.SimpleNamespace(
            connections=_Connections(),
            AnnSearchRequest=_AnnSearchRequest,
            RRFRanker=_RRFRanker,
        ),
    )

    class _FakeCollection:
        name = "summary_index__section_summary__default"

        def release(self) -> None:
            return None

    class _FakeAsyncClient:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        async def hybrid_search(self, **kwargs):
            self.calls.append(kwargs)
            reqs = kwargs["reqs"]
            assert [request.kwargs["expr"] for request in reqs] == [
                '(is_active == true and index_ready == true) and (doc_id in [42]) and (tenant_id == "tenant-a")',
                '(is_active == true and index_ready == true) and (doc_id in [42]) and (tenant_id == "tenant-a")',
            ]
            if kwargs["partition_names"] == ["hot"]:
                return []
            return [
                [
                    {
                        "id": 7,
                        "distance": 0.91,
                        "entity": {
                            "item_id": 7,
                            "doc_id": 42,
                            "source_id": 9,
                            "section_id": 7,
                            "summary_text": "Section summary",
                            "metadata_json": "{}",
                        },
                    }
                ]
            ]

    repo = MilvusVectorRepo("http://127.0.0.1:19530")
    fake_collection = _FakeCollection()
    fake_async_client = _FakeAsyncClient()
    monkeypatch.setattr(repo, "_has_collection", lambda **kwargs: True)
    monkeypatch.setattr(repo, "_collection", lambda **kwargs: fake_collection)
    monkeypatch.setattr(repo, "_supports_bm25_schema", lambda collection: True)
    monkeypatch.setattr(repo, "_get_async_client", lambda: fake_async_client)
    repo._collections[fake_collection.name] = fake_collection
    try:
        results = asyncio.run(
            repo.hybrid_search_async(
                query_vector=[0.1, 0.2],
                sparse_query="alpha policy",
                doc_ids=["42"],
                expr='tenant_id == "tenant-a"',
            )
        )
        assert [result.item_id for result in results] == ["7"]
        assert len(fake_async_client.calls) == 2
    finally:
        repo.close()


def test_milvus_vector_repo_does_not_reuse_async_client_across_event_loops(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(MilvusVectorRepo, "_connect", lambda self: setattr(self, "_connected", True))

    class _Connections:
        def disconnect(self, alias: str) -> None:
            return None

    class _AnnSearchRequest:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

    class _RRFRanker:
        pass

    clients: list[_FakeAsyncClient] = []

    class _FakeAsyncClient:
        def __init__(self, **kwargs) -> None:
            del kwargs
            self.loop = asyncio.get_running_loop()
            self.closed = False
            clients.append(self)

        async def hybrid_search(self, **kwargs):
            if asyncio.get_running_loop() is not self.loop:
                raise RuntimeError("async client reused across event loops")
            if self.closed:
                raise RuntimeError("async client reused after close")
            return [
                [
                    {
                        "id": 7,
                        "distance": 0.91,
                        "entity": {
                            "item_id": 7,
                            "doc_id": 42,
                            "source_id": 9,
                            "section_id": 7,
                            "summary_text": "Section summary",
                            "metadata_json": "{}",
                        },
                    }
                ]
            ]

        async def close(self) -> None:
            self.closed = True

    monkeypatch.setitem(
        sys.modules,
        "pymilvus",
        types.SimpleNamespace(
            connections=_Connections(),
            AnnSearchRequest=_AnnSearchRequest,
            RRFRanker=_RRFRanker,
            AsyncMilvusClient=_FakeAsyncClient,
        ),
    )

    class _FakeCollection:
        name = "summary_index__section_summary__default"

        def release(self) -> None:
            return None

    repo = MilvusVectorRepo("http://127.0.0.1:19530")
    fake_collection = _FakeCollection()
    monkeypatch.setattr(repo, "_has_collection", lambda **kwargs: True)
    monkeypatch.setattr(repo, "_collection", lambda **kwargs: fake_collection)
    monkeypatch.setattr(repo, "_supports_bm25_schema", lambda collection: True)
    repo._collections[fake_collection.name] = fake_collection
    try:
        for _ in range(2):
            results = asyncio.run(
                repo.hybrid_search_async(
                    query_vector=[0.1, 0.2],
                    sparse_query="alpha policy",
                    doc_ids=["42"],
                )
            )
            assert [result.item_id for result in results] == ["7"]
        assert len(clients) == 2
        assert all(client.closed for client in clients)
    finally:
        repo.close()


def test_milvus_vector_repo_weighted_hybrid_ranker_uses_alpha_when_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(MilvusVectorRepo, "_connect", lambda self: setattr(self, "_connected", True))

    class _Connections:
        def disconnect(self, alias: str) -> None:
            return None

    class _AnnSearchRequest:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

    class _RRFRanker:
        pass

    class _WeightedRanker:
        def __init__(self, dense_weight: float, sparse_weight: float) -> None:
            self.weights = (dense_weight, sparse_weight)

    monkeypatch.setitem(
        sys.modules,
        "pymilvus",
        types.SimpleNamespace(
            connections=_Connections(),
            AnnSearchRequest=_AnnSearchRequest,
            RRFRanker=_RRFRanker,
            WeightedRanker=_WeightedRanker,
        ),
    )

    class _FakeCollection:
        name = "summary_index__section_summary__default"

        def release(self) -> None:
            return None

    class _FakeAsyncClient:
        async def hybrid_search(self, **kwargs):
            ranker = kwargs["ranker"]
            assert isinstance(ranker, _WeightedRanker)
            assert ranker.weights == (0.7, 0.3)
            return []

    repo = MilvusVectorRepo("http://127.0.0.1:19530")
    fake_collection = _FakeCollection()
    monkeypatch.setattr(repo, "_has_collection", lambda **kwargs: True)
    monkeypatch.setattr(repo, "_collection", lambda **kwargs: fake_collection)
    monkeypatch.setattr(repo, "_supports_bm25_schema", lambda collection: True)
    monkeypatch.setattr(repo, "_get_async_client", lambda: _FakeAsyncClient())
    repo._collections[fake_collection.name] = fake_collection
    try:
        asyncio.run(
            repo.hybrid_search_async(
                query_vector=[0.1, 0.2],
                sparse_query="alpha policy",
                doc_ids=["42"],
                fusion_strategy="weighted_rrf",
                alpha=0.7,
            )
        )
    finally:
        repo.close()


def test_milvus_vector_repo_accepts_sparse_query_vector_for_dual_mode_sparse_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(MilvusVectorRepo, "_connect", lambda self: setattr(self, "_connected", True))

    class _Connections:
        def disconnect(self, alias: str) -> None:
            return None

    class _AnnSearchRequest:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

    class _RRFRanker:
        pass

    monkeypatch.setitem(
        sys.modules,
        "pymilvus",
        types.SimpleNamespace(
            connections=_Connections(),
            AnnSearchRequest=_AnnSearchRequest,
            RRFRanker=_RRFRanker,
        ),
    )

    class _FakeCollection:
        name = "summary_index__section_summary__default"

        def release(self) -> None:
            return None

    class _FakeAsyncClient:
        async def hybrid_search(self, **kwargs):
            # 第一版：即使传了 sparse_query_vector，也只创建 dense + BM25 请求
            assert len(kwargs["reqs"]) == 2  # dense + bm25，无 external sparse
            bm25_request = kwargs["reqs"][1]
            assert bm25_request.kwargs["anns_field"] == "bm25_sparse_embedding"
            assert bm25_request.kwargs["data"] == ["alpha policy"]
            assert bm25_request.kwargs["param"] == {"metric_type": "BM25", "params": {}}
            return []

    repo = MilvusVectorRepo("http://127.0.0.1:19530")
    fake_collection = _FakeCollection()
    monkeypatch.setattr(repo, "_has_collection", lambda **kwargs: True)
    monkeypatch.setattr(repo, "_collection", lambda **kwargs: fake_collection)
    monkeypatch.setattr(repo, "_supports_bm25_schema", lambda collection: True)
    monkeypatch.setattr(repo, "_get_async_client", lambda: _FakeAsyncClient())
    repo._collections[fake_collection.name] = fake_collection
    try:
        asyncio.run(
            repo.hybrid_search_async(
                query_vector=[0.1, 0.2],
                sparse_query="alpha policy",
                sparse_query_vector={1: 0.4, 7: 0.9},
                doc_ids=["42"],
            )
        )
    finally:
        repo.close()

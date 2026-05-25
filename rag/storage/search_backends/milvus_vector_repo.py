from __future__ import annotations

import inspect
import json
import re
from collections import defaultdict
from collections.abc import Iterable, Sequence
from datetime import datetime
from typing import Any, cast

from rag.schema.core import AssetSummaryRecord, DocSummaryRecord, SectionSummaryRecord
from rag.schema.runtime import StoredVectorEntry, VectorSearchResult

SummaryRecord = DocSummaryRecord | SectionSummaryRecord | AssetSummaryRecord


def _to_int(value: object, default: int = 0) -> int:
    """Safely coerce a metadata value to int (handles str from JSON round-tripping)."""
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return default


class MilvusVectorRepo:
    _UPSERT_BUFFER_SIZE = 256
    _SUPPORTED_KINDS = ("doc_summary", "section_summary", "asset_summary")
    _PARTITIONS = ("hot", "cold")

    def __init__(
        self,
        uri: str,
        *,
        token: str | None = None,
        db_name: str | None = None,
        collection_prefix: str = "knowledge_index",
        consistency_level: str = "Bounded",
    ) -> None:
        self._uri = uri
        self._token = token
        self._db_name = db_name
        self._collection_prefix = collection_prefix
        self._consistency_level = consistency_level
        self._alias = f"rag_milvus_{abs(hash((uri, collection_prefix))) % 1000000}"
        self._connected = False
        self._collections: dict[str, Any] = {}
        self._dirty_collections: set[str] = set()
        self._pending_upserts: dict[str, list[tuple[str, dict[str, Any]]]] = defaultdict(list)
        # AsyncMilvusClient is bound to the event loop that creates it. The
        # CLI benchmark path uses a sync bridge with short-lived event loops,
        # so hybrid search must create clients per call instead of caching one.
        self._connect()

    def upsert(
        self,
        item_id: str,
        vector: Iterable[float],
        *,
        metadata: dict[str, Any] | None = None,
        embedding_space: str = "default",
        item_kind: str = "section_summary",
    ) -> None:
        self._validate_item_kind(item_kind)
        row, partition_name = self._build_row(
            item_id=item_id,
            vector=vector,
            metadata=dict(metadata or {}),
            item_kind=item_kind,
        )
        collection = self._collection(
            item_kind=item_kind,
            embedding_space=embedding_space,
            dimension=len(cast(list[float], row["embedding"])),
        )
        self._pending_upserts[collection.name].append((partition_name, row))
        if len(self._pending_upserts[collection.name]) >= self._UPSERT_BUFFER_SIZE:
            self._drain_pending_collection(collection.name)

    def upsert_record(
        self,
        record: SummaryRecord,
        vector: Iterable[float],
        *,
        embedding_space: str = "default",
    ) -> None:
        item_kind = self._item_kind_for_record(record)
        metadata: dict[str, Any] = record.model_dump(mode="python")
        item_id = str(self._record_item_id(record))
        self.upsert(item_id, vector, metadata=metadata, embedding_space=embedding_space, item_kind=item_kind)

    def upsert_records(
        self,
        items: Sequence[tuple[SummaryRecord, Iterable[float]]],
        *,
        embedding_space: str = "default",
    ) -> None:
        for record, vector in items:
            self.upsert_record(record, vector, embedding_space=embedding_space)
        self._flush_dirty_collections()

    def search(
        self,
        query: Iterable[float],
        *,
        limit: int = 10,
        doc_ids: list[str] | None = None,
        expr: str | None = None,
        embedding_space: str = "default",
        item_kind: str = "section_summary",
    ) -> list[VectorSearchResult]:
        self._validate_item_kind(item_kind)
        query_vector = [float(value) for value in query]
        if not query_vector:
            return []
        self._flush_dirty_collections()
        if not self._has_collection(item_kind=item_kind, embedding_space=embedding_space):
            return []
        collection = self._collection(item_kind=item_kind, embedding_space=embedding_space)
        self._validate_query_dimension(collection, query_vector, embedding_space=embedding_space, item_kind=item_kind)
        final_expr = self._search_expr(doc_ids=doc_ids, user_expr=expr)
        output_fields = self._output_fields(item_kind=item_kind, include_embedding=False)
        hits = self._search_collection(
            collection,
            query_vector,
            expr=final_expr,
            limit=limit,
            output_fields=output_fields,
            partitions=["hot"],
        )
        if not hits:
            hits = self._search_collection(
                collection,
                query_vector,
                expr=final_expr,
                limit=limit,
                output_fields=output_fields,
                partitions=None,
            )
        results = [self._vector_result_from_hit(hit, item_kind=item_kind) for hit in hits]
        results.sort(key=lambda item: (-item.score, item.item_id))
        return results[:limit]

    def get_entry(
        self,
        item_id: str,
        *,
        embedding_space: str = "default",
        item_kind: str = "section_summary",
    ) -> StoredVectorEntry | None:
        self._validate_item_kind(item_kind)
        self._flush_dirty_collections()
        if not self._has_collection(item_kind=item_kind, embedding_space=embedding_space):
            return None
        collection = self._collection(item_kind=item_kind, embedding_space=embedding_space)
        rows = self._query_collection(
            collection,
            expr=self._item_id_expr(item_id),
            output_fields=self._output_fields(item_kind=item_kind, include_embedding=True),
        )
        if not rows:
            return None
        row = rows[0]
        metadata = self._load_metadata(row.get("metadata_json", "{}"))
        if (section_id := row.get("section_id")) is not None:
            metadata.setdefault("section_id", str(section_id))
        if (asset_id := row.get("asset_id")) is not None:
            metadata.setdefault("asset_id", str(asset_id))
        return StoredVectorEntry(
            item_id=str(row["item_id"]),
            item_kind=item_kind,
            embedding_space=embedding_space,
            doc_id=_to_int(row["doc_id"]),
            text=self._entry_text(row, item_kind=item_kind),
            metadata=metadata,
            vector=[float(value) for value in cast(list[float], row.get("embedding", []))],
        )

    def existing_item_ids(
        self,
        item_ids: Sequence[str],
        *,
        embedding_space: str | None = None,
        item_kind: str | None = "section_summary",
    ) -> set[str]:
        normalized = tuple(dict.fromkeys(item_ids))
        if not normalized:
            return set()
        self._flush_dirty_collections()
        rows_found: set[str] = set()
        for collection in self._iter_target_collections(item_kind=item_kind, embedding_space=embedding_space):
            rows = self._query_collection(
                collection,
                expr=self._item_ids_expr(normalized),
                output_fields=["item_id"],
            )
            rows_found.update(str(row["item_id"]) for row in rows)
        return rows_found

    def count_vectors(
        self,
        *,
        embedding_space: str | None = None,
        item_kind: str | None = None,
        distinct_records: bool = False,
    ) -> int:
        del distinct_records
        self._flush_dirty_collections()
        total = 0
        for collection in self._iter_target_collections(item_kind=item_kind, embedding_space=embedding_space):
            rows = self._query_collection(collection, expr="item_id >= 0", output_fields=["item_id"])
            total += len(rows)
        return total

    def delete_for_documents(
        self,
        doc_ids: Sequence[str],
        *,
        item_kind: str | None = None,
    ) -> int:
        normalized = tuple(dict.fromkeys(doc_ids))
        if not normalized:
            return 0
        self._flush_dirty_collections()
        deleted = 0
        for collection in self._iter_target_collections(item_kind=item_kind, embedding_space=None):
            result = cast(Any, collection).delete(self._doc_ids_expr(normalized))
            cast(Any, collection).flush()
            deleted += int(getattr(result, "delete_count", 0))
        return deleted

    def close(self) -> None:
        if not self._connected:
            return
        from pymilvus import connections

        self._flush_dirty_collections()
        self._close_async_client()
        for collection in self._collections.values():
            release = getattr(collection, "release", None)
            if callable(release):
                release()
        connections.disconnect(self._alias)
        self._collections.clear()
        self._connected = False

    def _connect(self) -> None:
        if self._connected:
            return
        from pymilvus import connections

        kwargs: dict[str, object] = {"alias": self._alias, "uri": self._uri}
        if self._token:
            kwargs["token"] = self._token
        if self._db_name:
            self._ensure_database_exists()
            kwargs["db_name"] = self._db_name
        connections.connect(**kwargs)
        self._connected = True

    def _ensure_database_exists(self) -> None:
        from pymilvus import connections, db

        if not self._db_name or self._db_name == "default":
            return
        bootstrap_alias = f"{self._alias}_bootstrap"
        kwargs: dict[str, object] = {"alias": bootstrap_alias, "uri": self._uri}
        if self._token:
            kwargs["token"] = self._token
        connections.connect(**kwargs)
        try:
            existing = set(cast(list[str], db.list_database(using=bootstrap_alias)))
            if self._db_name not in existing:
                db.create_database(self._db_name, using=bootstrap_alias)
        finally:
            connections.disconnect(bootstrap_alias)

    def _collection(self, *, item_kind: str, embedding_space: str, dimension: int | None = None) -> Any:
        self._validate_item_kind(item_kind)
        from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, Function, FunctionType, utility

        name = self._collection_name(item_kind=item_kind, embedding_space=embedding_space)
        cached = self._collections.get(name)
        if cached is not None:
            return cached
        if not utility.has_collection(name, using=self._alias):
            if dimension is None:
                raise RuntimeError(f"Milvus collection {name} does not exist and no vector dimension was provided")
            schema = CollectionSchema(
                fields=self._collection_fields(
                    item_kind=item_kind,
                    dimension=dimension,
                    data_type=DataType,
                    field_schema=FieldSchema,
                ),
                functions=self._collection_functions(function=Function, function_type=FunctionType),
                description=f"Summary index for {item_kind}",
                enable_dynamic_field=False,
            )
            collection = Collection(name=name, schema=schema, using=self._alias)
            self._ensure_partitions(collection)
            self._create_indexes(collection, item_kind=item_kind)
        else:
            collection = Collection(name=name, using=self._alias)
            self._ensure_partitions(collection)
        collection.load()
        self._collections[name] = collection
        return collection

    def _collection_fields(self, *, item_kind: str, dimension: int, data_type: Any, field_schema: Any) -> list[Any]:
        common = [
            field_schema(name="item_id", dtype=data_type.INT64, is_primary=True),
            field_schema(name="doc_id", dtype=data_type.INT64),
            field_schema(name="source_id", dtype=data_type.INT64),
            field_schema(name="version_group_id", dtype=data_type.INT64),
            field_schema(name="version_no", dtype=data_type.INT32),
            field_schema(name="doc_status", dtype=data_type.VARCHAR, max_length=32),
            field_schema(name="effective_date", dtype=data_type.INT64),
            field_schema(name="updated_at", dtype=data_type.INT64),
            field_schema(name="is_active", dtype=data_type.BOOL),
            field_schema(name="index_ready", dtype=data_type.BOOL),
            field_schema(name="tenant_id", dtype=data_type.VARCHAR, max_length=64),
            field_schema(name="department_id", dtype=data_type.VARCHAR, max_length=64),
            field_schema(name="auth_tag", dtype=data_type.VARCHAR, max_length=128),
            field_schema(name="embedding_model_id", dtype=data_type.VARCHAR, max_length=64),
            field_schema(name="partition_key", dtype=data_type.VARCHAR, max_length=16),
            field_schema(
                name="bm25_text",
                dtype=data_type.VARCHAR,
                max_length=65535,
                enable_match=True,
                enable_analyzer=True,
            ),
            field_schema(name="metadata_json", dtype=data_type.VARCHAR, max_length=65535),
            field_schema(name="embedding", dtype=data_type.FLOAT_VECTOR, dim=dimension),
            field_schema(name="bm25_sparse_embedding", dtype=data_type.SPARSE_FLOAT_VECTOR),
            field_schema(name="external_sparse_embedding", dtype=data_type.SPARSE_FLOAT_VECTOR),
        ]
        if item_kind == "doc_summary":
            return common + [
                field_schema(name="title", dtype=data_type.VARCHAR, max_length=4096),
                field_schema(name="source_type", dtype=data_type.VARCHAR, max_length=32),
                field_schema(name="summary_text", dtype=data_type.VARCHAR, max_length=65535),
            ]
        if item_kind == "section_summary":
            return common + [
                field_schema(name="section_id", dtype=data_type.INT64),
                field_schema(name="page_start", dtype=data_type.INT32),
                field_schema(name="page_end", dtype=data_type.INT32),
                field_schema(name="section_kind", dtype=data_type.VARCHAR, max_length=64),
                field_schema(name="toc_path_text", dtype=data_type.VARCHAR, max_length=4096),
                field_schema(name="source_type", dtype=data_type.VARCHAR, max_length=32),
                field_schema(name="summary_text", dtype=data_type.VARCHAR, max_length=65535),
            ]
        return common + [
            field_schema(name="asset_id", dtype=data_type.INT64),
            field_schema(name="section_id", dtype=data_type.INT64),
            field_schema(name="asset_type", dtype=data_type.VARCHAR, max_length=64),
            field_schema(name="page_no", dtype=data_type.INT32),
            field_schema(name="caption", dtype=data_type.VARCHAR, max_length=65535),
            field_schema(name="summary_text", dtype=data_type.VARCHAR, max_length=65535),
        ]

    @staticmethod
    def _collection_functions(*, function: Any, function_type: Any) -> list[Any]:
        return [
            function(
                name="bm25_summary_text",
                function_type=function_type.BM25,
                input_field_names=["bm25_text"],
                output_field_names=["bm25_sparse_embedding"],
            )
        ]

    def _ensure_partitions(self, collection: Any) -> None:
        for partition_name in self._PARTITIONS:
            try:
                has_partition = getattr(collection, "has_partition", None)
                if callable(has_partition) and has_partition(partition_name):
                    continue
                cast(Any, collection).create_partition(partition_name)
            except Exception:
                continue

    def _create_indexes(self, collection: Any, *, item_kind: str) -> None:
        if item_kind == "asset_summary":
            vector_index = {"index_type": "HNSW", "metric_type": "COSINE", "params": {"M": 16, "efConstruction": 200}}
        else:
            vector_index = {"index_type": "IVF_SQ8", "metric_type": "COSINE", "params": {"nlist": 1024}}
        self._safe_create_index(collection, "embedding", vector_index)
        self._safe_create_index(collection, "is_active", {"index_type": "BITMAP"})
        self._safe_create_index(collection, "index_ready", {"index_type": "BITMAP"})
        self._safe_create_index(collection, "tenant_id", {"index_type": "INVERTED"})
        self._safe_create_index(collection, "department_id", {"index_type": "INVERTED"})
        self._safe_create_index(collection, "auth_tag", {"index_type": "INVERTED"})
        self._safe_create_index(collection, "embedding_model_id", {"index_type": "INVERTED"})
        self._safe_create_index(collection, "partition_key", {"index_type": "INVERTED"})
        self._safe_create_index(
            collection,
            "bm25_sparse_embedding",
            {"index_type": "SPARSE_INVERTED_INDEX", "metric_type": "BM25", "params": {}},
        )
        self._safe_create_index(
            collection,
            "external_sparse_embedding",
            {"index_type": "SPARSE_INVERTED_INDEX", "metric_type": "IP", "params": {}},
        )

    def _safe_create_index(self, collection: Any, field_name: str, index_params: dict[str, Any]) -> None:
        try:
            cast(Any, collection).create_index(field_name=field_name, index_params=index_params)
        except Exception:
            return

    def _has_collection(self, *, item_kind: str, embedding_space: str) -> bool:
        self._validate_item_kind(item_kind)
        from pymilvus import utility

        return bool(
            utility.has_collection(
                self._collection_name(item_kind=item_kind, embedding_space=embedding_space),
                using=self._alias,
            )
        )

    def _flush_dirty_collections(self) -> None:
        for name in list(self._pending_upserts):
            self._drain_pending_collection(name)
        for name in list(self._dirty_collections):
            collection = self._collections.get(name)
            if collection is None:
                continue
            cast(Any, collection).flush()
            self._dirty_collections.discard(name)

    def _drain_pending_collection(self, name: str) -> None:
        buffered = self._pending_upserts.get(name)
        if not buffered:
            return
        collection = self._collections.get(name)
        if collection is None:
            return
        latest_by_item_id: dict[int, tuple[str, dict[str, Any]]] = {}
        for partition_name, row in buffered:
            latest_by_item_id[int(row["item_id"])] = (partition_name, row)
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for partition_name, row in latest_by_item_id.values():
            grouped[partition_name].append(row)
        for partition_name, rows in grouped.items():
            self._upsert_rows(collection, rows, partition_name=partition_name)
        self._dirty_collections.add(name)
        buffered.clear()

    def _upsert_rows(self, collection: Any, rows: list[dict[str, Any]], *, partition_name: str) -> None:
        try:
            cast(Any, collection).upsert(rows, partition_name=partition_name)
        except TypeError:
            cast(Any, collection).upsert(rows)

    def _iter_target_collections(self, *, item_kind: str | None, embedding_space: str | None) -> list[Any]:
        from pymilvus import utility

        names = list(cast(list[str], utility.list_collections(using=self._alias)))
        targets: list[Any] = []
        prefix = f"{self._sanitize(self._collection_prefix)}__"
        for name in names:
            if not name.startswith(prefix):
                continue
            suffix = name[len(prefix) :]
            parts = suffix.split("__", 1)
            if len(parts) != 2:
                continue
            current_kind, current_space = parts
            if item_kind is not None and current_kind != self._sanitize(item_kind):
                continue
            if embedding_space is not None and current_space != self._sanitize(embedding_space):
                continue
            targets.append(self._collection(item_kind=current_kind, embedding_space=current_space))
        return targets

    def _collection_name(self, *, item_kind: str, embedding_space: str) -> str:
        return (
            f"{self._sanitize(self._collection_prefix)}__"
            f"{self._sanitize(item_kind)}__"
            f"{self._sanitize(embedding_space)}"
        )

    @staticmethod
    def _sanitize(value: str) -> str:
        cleaned = re.sub(r"[^A-Za-z0-9_]", "_", value)
        return cleaned[:200] or "default"

    def _build_row(
        self,
        *,
        item_id: str,
        vector: Iterable[float],
        metadata: dict[str, Any],
        item_kind: str,
    ) -> tuple[dict[str, Any], str]:
        record = self._normalize_metadata(metadata)
        partition_name = self._partition_name(record)
        row: dict[str, Any] = {
            "item_id": int(item_id),
            "doc_id": int(record.get("doc_id", item_id)),
            "source_id": int(record.get("source_id", 0)),
            "version_group_id": int(record.get("version_group_id", record.get("doc_id", item_id))),
            "version_no": int(record.get("version_no", 1)),
            "doc_status": self._text(record.get("doc_status", "published"), limit=32),
            "effective_date": self._timestamp_ms(record.get("effective_date")),
            "updated_at": self._timestamp_ms(record.get("updated_at")),
            "is_active": bool(record.get("is_active", True)),
            "index_ready": bool(record.get("index_ready", True)),
            "tenant_id": self._text(record.get("tenant_id"), limit=64),
            "department_id": self._text(record.get("department_id"), limit=64),
            "auth_tag": self._text(record.get("auth_tag"), limit=128),
            "embedding_model_id": self._text(record.get("embedding_model_id", "default"), limit=64),
            "partition_key": partition_name,
            "bm25_text": self._text(self._bm25_text(record, item_kind=item_kind), limit=65535),
            "metadata_json": json.dumps(self._serialize_metadata(record), ensure_ascii=True, sort_keys=True),
            "embedding": [float(value) for value in vector],
            "external_sparse_embedding": {},
        }
        if item_kind == "doc_summary":
            row.update(
                {
                    "title": self._text(record.get("title"), limit=4096),
                    "source_type": self._text(record.get("source_type"), limit=32),
                    "summary_text": self._text(record.get("summary_text"), limit=65535),
                }
            )
        elif item_kind == "section_summary":
            row.update(
                {
                    "section_id": int(record.get("section_id", item_id)),
                    "page_start": int(record.get("page_start") or 0),
                    "page_end": int(record.get("page_end") or 0),
                    "section_kind": self._text(record.get("section_kind", "section"), limit=64),
                    "toc_path_text": self._text(" / ".join(cast(list[str], record.get("toc_path", []))), limit=4096),
                    "source_type": self._text(record.get("source_type"), limit=32),
                    "summary_text": self._text(record.get("summary_text"), limit=65535),
                }
            )
        else:
            row.update(
                {
                    "asset_id": int(record.get("asset_id", item_id)),
                    "section_id": int(record.get("section_id") or 0),
                    "asset_type": self._text(record.get("asset_type", "asset"), limit=64),
                    "page_no": int(record.get("page_no") or 0),
                    "caption": self._text(record.get("caption"), limit=65535),
                    "summary_text": self._text(record.get("summary_text"), limit=65535),
                }
            )
        return row, partition_name

    @staticmethod
    def _bm25_text(record: dict[str, Any], *, item_kind: str) -> str:
        if item_kind == "doc_summary":
            parts = [str(record.get("title") or "").strip(), str(record.get("summary_text") or "").strip()]
        elif item_kind == "asset_summary":
            parts = [str(record.get("caption") or "").strip(), str(record.get("summary_text") or "").strip()]
        else:
            toc_path = " / ".join(cast(list[str], record.get("toc_path", [])))
            parts = [toc_path.strip(), str(record.get("summary_text") or "").strip()]
        return "\n".join(part for part in parts if part)

    def _search_collection(
        self,
        collection: Any,
        query_vector: list[float],
        *,
        expr: str,
        limit: int,
        output_fields: list[str],
        partitions: list[str] | None,
    ) -> list[Any]:
        kwargs: dict[str, Any] = {
            "data": [query_vector],
            "anns_field": "embedding",
            "param": {"metric_type": "COSINE", "params": {}},
            "limit": limit,
            "expr": expr,
            "output_fields": output_fields,
            "partition_names": partitions,
            "consistency_level": self._consistency_level,
        }
        try:
            results = cast(list[Any], cast(Any, collection).search(**kwargs))
        except TypeError:
            kwargs.pop("consistency_level", None)
            results = cast(list[Any], cast(Any, collection).search(**kwargs))
        return [] if not results else list(results[0])

    def _validate_query_dimension(
        self,
        collection: Any,
        query_vector: Sequence[float],
        *,
        embedding_space: str,
        item_kind: str,
    ) -> None:
        expected = self._collection_embedding_dimension(collection)
        if expected is None or expected == len(query_vector):
            return
        raise RuntimeError(
            "Milvus embedding dimension mismatch before search: "
            f"collection={getattr(collection, 'name', '<unknown>')!r}, "
            f"item_kind={item_kind!r}, embedding_space={embedding_space!r}, "
            f"expected_dimension={expected}, actual_dimension={len(query_vector)}. "
            "Use the same embedding model/embedding space as the indexed vectors, "
            "or rebuild into a new collection prefix."
        )

    @staticmethod
    def _collection_embedding_dimension(collection: Any) -> int | None:
        schema = getattr(collection, "schema", None)
        fields = getattr(schema, "fields", None)
        if fields is None and isinstance(schema, dict):
            fields = schema.get("fields")
        if not fields:
            return None
        for field in fields:
            name = getattr(field, "name", None)
            params = getattr(field, "params", None)
            if isinstance(field, dict):
                name = field.get("name")
                params = field.get("params")
            if name != "embedding" or not isinstance(params, dict):
                continue
            value = params.get("dim")
            if value is None:
                return None
            try:
                return int(value)
            except (TypeError, ValueError):
                return None
        return None

    def _query_collection(self, collection: Any, *, expr: str, output_fields: list[str]) -> list[dict[str, Any]]:
        kwargs: dict[str, Any] = {
            "expr": expr,
            "output_fields": output_fields,
            "consistency_level": self._consistency_level,
        }
        try:
            rows = cast(list[dict[str, Any]], cast(Any, collection).query(**kwargs))
        except TypeError:
            kwargs.pop("consistency_level", None)
            rows = cast(list[dict[str, Any]], cast(Any, collection).query(**kwargs))
        return rows

    @classmethod
    def _output_fields(cls, *, item_kind: str, include_embedding: bool) -> list[str]:
        common = [
            "item_id",
            "doc_id",
            "source_id",
            "version_group_id",
            "version_no",
            "doc_status",
            "is_active",
            "index_ready",
            "embedding_model_id",
            "metadata_json",
        ]
        if item_kind == "doc_summary":
            fields = common + ["title", "summary_text"]
        elif item_kind == "section_summary":
            fields = common + ["section_id", "page_start", "page_end", "section_kind", "toc_path_text", "summary_text"]
        else:
            fields = common + ["asset_id", "section_id", "asset_type", "page_no", "caption", "summary_text"]
        if include_embedding:
            fields.append("embedding")
        return fields

    def _validate_item_kind(self, item_kind: str) -> None:
        if item_kind not in self._SUPPORTED_KINDS:
            raise ValueError(f"unsupported Milvus item kind: {item_kind}")

    def _search_expr(self, *, doc_ids: list[str] | None, user_expr: str | None) -> str:
        system_expr = "is_active == true and index_ready == true"
        user_clauses: list[str] = []
        if doc_ids:
            user_clauses.append(self._doc_ids_expr(doc_ids))
        if user_expr:
            user_clauses.append(user_expr)
        if not user_clauses:
            return system_expr
        return f"({system_expr}) and " + " and ".join(f"({clause})" for clause in user_clauses)

    def delete(
        self,
        *,
        expr: str,
        item_kind: str | None = None,
        embedding_space: str | None = None,
    ) -> int:
        self._flush_dirty_collections()
        deleted = 0
        for collection in self._iter_target_collections(item_kind=item_kind, embedding_space=embedding_space):
            result = cast(Any, collection).delete(expr)
            cast(Any, collection).flush()
            deleted += int(getattr(result, "delete_count", 0))
        return deleted

    @staticmethod
    def _item_id_expr(item_id: str) -> str:
        return f"item_id in [{int(item_id)}]"

    @staticmethod
    def _item_ids_expr(item_ids: Sequence[str]) -> str:
        return "item_id in [" + ", ".join(str(int(item_id)) for item_id in item_ids) + "]"

    @staticmethod
    def _doc_ids_expr(doc_ids: Sequence[str]) -> str:
        return "doc_id in [" + ", ".join(str(int(doc_id)) for doc_id in doc_ids) + "]"

    @staticmethod
    def _load_metadata(raw_json: object) -> dict[str, str]:
        raw: dict[str, Any]
        if isinstance(raw_json, str):
            loaded = json.loads(raw_json)
            raw = cast(dict[str, Any], loaded if isinstance(loaded, dict) else {})
        else:
            raw = cast(dict[str, Any], raw_json if isinstance(raw_json, dict) else {})
        normalized: dict[str, str] = {}
        for key, value in raw.items():
            if isinstance(value, (dict, list)):
                normalized[key] = json.dumps(value, ensure_ascii=True, sort_keys=True)
            elif value is None:
                normalized[key] = ""
            else:
                normalized[key] = str(value)
        return normalized

    @classmethod
    def _vector_result_from_hit(cls, hit: object, *, item_kind: str) -> VectorSearchResult:
        entity = getattr(hit, "entity", None)
        if entity is None:
            raise RuntimeError("Milvus search hit did not include entity data")
        metadata = cls._load_metadata(entity.get("metadata_json", "{}"))
        if (section_id := entity.get("section_id")) is not None:
            metadata.setdefault("section_id", str(section_id))
        if (asset_id := entity.get("asset_id")) is not None:
            metadata.setdefault("asset_id", str(asset_id))
        return VectorSearchResult(
            item_id=str(entity.get("item_id", "")),
            score=float(getattr(hit, "score", 0.0) or 0.0),
            item_kind=item_kind,
            doc_id=_to_int(entity.get("doc_id", 0)),
            source_id=_to_int(entity.get("source_id", metadata.get("source_id", 0))),
            text=cls._entry_text(entity, item_kind=item_kind),
            metadata=metadata,
        )

    async def hybrid_search_async(
        self,
        *,
        query_vector: Iterable[float],
        sparse_query: str,
        sparse_query_vector: dict[int, float] | None = None,
        limit: int = 10,
        doc_ids: list[str] | None = None,
        expr: str | None = None,
        embedding_space: str = "default",
        item_kind: str = "section_summary",
        fusion_strategy: str = "rrf",
        alpha: float | None = None,
    ) -> list[VectorSearchResult]:
        self._validate_item_kind(item_kind)
        dense_vector = [float(value) for value in query_vector]
        has_sparse_query = bool(sparse_query.strip())
        if not dense_vector or not has_sparse_query:
            return []
        self._flush_dirty_collections()
        if not self._has_collection(item_kind=item_kind, embedding_space=embedding_space):
            return []
        collection = self._collection(item_kind=item_kind, embedding_space=embedding_space)
        self._validate_query_dimension(collection, dense_vector, embedding_space=embedding_space, item_kind=item_kind)
        supports_bm25 = self._supports_bm25_schema(collection)
        if not supports_bm25:
            raise RuntimeError(
                "Milvus collection schema is incompatible with the current sparse layout. "
                "This collection uses the old sparse_embedding field. "
                "Please rebuild the index with a new storage_root or collection prefix."
            )
        final_expr = self._search_expr(doc_ids=doc_ids, user_expr=expr)
        output_fields = self._output_fields(item_kind=item_kind, include_embedding=False)
        async_client = self._get_async_client()
        from pymilvus import AnnSearchRequest, RRFRanker

        requests: list[AnnSearchRequest] = []

        # dense request: 始终创建
        dense_request = AnnSearchRequest(
            data=[dense_vector],
            anns_field="embedding",
            param={"metric_type": "COSINE", "params": {}},
            limit=limit,
            expr=final_expr,
        )
        requests.append(dense_request)

        # BM25 request: 字段 bm25_sparse_embedding，输入 sparse_query（文本），metric BM25
        if not supports_bm25:
            raise RuntimeError(
                "BM25 hybrid search requested but collection schema does not have "
                "bm25_sparse_embedding field. Please rebuild the index with a new storage_root."
            )
        bm25_request = AnnSearchRequest(
            data=[sparse_query],
            anns_field="bm25_sparse_embedding",
            param=self._bm25_search_params(),
            limit=limit,
            expr=final_expr,
        )
        requests.append(bm25_request)

        # TODO: external_sparse_embedding 路径预留。
        # 第一版不实现：文档侧暂无 external sparse embedding 生成逻辑。
        # 即使 query 侧 BGE-M3 产出了 sparse_query_vector，也不创建 external sparse 搜索。

        ranker = self._hybrid_ranker(
            fusion_strategy=fusion_strategy,
            alpha=alpha,
            rrf_ranker=RRFRanker,
        )
        try:
            hits = await async_client.hybrid_search(
                collection_name=collection.name,
                reqs=requests,
                ranker=ranker,
                limit=limit,
                output_fields=output_fields,
                partition_names=["hot"],
                consistency_level=self._consistency_level,
            )
            if not hits:
                hits = await async_client.hybrid_search(
                    collection_name=collection.name,
                    reqs=requests,
                    ranker=ranker,
                    limit=limit,
                    output_fields=output_fields,
                    partition_names=None,
                    consistency_level=self._consistency_level,
                )
            results = [
                self._vector_result_from_client_hit(hit, item_kind=item_kind)
                for hit in (hits[0] if hits else [])
            ]
            results.sort(key=lambda item: (-item.score, item.item_id))
            return results[:limit]
        finally:
            await self._close_async_client_instance(async_client)

    def supports_hybrid_search(self) -> bool:
        return True

    @staticmethod
    def _hybrid_ranker(
        *,
        fusion_strategy: str,
        alpha: float | None,
        rrf_ranker: Any,
    ) -> object:
        normalized_alpha = 0.5 if alpha is None else max(0.0, min(float(alpha), 1.0))
        if fusion_strategy == "weighted_rrf":
            try:
                from pymilvus import WeightedRanker

                dense_weight = round(normalized_alpha, 6)
                sparse_weight = round(1.0 - normalized_alpha, 6)
                return WeightedRanker(dense_weight, sparse_weight)
            except Exception:
                return rrf_ranker()
        return rrf_ranker()

    @staticmethod
    def _bm25_search_params() -> dict[str, Any]:
        """bm25_sparse_embedding 字段的搜索参数。字段→metric 一一对应。"""
        return {"metric_type": "BM25", "params": {}}

    @staticmethod
    def _external_sparse_search_params() -> dict[str, Any]:
        """external_sparse_embedding 字段的搜索参数。字段→metric 一一对应。"""
        return {"metric_type": "IP", "params": {}}

    def _get_async_client(self) -> Any:
        from pymilvus import AsyncMilvusClient

        kwargs: dict[str, object] = {"uri": self._uri}
        if self._token:
            kwargs["token"] = self._token
        if self._db_name:
            kwargs["db_name"] = self._db_name
        return AsyncMilvusClient(**kwargs)

    def _close_async_client(self) -> None:
        return None

    @staticmethod
    async def _close_async_client_instance(async_client: Any) -> None:
        close = getattr(async_client, "close", None)
        if callable(close):
            result = close()
            if inspect.isawaitable(result):
                await result

    @staticmethod
    def _supports_bm25_schema(collection: Any) -> bool:
        schema = getattr(collection, "schema", None)
        fields = getattr(schema, "fields", None)
        if not isinstance(fields, list):
            return False
        field_names = {getattr(field, "name", None) for field in fields}
        return {"bm25_text", "bm25_sparse_embedding", "embedding"} <= field_names

    @staticmethod
    def _supports_external_sparse_schema(collection: Any) -> bool:
        schema = getattr(collection, "schema", None)
        fields = getattr(schema, "fields", None)
        if not isinstance(fields, list):
            return False
        field_names = {getattr(field, "name", None) for field in fields}
        return "external_sparse_embedding" in field_names

    @classmethod
    def _vector_result_from_client_hit(cls, hit: dict[str, Any], *, item_kind: str) -> VectorSearchResult:
        entity = cast(dict[str, Any], hit.get("entity") if isinstance(hit.get("entity"), dict) else hit)
        metadata = cls._load_metadata(entity.get("metadata_json", "{}"))
        for field_name in (
            "source_id",
            "version_group_id",
            "version_no",
            "doc_status",
            "is_active",
            "index_ready",
            "tenant_id",
            "department_id",
            "auth_tag",
            "embedding_model_id",
            "section_id",
            "page_start",
            "page_end",
            "section_kind",
            "toc_path_text",
            "title",
            "source_type",
            "summary_text",
            "asset_id",
            "asset_type",
            "page_no",
            "caption",
        ):
            if field_name in entity and entity[field_name] is not None:
                metadata.setdefault(field_name, str(entity[field_name]))
        item_id = entity.get("item_id", hit.get("id", ""))
        score = hit.get("distance", hit.get("score", 0.0))
        return VectorSearchResult(
            item_id=str(item_id),
            score=float(score or 0.0),
            item_kind=item_kind,
            doc_id=_to_int(entity.get("doc_id", 0)),
            source_id=_to_int(entity.get("source_id", metadata.get("source_id", 0))),
            text=cls._entry_text(entity, item_kind=item_kind),
            metadata=metadata,
        )

    @staticmethod
    def _entry_text(mapping: Any, *, item_kind: str) -> str:
        if item_kind == "doc_summary":
            return str(mapping.get("summary_text") or mapping.get("title") or "")
        if item_kind == "asset_summary":
            return str(mapping.get("summary_text") or mapping.get("caption") or "")
        return str(mapping.get("summary_text") or "")

    @staticmethod
    def _item_kind_for_record(record: SummaryRecord) -> str:
        if isinstance(record, DocSummaryRecord):
            return "doc_summary"
        if isinstance(record, SectionSummaryRecord):
            return "section_summary"
        return "asset_summary"

    @staticmethod
    def _record_item_id(record: SummaryRecord) -> int:
        if isinstance(record, DocSummaryRecord):
            return record.doc_id
        if isinstance(record, SectionSummaryRecord):
            return record.section_id
        return record.asset_id

    @staticmethod
    def _normalize_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
        data = dict(metadata)
        toc_path = data.get("toc_path")
        if toc_path is None:
            data["toc_path"] = []
        elif isinstance(toc_path, tuple):
            data["toc_path"] = list(toc_path)
        return data

    @staticmethod
    def _serialize_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
        serialized: dict[str, Any] = {}
        for key, value in metadata.items():
            if isinstance(value, datetime):
                serialized[key] = value.isoformat()
            elif hasattr(value, "value"):
                serialized[key] = value.value
            elif isinstance(value, tuple):
                serialized[key] = list(value)
            else:
                serialized[key] = value
        return serialized

    @staticmethod
    def _timestamp_ms(value: object) -> int:
        if isinstance(value, datetime):
            return int(value.timestamp() * 1000)
        if isinstance(value, str) and value:
            return int(datetime.fromisoformat(value).timestamp() * 1000)
        if isinstance(value, (int, float)):
            return int(value)
        return 0

    @staticmethod
    def _text(value: object, *, limit: int) -> str:
        if value is None:
            return ""
        text = getattr(value, "value", value)
        return str(text)[:limit]

    @staticmethod
    def _partition_name(metadata: dict[str, Any]) -> str:
        raw = metadata.get("partition_key") or metadata.get("storage_tier")
        normalized = str(getattr(raw, "value", raw or "")).lower()
        if normalized in {"cold", "archive", "historical"}:
            return "cold"
        if metadata.get("is_active") is False:
            return "cold"
        return "hot"

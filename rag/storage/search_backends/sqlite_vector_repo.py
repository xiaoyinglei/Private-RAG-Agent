from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable, Sequence
from math import sqrt
from pathlib import Path
from typing import Any

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


class SQLiteVectorRepo:
    _SUPPORTED_KINDS = {"doc_summary", "section_summary", "asset_summary"}

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA temp_store=MEMORY")
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        columns = {row["name"] for row in self._conn.execute("PRAGMA table_info(vectors)").fetchall()}
        if columns and (
            "item_id" not in columns
            or "item_kind" not in columns
            or "metadata_json" not in columns
            or "source_id" not in columns
        ):
            self._conn.execute("ALTER TABLE vectors RENAME TO vectors_old_contract")

        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS vectors (
                item_id TEXT NOT NULL,
                item_kind TEXT NOT NULL,
                embedding_space TEXT NOT NULL,
                doc_id TEXT NOT NULL,
                source_id TEXT NOT NULL,
                text TEXT NOT NULL,
                metadata_json TEXT NOT NULL,
                vector_norm REAL NOT NULL DEFAULT 0.0,
                vector_json TEXT NOT NULL,
                PRIMARY KEY(item_id, item_kind, embedding_space)
            );

            CREATE INDEX IF NOT EXISTS idx_vectors_doc_id
            ON vectors(doc_id);

            CREATE INDEX IF NOT EXISTS idx_vectors_space_doc_id
            ON vectors(embedding_space, doc_id);

            CREATE INDEX IF NOT EXISTS idx_vectors_kind_space
            ON vectors(item_kind, embedding_space, doc_id);
            """
        )
        columns = {row["name"] for row in self._conn.execute("PRAGMA table_info(vectors)").fetchall()}
        if "vector_norm" not in columns:
            self._conn.execute("ALTER TABLE vectors ADD COLUMN vector_norm REAL NOT NULL DEFAULT 0.0")
        old_contract_exists = self._conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'vectors_old_contract'"
        ).fetchone()
        if old_contract_exists is not None:
            self._conn.execute("DROP TABLE vectors_old_contract")
        self._backfill_vector_norms()
        self._conn.commit()

    def upsert(
        self,
        item_id: str,
        vector: Iterable[float],
        *,
        metadata: dict[str, str] | None = None,
        embedding_space: str = "default",
        item_kind: str = "section_summary",
    ) -> None:
        payload = dict(metadata or {})
        normalized_vector = [float(value) for value in vector]
        self._conn.execute(
            """
            INSERT INTO vectors (
                item_id,
                item_kind,
                embedding_space,
                doc_id,
                source_id,
                text,
                metadata_json,
                vector_norm,
                vector_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(item_id, item_kind, embedding_space) DO UPDATE SET
                doc_id=excluded.doc_id,
                source_id=excluded.source_id,
                text=excluded.text,
                metadata_json=excluded.metadata_json,
                vector_norm=excluded.vector_norm,
                vector_json=excluded.vector_json
            """,
            (
                item_id,
                item_kind,
                embedding_space,
                payload.get("doc_id", ""),
                payload.get("source_id", ""),
                payload.get("text", ""),
                json.dumps(payload, ensure_ascii=True, sort_keys=True),
                self._vector_norm(normalized_vector),
                json.dumps(normalized_vector, ensure_ascii=True),
            ),
        )
        self._conn.commit()

    def upsert_record(
        self,
        record: SummaryRecord,
        vector: Iterable[float],
        *,
        embedding_space: str = "default",
    ) -> None:
        item_kind = self._item_kind_for_record(record)
        payload = self._metadata_for_record(record)
        self.upsert(
            str(self._record_item_id(record)),
            vector,
            metadata=payload,
            embedding_space=embedding_space,
            item_kind=item_kind,
        )

    def upsert_records(
        self,
        items: Sequence[tuple[SummaryRecord, Iterable[float]]],
        *,
        embedding_space: str = "default",
    ) -> None:
        if not items:
            return
        rows = []
        for record, vector in items:
            item_kind = self._item_kind_for_record(record)
            payload = self._metadata_for_record(record)
            normalized_vector = [float(value) for value in vector]
            rows.append(
                (
                    str(self._record_item_id(record)),
                    item_kind,
                    embedding_space,
                    payload.get("doc_id", ""),
                    payload.get("source_id", ""),
                    payload.get("text", ""),
                    json.dumps(payload, ensure_ascii=True, sort_keys=True),
                    self._vector_norm(normalized_vector),
                    json.dumps(normalized_vector, ensure_ascii=True),
                )
            )
        self._conn.executemany(
            """
            INSERT INTO vectors (
                item_id,
                item_kind,
                embedding_space,
                doc_id,
                source_id,
                text,
                metadata_json,
                vector_norm,
                vector_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(item_id, item_kind, embedding_space) DO UPDATE SET
                doc_id=excluded.doc_id,
                source_id=excluded.source_id,
                text=excluded.text,
                metadata_json=excluded.metadata_json,
                vector_norm=excluded.vector_norm,
                vector_json=excluded.vector_json
            """,
            rows,
        )
        self._conn.commit()

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
        query_vector = tuple(float(value) for value in query)
        if not query_vector:
            return []
        query_norm = self._vector_norm(query_vector)
        if query_norm == 0.0:
            return []

        sql = """
            SELECT item_id, doc_id, source_id, text, metadata_json, vector_norm, vector_json
            FROM vectors
            WHERE embedding_space = ? AND item_kind = ?
        """
        params: list[object] = [embedding_space, item_kind]
        rows = self._conn.execute(sql, tuple(params)).fetchall()

        scored: list[VectorSearchResult] = []
        requested_scope = set(doc_ids or [])
        for row in rows:
            vector = tuple(float(value) for value in json.loads(row["vector_json"]))
            metadata = json.loads(row["metadata_json"])
            if requested_scope and not (
                self._vector_scope_tokens(metadata=metadata, row_doc_id=row["doc_id"]) & requested_scope
            ):
                continue
            if not self._matches_expr(metadata, expr):
                continue
            vector_norm = float(row["vector_norm"]) if row["vector_norm"] is not None else self._vector_norm(vector)
            scored.append(
                VectorSearchResult(
                    item_id=row["item_id"],
                    score=self._cosine_similarity(query_vector, query_norm, vector, vector_norm),
                    item_kind=str(item_kind),
                    doc_id=_to_int(row["doc_id"]),
                    source_id=_to_int(metadata.get("source_id", 0)),
                    text=str(row["text"]),
                    metadata=metadata
                    | {
                        "doc_id": row["doc_id"],
                        "source_id": row["source_id"],
                        "text": row["text"],
                    },
                )
            )

        scored.sort(key=lambda result: (-result.score, result.item_id))
        return scored[:limit]

    def get_entry(
        self,
        item_id: str,
        *,
        embedding_space: str = "default",
        item_kind: str = "section_summary",
    ) -> StoredVectorEntry | None:
        row = self._conn.execute(
            """
            SELECT item_id, item_kind, embedding_space, doc_id, source_id, text, metadata_json, vector_json
            FROM vectors
            WHERE item_id = ? AND item_kind = ? AND embedding_space = ?
            """,
            (item_id, item_kind, embedding_space),
        ).fetchone()
        if row is None:
            return None
        return StoredVectorEntry(
            item_id=str(row["item_id"]),
            item_kind=str(row["item_kind"]),
            embedding_space=str(row["embedding_space"]),
            doc_id=_to_int(row["doc_id"]),
            text=str(row["text"]),
            metadata=json.loads(row["metadata_json"]) | {"source_id": str(row["source_id"])},
            vector=[float(value) for value in json.loads(row["vector_json"])],
        )

    def existing_item_ids(
        self,
        item_ids: tuple[str, ...] | list[str],
        *,
        embedding_space: str | None = None,
        item_kind: str | None = "section_summary",
    ) -> set[str]:
        normalized_ids = tuple(dict.fromkeys(item_ids))
        if not normalized_ids:
            return set()
        placeholders = ", ".join("?" for _ in normalized_ids)
        sql = f"SELECT item_id FROM vectors WHERE item_id IN ({placeholders})"
        params: list[object] = list(normalized_ids)
        if item_kind is not None:
            sql += " AND item_kind = ?"
            params.append(item_kind)
        if embedding_space is not None:
            sql += " AND embedding_space = ?"
            params.append(embedding_space)
        rows = self._conn.execute(sql, tuple(params)).fetchall()
        return {str(row["item_id"]) for row in rows}

    def count_vectors(
        self,
        *,
        embedding_space: str | None = None,
        item_kind: str | None = None,
        distinct_records: bool = False,
    ) -> int:
        select = "COUNT(DISTINCT item_id)" if distinct_records else "COUNT(*)"
        sql = f"SELECT {select} AS count FROM vectors"
        clauses = []
        params: list[object] = []
        if embedding_space is not None:
            clauses.append("embedding_space = ?")
            params.append(embedding_space)
        if item_kind is not None:
            clauses.append("item_kind = ?")
            params.append(item_kind)
        if clauses:
            sql += f" WHERE {' AND '.join(clauses)}"
        row = self._conn.execute(sql, params).fetchone()
        return int(row["count"]) if row is not None else 0

    def delete(
        self,
        *,
        expr: str,
        item_kind: str | None = None,
        embedding_space: str | None = None,
    ) -> int:
        clauses: list[str] = []
        params: list[object] = []
        if item_kind is not None:
            clauses.append("item_kind = ?")
            params.append(item_kind)
        if embedding_space is not None:
            clauses.append("embedding_space = ?")
            params.append(embedding_space)
        doc_ids = self._ids_from_in_expr(expr, "doc_id")
        if doc_ids:
            placeholders = ", ".join("?" for _ in doc_ids)
            clauses.append(f"doc_id IN ({placeholders})")
            params.extend(doc_ids)
        item_ids = self._ids_from_in_expr(expr, "item_id")
        if item_ids:
            placeholders = ", ".join("?" for _ in item_ids)
            clauses.append(f"item_id IN ({placeholders})")
            params.extend(item_ids)
        if not clauses:
            return 0
        cursor = self._conn.execute(f"DELETE FROM vectors WHERE {' AND '.join(clauses)}", tuple(params))
        self._conn.commit()
        return int(cursor.rowcount)

    def delete_for_documents(
        self,
        doc_ids: tuple[str, ...] | list[str],
        *,
        item_kind: str | None = None,
    ) -> int:
        normalized_ids = tuple(dict.fromkeys(doc_ids))
        if not normalized_ids:
            return 0
        placeholders = ", ".join("?" for _ in normalized_ids)
        sql = f"DELETE FROM vectors WHERE doc_id IN ({placeholders})"
        params: list[object] = list(normalized_ids)
        if item_kind is not None:
            sql += " AND item_kind = ?"
            params.append(item_kind)
        cursor = self._conn.execute(sql, tuple(params))
        self._conn.commit()
        return int(cursor.rowcount)

    def close(self) -> None:
        self._conn.close()

    @staticmethod
    def _vector_scope_tokens(*, metadata: dict[str, str], row_doc_id: str) -> set[str]:
        tokens = {row_doc_id}
        for key in ("doc_id", "source_id"):
            value = metadata.get(key)
            if value:
                tokens.add(value)
        for key in ("doc_ids", "source_ids"):
            value = metadata.get(key)
            if not value:
                continue
            tokens.update(item.strip() for item in value.split(",") if item.strip())
        return tokens

    @classmethod
    def _item_kind_for_record(cls, record: SummaryRecord) -> str:
        if isinstance(record, DocSummaryRecord):
            return "doc_summary"
        if isinstance(record, AssetSummaryRecord):
            return "asset_summary"
        return "section_summary"

    @staticmethod
    def _record_item_id(record: SummaryRecord) -> int:
        if isinstance(record, DocSummaryRecord):
            return record.doc_id
        if isinstance(record, SectionSummaryRecord):
            return record.section_id
        return record.asset_id

    @classmethod
    def _metadata_for_record(cls, record: SummaryRecord) -> dict[str, str]:
        payload = record.model_dump(mode="json")
        item_kind = cls._item_kind_for_record(record)
        payload["text"] = cls._entry_text(payload, item_kind=item_kind)
        return {key: cls._metadata_value(value) for key, value in payload.items()}

    @staticmethod
    def _metadata_value(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, (dict, list)):
            return json.dumps(value, ensure_ascii=True, sort_keys=True)
        return str(value)

    @staticmethod
    def _entry_text(mapping: dict[str, Any], *, item_kind: str) -> str:
        if item_kind == "doc_summary":
            return str(mapping.get("summary_text") or mapping.get("title") or "")
        if item_kind == "asset_summary":
            return str(mapping.get("summary_text") or mapping.get("caption") or "")
        return str(mapping.get("summary_text") or "")

    @classmethod
    def _matches_expr(cls, metadata: dict[str, str], expr: str | None) -> bool:
        if not expr:
            return True
        if "is_active == true" in expr and metadata.get("is_active", "True").lower() not in {"true", "1"}:
            return False
        if "index_ready == true" in expr and metadata.get("index_ready", "True").lower() not in {"true", "1"}:
            return False
        for field in (
            "doc_id",
            "source_id",
            "tenant_id",
            "department_id",
            "auth_tag",
            "source_type",
            "asset_type",
            "page_no",
        ):
            allowed = cls._values_from_in_expr(expr, field)
            if allowed and metadata.get(field, "") not in allowed:
                return False
        return True

    @classmethod
    def _ids_from_in_expr(cls, expr: str, field: str) -> list[str]:
        return sorted(cls._values_from_in_expr(expr, field))

    @staticmethod
    def _values_from_in_expr(expr: str, field: str) -> set[str]:
        marker = f"{field} in ["
        start = expr.find(marker)
        if start < 0:
            return set()
        start += len(marker)
        end = expr.find("]", start)
        if end < 0:
            return set()
        raw_values = expr[start:end].split(",")
        values: set[str] = set()
        for raw_value in raw_values:
            value = raw_value.strip().strip("'\"")
            if value:
                values.add(value)
        return values

    @staticmethod
    def _vector_norm(vector: Iterable[float]) -> float:
        values = tuple(float(value) for value in vector)
        return sqrt(sum(value * value for value in values))

    @staticmethod
    def _cosine_similarity(
        left: tuple[float, ...],
        left_norm: float,
        right: tuple[float, ...],
        right_norm: float,
    ) -> float:
        if len(left) != len(right):
            return 0.0
        if left_norm == 0.0 or right_norm == 0.0:
            return 0.0
        dot = sum(lv * rv for lv, rv in zip(left, right, strict=True))
        return dot / (left_norm * right_norm)

    def _backfill_vector_norms(self) -> None:
        rows = self._conn.execute(
            """
            SELECT item_id, item_kind, embedding_space, vector_json
            FROM vectors
            WHERE vector_norm = 0.0
            """
        ).fetchall()
        for row in rows:
            vector = json.loads(row["vector_json"])
            self._conn.execute(
                """
                UPDATE vectors
                SET vector_norm = ?
                WHERE item_id = ? AND item_kind = ? AND embedding_space = ?
                """,
                (
                    self._vector_norm(vector),
                    row["item_id"],
                    row["item_kind"],
                    row["embedding_space"],
                ),
            )

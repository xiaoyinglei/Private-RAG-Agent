from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from threading import RLock
from typing import TypeVar

from pydantic import BaseModel

from rag.schema.core import (
    AssetRecord,
    Document,
    LayoutMetaCacheRecord,
    ProcessingStateRecord,
    SectionRecord,
    Source,
)
from rag.schema.query import KnowledgeArtifact
from rag.schema.runtime import CacheEntry, DocumentStatusRecord

TModel = TypeVar("TModel", bound=BaseModel)


class SQLiteMetadataRepo:
    """SQLite metadata repository for the new L1/L2 data contract.

    This implementation intentionally stores only the runtime contract models
    used by the new pipeline: Source, Document, SectionRecord, AssetRecord,
    LayoutMetaCacheRecord, ProcessingStateRecord, artifacts, document status,
    and cache entries. Legacy retrieval-row objects are not part of this repo.
    """

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA temp_store=MEMORY")
        self._lock = RLock()
        self._transaction_depth = 0
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS id_sequence (
                sequence_key TEXT PRIMARY KEY,
                value INTEGER NOT NULL
            );

            CREATE TABLE IF NOT EXISTS sources (
                source_id INTEGER PRIMARY KEY,
                source_type TEXT NOT NULL,
                location TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                ingest_version INTEGER NOT NULL,
                owner_id TEXT,
                updated_at TEXT NOT NULL,
                payload TEXT NOT NULL,
                UNIQUE(location, content_hash, ingest_version)
            );

            CREATE INDEX IF NOT EXISTS idx_sources_location_hash
            ON sources(location, content_hash);

            CREATE TABLE IF NOT EXISTS documents (
                doc_id INTEGER PRIMARY KEY,
                source_id INTEGER NOT NULL,
                location TEXT NOT NULL,
                file_hash TEXT NOT NULL,
                version_group_id INTEGER NOT NULL,
                version_no INTEGER NOT NULL,
                is_active INTEGER NOT NULL,
                index_ready INTEGER NOT NULL,
                storage_tier TEXT NOT NULL,
                tenant_id TEXT,
                department_id TEXT,
                auth_tag TEXT,
                updated_at TEXT NOT NULL,
                payload TEXT NOT NULL,
                UNIQUE(source_id, file_hash, version_no),
                FOREIGN KEY(source_id) REFERENCES sources(source_id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_documents_source
            ON documents(source_id);

            CREATE INDEX IF NOT EXISTS idx_documents_hash
            ON documents(file_hash);

            CREATE INDEX IF NOT EXISTS idx_documents_version
            ON documents(version_group_id, version_no DESC);

            CREATE INDEX IF NOT EXISTS idx_documents_scope
            ON documents(tenant_id, department_id, auth_tag);

            CREATE TABLE IF NOT EXISTS sections (
                section_id INTEGER PRIMARY KEY,
                doc_id INTEGER NOT NULL,
                source_id INTEGER NOT NULL,
                order_index INTEGER NOT NULL,
                page_start INTEGER,
                page_end INTEGER,
                content_hash TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                payload TEXT NOT NULL,
                UNIQUE(doc_id, order_index),
                FOREIGN KEY(doc_id) REFERENCES documents(doc_id) ON DELETE CASCADE,
                FOREIGN KEY(source_id) REFERENCES sources(source_id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_sections_doc_order
            ON sections(doc_id, order_index);

            CREATE INDEX IF NOT EXISTS idx_sections_source_doc
            ON sections(source_id, doc_id);

            CREATE TABLE IF NOT EXISTS assets (
                asset_id INTEGER PRIMARY KEY,
                doc_id INTEGER NOT NULL,
                source_id INTEGER NOT NULL,
                section_id INTEGER,
                page_no INTEGER NOT NULL,
                content_hash TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                payload TEXT NOT NULL,
                UNIQUE(doc_id, content_hash),
                FOREIGN KEY(doc_id) REFERENCES documents(doc_id) ON DELETE CASCADE,
                FOREIGN KEY(source_id) REFERENCES sources(source_id) ON DELETE CASCADE,
                FOREIGN KEY(section_id) REFERENCES sections(section_id) ON DELETE SET NULL
            );

            CREATE INDEX IF NOT EXISTS idx_assets_doc
            ON assets(doc_id);

            CREATE INDEX IF NOT EXISTS idx_assets_section
            ON assets(section_id);

            CREATE TABLE IF NOT EXISTS layout_meta_cache (
                cache_id INTEGER PRIMARY KEY,
                source_id INTEGER NOT NULL,
                doc_id INTEGER,
                content_hash TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                payload TEXT NOT NULL,
                UNIQUE(source_id, content_hash),
                FOREIGN KEY(source_id) REFERENCES sources(source_id) ON DELETE CASCADE,
                FOREIGN KEY(doc_id) REFERENCES documents(doc_id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_layout_meta_cache_doc
            ON layout_meta_cache(doc_id);

            CREATE TABLE IF NOT EXISTS processing_state (
                doc_id INTEGER PRIMARY KEY,
                source_id INTEGER NOT NULL,
                stage TEXT NOT NULL,
                status TEXT NOT NULL,
                priority TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                payload TEXT NOT NULL,
                FOREIGN KEY(doc_id) REFERENCES documents(doc_id) ON DELETE CASCADE,
                FOREIGN KEY(source_id) REFERENCES sources(source_id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_processing_state_stage
            ON processing_state(stage, status, updated_at);

            CREATE TABLE IF NOT EXISTS artifacts (
                artifact_id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                saved_at TEXT NOT NULL,
                payload TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS document_status (
                doc_id INTEGER PRIMARY KEY,
                source_id INTEGER NOT NULL,
                location TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                status TEXT NOT NULL,
                stage TEXT NOT NULL,
                attempts INTEGER NOT NULL,
                error_message TEXT,
                updated_at TEXT NOT NULL,
                payload TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_document_status_source
            ON document_status(source_id, status, updated_at);

            CREATE TABLE IF NOT EXISTS cache_entries (
                namespace TEXT NOT NULL,
                cache_key TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                expires_at TEXT,
                payload TEXT NOT NULL,
                PRIMARY KEY(namespace, cache_key)
            );

            CREATE INDEX IF NOT EXISTS idx_cache_entries_updated
            ON cache_entries(namespace, updated_at);
            """
        )
        self._commit()

    @contextmanager
    def transaction(self) -> Iterator[None]:
        with self._lock:
            outermost = self._transaction_depth == 0
            if outermost:
                self._conn.execute("BEGIN")
            self._transaction_depth += 1
            try:
                yield
            except Exception:
                self._transaction_depth -= 1
                if outermost:
                    self._conn.rollback()
                raise
            else:
                self._transaction_depth -= 1
                if outermost:
                    self._conn.commit()

    def _commit(self) -> None:
        if self._transaction_depth <= 0:
            self._conn.commit()

    @staticmethod
    def _dump(model: BaseModel) -> str:
        return json.dumps(model.model_dump(mode="json"), ensure_ascii=True)

    @staticmethod
    def _load(model_type: type[TModel], payload: str) -> TModel:
        return model_type.model_validate(json.loads(payload))

    def _next_id(self) -> int:
        with self._lock:
            self._conn.execute(
                "INSERT INTO id_sequence(sequence_key, value) VALUES ('global', 0) ON CONFLICT(sequence_key) DO NOTHING"
            )
            self._conn.execute("UPDATE id_sequence SET value = value + 1 WHERE sequence_key = 'global'")
            row = self._conn.execute("SELECT value FROM id_sequence WHERE sequence_key = 'global'").fetchone()
            self._commit()
            if row is None:
                raise RuntimeError("failed to allocate sqlite metadata id")
            return int(row["value"])

    def _source_location_for_document(self, document: Document) -> str:
        source = self.get_source(document.source_id)
        if source is not None:
            return source.location
        return str(document.metadata_json.get("location", "") or document.title or document.doc_id)

    def _existing_source_for_key(self, source: Source) -> Source | None:
        row = self._conn.execute(
            """
            SELECT payload
            FROM sources
            WHERE location = ? AND content_hash = ? AND ingest_version = ?
            LIMIT 1
            """,
            (source.location, source.content_hash, source.ingest_version),
        ).fetchone()
        return None if row is None else self._load(Source, row["payload"])

    def _existing_document_for_key(self, document: Document) -> Document | None:
        row = self._conn.execute(
            """
            SELECT payload
            FROM documents
            WHERE source_id = ? AND file_hash = ? AND version_no = ?
            LIMIT 1
            """,
            (document.source_id, document.file_hash, document.version_no),
        ).fetchone()
        return None if row is None else self._load(Document, row["payload"])

    def save_source(self, source: Source) -> Source:
        now = datetime.now(UTC)
        existing = self._existing_source_for_key(source) if source.source_id <= 0 else None
        source_id = existing.source_id if existing is not None else source.source_id
        saved = source.model_copy(
            update={
                "source_id": source_id if source_id > 0 else self._next_id(),
                "updated_at": now,
            }
        )
        self._conn.execute(
            """
            INSERT INTO sources (
                source_id, source_type, location, content_hash, ingest_version,
                owner_id, updated_at, payload
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(source_id) DO UPDATE SET
                source_type=excluded.source_type,
                location=excluded.location,
                content_hash=excluded.content_hash,
                ingest_version=excluded.ingest_version,
                owner_id=excluded.owner_id,
                updated_at=excluded.updated_at,
                payload=excluded.payload
            """,
            (
                saved.source_id,
                saved.source_type.value,
                saved.location,
                saved.content_hash,
                saved.ingest_version,
                saved.owner_id,
                saved.updated_at.isoformat(),
                self._dump(saved),
            ),
        )
        self._commit()
        return saved

    def get_source(self, source_id: int) -> Source | None:
        row = self._conn.execute("SELECT payload FROM sources WHERE source_id = ?", (source_id,)).fetchone()
        return None if row is None else self._load(Source, row["payload"])

    def get_source_by_location_and_hash(self, location: str, content_hash: str) -> Source | None:
        row = self._conn.execute(
            """
            SELECT payload
            FROM sources
            WHERE location = ? AND content_hash = ?
            ORDER BY ingest_version DESC, updated_at DESC
            LIMIT 1
            """,
            (location, content_hash),
        ).fetchone()
        return None if row is None else self._load(Source, row["payload"])

    def find_source_by_content_hash(self, content_hash: str) -> Source | None:
        row = self._conn.execute(
            """
            SELECT payload
            FROM sources
            WHERE content_hash = ?
            ORDER BY updated_at DESC, source_id DESC
            LIMIT 1
            """,
            (content_hash,),
        ).fetchone()
        return None if row is None else self._load(Source, row["payload"])

    def get_latest_source_for_location(self, location: str) -> Source | None:
        row = self._conn.execute(
            """
            SELECT payload
            FROM sources
            WHERE location = ?
            ORDER BY ingest_version DESC, updated_at DESC
            LIMIT 1
            """,
            (location,),
        ).fetchone()
        return None if row is None else self._load(Source, row["payload"])

    def list_sources(self, location: str | None = None) -> list[Source]:
        if location is None:
            rows = self._conn.execute("SELECT payload FROM sources ORDER BY updated_at DESC, source_id DESC").fetchall()
        else:
            rows = self._conn.execute(
                "SELECT payload FROM sources WHERE location = ? ORDER BY updated_at DESC, source_id DESC",
                (location,),
            ).fetchall()
        return [self._load(Source, row["payload"]) for row in rows]

    def delete_source(self, source_id: int) -> int:
        cursor = self._conn.execute("DELETE FROM sources WHERE source_id = ?", (source_id,))
        self._commit()
        return int(cursor.rowcount)

    def save_document(self, document: Document) -> Document:
        now = datetime.now(UTC)
        existing = self._existing_document_for_key(document) if document.doc_id <= 0 else None
        doc_id = existing.doc_id if existing is not None else document.doc_id
        doc_id = doc_id if doc_id > 0 else self._next_id()
        version_group_id = document.version_group_id if document.version_group_id > 0 else doc_id
        saved = document.model_copy(
            update={
                "doc_id": doc_id,
                "version_group_id": version_group_id,
                "updated_at": now,
            }
        )
        self._conn.execute(
            """
            INSERT INTO documents (
                doc_id, source_id, location, file_hash, version_group_id,
                version_no, is_active, index_ready, storage_tier, tenant_id,
                department_id, auth_tag, updated_at, payload
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(doc_id) DO UPDATE SET
                source_id=excluded.source_id,
                location=excluded.location,
                file_hash=excluded.file_hash,
                version_group_id=excluded.version_group_id,
                version_no=excluded.version_no,
                is_active=excluded.is_active,
                index_ready=excluded.index_ready,
                storage_tier=excluded.storage_tier,
                tenant_id=excluded.tenant_id,
                department_id=excluded.department_id,
                auth_tag=excluded.auth_tag,
                updated_at=excluded.updated_at,
                payload=excluded.payload
            """,
            (
                saved.doc_id,
                saved.source_id,
                self._source_location_for_document(saved),
                saved.file_hash,
                saved.version_group_id,
                saved.version_no,
                1 if saved.is_active else 0,
                1 if saved.index_ready else 0,
                str(saved.storage_tier.value if hasattr(saved.storage_tier, "value") else saved.storage_tier),
                saved.tenant_id,
                saved.department_id,
                saved.auth_tag,
                saved.updated_at.isoformat(),
                self._dump(saved),
            ),
        )
        self._commit()
        return saved

    def get_document(self, doc_id: int) -> Document | None:
        row = self._conn.execute("SELECT payload FROM documents WHERE doc_id = ?", (doc_id,)).fetchone()
        return None if row is None else self._load(Document, row["payload"])

    def find_document_by_hash(self, file_hash: str) -> Document | None:
        row = self._conn.execute(
            """
            SELECT payload
            FROM documents
            WHERE file_hash = ?
            ORDER BY updated_at DESC, doc_id DESC
            LIMIT 1
            """,
            (file_hash,),
        ).fetchone()
        return None if row is None else self._load(Document, row["payload"])

    def list_documents(
        self,
        source_id: int | None = None,
        *,
        active_only: bool = False,
        version_group_id: int | None = None,
    ) -> list[Document]:
        clauses: list[str] = []
        params: list[object] = []
        if source_id is not None:
            clauses.append("source_id = ?")
            params.append(source_id)
        if active_only:
            clauses.append("is_active = 1")
        if version_group_id is not None:
            clauses.append("version_group_id = ?")
            params.append(version_group_id)
        where_sql = "" if not clauses else f" WHERE {' AND '.join(clauses)}"
        rows = self._conn.execute(
            f"SELECT payload FROM documents{where_sql} ORDER BY updated_at DESC, doc_id DESC",
            tuple(params),
        ).fetchall()
        return [self._load(Document, row["payload"]) for row in rows]

    def is_document_active(self, doc_id: int) -> bool:
        row = self._conn.execute("SELECT is_active FROM documents WHERE doc_id = ?", (doc_id,)).fetchone()
        return bool(row["is_active"]) if row is not None else False

    def get_active_document_by_location_and_hash(self, location: str, content_hash: str) -> Document | None:
        row = self._conn.execute(
            """
            SELECT payload
            FROM documents
            WHERE location = ? AND file_hash = ? AND is_active = 1
            ORDER BY updated_at DESC, doc_id DESC
            LIMIT 1
            """,
            (location, content_hash),
        ).fetchone()
        return None if row is None else self._load(Document, row["payload"])

    def get_latest_document_for_location(self, location: str) -> Document | None:
        row = self._conn.execute(
            """
            SELECT payload
            FROM documents
            WHERE location = ?
            ORDER BY updated_at DESC, doc_id DESC
            LIMIT 1
            """,
            (location,),
        ).fetchone()
        return None if row is None else self._load(Document, row["payload"])

    def deactivate_documents_for_location(self, location: str) -> None:
        for row in self._conn.execute("SELECT payload FROM documents WHERE location = ?", (location,)).fetchall():
            document = self._load(Document, row["payload"])
            self.save_document(document.model_copy(update={"is_active": False}))

    def deactivate_document(self, doc_id: int) -> Document:
        document = self.get_document(doc_id)
        if document is None:
            raise KeyError(f"document {doc_id} not found")
        return self.save_document(document.model_copy(update={"is_active": False}))

    def set_document_active(self, doc_id: int, *, active: bool) -> None:
        document = self.get_document(doc_id)
        if document is not None:
            self.save_document(document.model_copy(update={"is_active": active}))

    def set_document_index_state(
        self,
        doc_id: int,
        *,
        is_indexed: bool | None = None,
        index_ready: bool | None = None,
        embedding_model_id: str | None = None,
        indexed_at: datetime | None = None,
        last_index_error: str | None = None,
    ) -> Document:
        document = self.get_document(doc_id)
        if document is None:
            raise KeyError(f"document {doc_id} not found")
        updates: dict[str, object] = {"updated_at": datetime.now(UTC), "last_index_error": last_index_error}
        if is_indexed is not None:
            updates["is_indexed"] = is_indexed
        if index_ready is not None:
            updates["index_ready"] = index_ready
        if embedding_model_id is not None:
            updates["embedding_model_id"] = embedding_model_id
        if indexed_at is not None:
            updates["indexed_at"] = indexed_at
        elif is_indexed:
            updates["indexed_at"] = updates["updated_at"]
        return self.save_document(document.model_copy(update=updates))

    def increment_document_reference_count(self, doc_id: int, *, amount: int = 1) -> Document:
        document = self.get_document(doc_id)
        if document is None:
            raise KeyError(f"document {doc_id} not found")
        return self.save_document(
            document.model_copy(update={"reference_count": max(0, document.reference_count + amount)})
        )

    def set_document_storage_tier(self, doc_id: int, *, storage_tier: object) -> Document:
        document = self.get_document(doc_id)
        if document is None:
            raise KeyError(f"document {doc_id} not found")
        return self.save_document(document.model_copy(update={"storage_tier": storage_tier}))

    def delete_document(self, doc_id: int) -> int:
        cursor = self._conn.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))
        self._commit()
        return int(cursor.rowcount)

    def save_section(self, section: SectionRecord) -> SectionRecord:
        existing = None
        if section.section_id <= 0:
            existing = self._conn.execute(
                "SELECT payload FROM sections WHERE doc_id = ? AND order_index = ?",
                (section.doc_id, section.order_index),
            ).fetchone()
        existing_section = None if existing is None else self._load(SectionRecord, existing["payload"])
        saved = section.model_copy(
            update={
                "section_id": (
                    existing_section.section_id
                    if existing_section is not None
                    else section.section_id if section.section_id > 0 else self._next_id()
                ),
                "updated_at": datetime.now(UTC),
            }
        )
        self._conn.execute(
            """
            INSERT INTO sections (
                section_id, doc_id, source_id, order_index, page_start, page_end,
                content_hash, updated_at, payload
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(section_id) DO UPDATE SET
                doc_id=excluded.doc_id,
                source_id=excluded.source_id,
                order_index=excluded.order_index,
                page_start=excluded.page_start,
                page_end=excluded.page_end,
                content_hash=excluded.content_hash,
                updated_at=excluded.updated_at,
                payload=excluded.payload
            """,
            (
                saved.section_id,
                saved.doc_id,
                saved.source_id,
                saved.order_index,
                saved.page_start,
                saved.page_end,
                saved.content_hash,
                saved.updated_at.isoformat(),
                self._dump(saved),
            ),
        )
        self._commit()
        return saved

    def save_sections(self, sections: list[SectionRecord]) -> list[SectionRecord]:
        return [self.save_section(section) for section in sections]

    def get_section(self, section_id: int) -> SectionRecord | None:
        row = self._conn.execute("SELECT payload FROM sections WHERE section_id = ?", (section_id,)).fetchone()
        return None if row is None else self._load(SectionRecord, row["payload"])

    def list_sections(
        self,
        *,
        doc_id: int | None = None,
        source_id: int | None = None,
    ) -> list[SectionRecord]:
        clauses: list[str] = []
        params: list[object] = []
        if doc_id is not None:
            clauses.append("doc_id = ?")
            params.append(doc_id)
        if source_id is not None:
            clauses.append("source_id = ?")
            params.append(source_id)
        where_sql = "" if not clauses else f" WHERE {' AND '.join(clauses)}"
        rows = self._conn.execute(
            f"SELECT payload FROM sections{where_sql} ORDER BY doc_id, order_index, section_id",
            tuple(params),
        ).fetchall()
        return [self._load(SectionRecord, row["payload"]) for row in rows]

    def delete_sections_for_document(self, *, doc_id: int) -> int:
        cursor = self._conn.execute("DELETE FROM sections WHERE doc_id = ?", (doc_id,))
        self._commit()
        return int(cursor.rowcount)

    def save_asset(self, asset: AssetRecord) -> AssetRecord:
        existing = None
        if asset.asset_id <= 0:
            existing = self._conn.execute(
                "SELECT payload FROM assets WHERE doc_id = ? AND content_hash = ?",
                (asset.doc_id, asset.content_hash),
            ).fetchone()
        existing_asset = None if existing is None else self._load(AssetRecord, existing["payload"])
        saved = asset.model_copy(
            update={
                "asset_id": (
                    existing_asset.asset_id
                    if existing_asset is not None
                    else asset.asset_id if asset.asset_id > 0 else self._next_id()
                ),
                "updated_at": datetime.now(UTC),
            }
        )
        self._conn.execute(
            """
            INSERT INTO assets (
                asset_id, doc_id, source_id, section_id, page_no, content_hash,
                updated_at, payload
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(asset_id) DO UPDATE SET
                doc_id=excluded.doc_id,
                source_id=excluded.source_id,
                section_id=excluded.section_id,
                page_no=excluded.page_no,
                content_hash=excluded.content_hash,
                updated_at=excluded.updated_at,
                payload=excluded.payload
            """,
            (
                saved.asset_id,
                saved.doc_id,
                saved.source_id,
                saved.section_id,
                saved.page_no,
                saved.content_hash,
                saved.updated_at.isoformat(),
                self._dump(saved),
            ),
        )
        self._commit()
        return saved

    def save_assets(self, assets: list[AssetRecord]) -> list[AssetRecord]:
        return [self.save_asset(asset) for asset in assets]

    def get_asset(self, asset_id: int) -> AssetRecord | None:
        row = self._conn.execute("SELECT payload FROM assets WHERE asset_id = ?", (asset_id,)).fetchone()
        return None if row is None else self._load(AssetRecord, row["payload"])

    def list_assets(
        self,
        *,
        doc_id: int | None = None,
        source_id: int | None = None,
        section_id: int | None = None,
    ) -> list[AssetRecord]:
        clauses: list[str] = []
        params: list[object] = []
        if doc_id is not None:
            clauses.append("doc_id = ?")
            params.append(doc_id)
        if source_id is not None:
            clauses.append("source_id = ?")
            params.append(source_id)
        if section_id is not None:
            clauses.append("section_id = ?")
            params.append(section_id)
        where_sql = "" if not clauses else f" WHERE {' AND '.join(clauses)}"
        rows = self._conn.execute(
            f"SELECT payload FROM assets{where_sql} ORDER BY doc_id, page_no, asset_id",
            tuple(params),
        ).fetchall()
        return [self._load(AssetRecord, row["payload"]) for row in rows]

    def delete_assets_for_document(self, *, doc_id: int) -> int:
        cursor = self._conn.execute("DELETE FROM assets WHERE doc_id = ?", (doc_id,))
        self._commit()
        return int(cursor.rowcount)

    def save_layout_meta_cache(self, record: LayoutMetaCacheRecord) -> LayoutMetaCacheRecord:
        existing = None
        if record.cache_id <= 0:
            existing = self._conn.execute(
                "SELECT payload FROM layout_meta_cache WHERE source_id = ? AND content_hash = ?",
                (record.source_id, record.content_hash),
            ).fetchone()
        existing_cache = None if existing is None else self._load(LayoutMetaCacheRecord, existing["payload"])
        saved = record.model_copy(
            update={
                "cache_id": (
                    existing_cache.cache_id
                    if existing_cache is not None
                    else record.cache_id if record.cache_id > 0 else self._next_id()
                ),
                "updated_at": datetime.now(UTC),
            }
        )
        self._conn.execute(
            """
            INSERT INTO layout_meta_cache (
                cache_id, source_id, doc_id, content_hash, updated_at, payload
            )
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(cache_id) DO UPDATE SET
                source_id=excluded.source_id,
                doc_id=excluded.doc_id,
                content_hash=excluded.content_hash,
                updated_at=excluded.updated_at,
                payload=excluded.payload
            """,
            (
                saved.cache_id,
                saved.source_id,
                saved.doc_id,
                saved.content_hash,
                saved.updated_at.isoformat(),
                self._dump(saved),
            ),
        )
        self._commit()
        return saved

    def get_layout_meta_cache(
        self,
        *,
        source_id: int | None = None,
        doc_id: int | None = None,
        content_hash: str | None = None,
    ) -> LayoutMetaCacheRecord | None:
        clauses: list[str] = []
        params: list[object] = []
        if source_id is not None:
            clauses.append("source_id = ?")
            params.append(source_id)
        if doc_id is not None:
            clauses.append("doc_id = ?")
            params.append(doc_id)
        if content_hash is not None:
            clauses.append("content_hash = ?")
            params.append(content_hash)
        if not clauses:
            raise ValueError("at least one layout cache filter is required")
        row = self._conn.execute(
            f"""
            SELECT payload
            FROM layout_meta_cache
            WHERE {' AND '.join(clauses)}
            ORDER BY updated_at DESC, cache_id DESC
            LIMIT 1
            """,
            tuple(params),
        ).fetchone()
        return None if row is None else self._load(LayoutMetaCacheRecord, row["payload"])

    def list_layout_meta_cache(
        self,
        *,
        source_id: int | None = None,
        doc_id: int | None = None,
    ) -> list[LayoutMetaCacheRecord]:
        clauses: list[str] = []
        params: list[object] = []
        if source_id is not None:
            clauses.append("source_id = ?")
            params.append(source_id)
        if doc_id is not None:
            clauses.append("doc_id = ?")
            params.append(doc_id)
        where_sql = "" if not clauses else f" WHERE {' AND '.join(clauses)}"
        rows = self._conn.execute(
            f"SELECT payload FROM layout_meta_cache{where_sql} ORDER BY updated_at DESC, cache_id DESC",
            tuple(params),
        ).fetchall()
        return [self._load(LayoutMetaCacheRecord, row["payload"]) for row in rows]

    def save_processing_state(self, record: ProcessingStateRecord) -> ProcessingStateRecord:
        saved = record.model_copy(update={"updated_at": datetime.now(UTC)})
        self._conn.execute(
            """
            INSERT INTO processing_state (
                doc_id, source_id, stage, status, priority, updated_at, payload
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(doc_id) DO UPDATE SET
                source_id=excluded.source_id,
                stage=excluded.stage,
                status=excluded.status,
                priority=excluded.priority,
                updated_at=excluded.updated_at,
                payload=excluded.payload
            """,
            (
                saved.doc_id,
                saved.source_id,
                saved.stage,
                saved.status,
                saved.priority,
                saved.updated_at.isoformat(),
                self._dump(saved),
            ),
        )
        self._commit()
        return saved

    def get_processing_state(self, doc_id: int) -> ProcessingStateRecord | None:
        row = self._conn.execute("SELECT payload FROM processing_state WHERE doc_id = ?", (doc_id,)).fetchone()
        return None if row is None else self._load(ProcessingStateRecord, row["payload"])

    def list_processing_states(
        self,
        *,
        source_id: int | None = None,
        status: str | None = None,
        stage: str | None = None,
    ) -> list[ProcessingStateRecord]:
        clauses: list[str] = []
        params: list[object] = []
        if source_id is not None:
            clauses.append("source_id = ?")
            params.append(source_id)
        if status is not None:
            clauses.append("status = ?")
            params.append(status)
        if stage is not None:
            clauses.append("stage = ?")
            params.append(stage)
        where_sql = "" if not clauses else f" WHERE {' AND '.join(clauses)}"
        rows = self._conn.execute(
            f"SELECT payload FROM processing_state{where_sql} ORDER BY updated_at DESC, doc_id DESC",
            tuple(params),
        ).fetchall()
        return [self._load(ProcessingStateRecord, row["payload"]) for row in rows]

    def delete_processing_state(self, doc_id: int) -> None:
        self._conn.execute("DELETE FROM processing_state WHERE doc_id = ?", (doc_id,))
        self._commit()

    def save_artifact(self, artifact: KnowledgeArtifact) -> None:
        self._conn.execute(
            """
            INSERT INTO artifacts (artifact_id, status, saved_at, payload)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(artifact_id) DO UPDATE SET
                status=excluded.status,
                saved_at=excluded.saved_at,
                payload=excluded.payload
            """,
            (artifact.artifact_id, artifact.status.value, datetime.now(UTC).isoformat(), self._dump(artifact)),
        )
        self._commit()

    def get_artifact(self, artifact_id: str) -> KnowledgeArtifact | None:
        row = self._conn.execute("SELECT payload FROM artifacts WHERE artifact_id = ?", (artifact_id,)).fetchone()
        return None if row is None else self._load(KnowledgeArtifact, row["payload"])

    def list_artifacts(self) -> list[KnowledgeArtifact]:
        rows = self._conn.execute("SELECT payload FROM artifacts ORDER BY saved_at, artifact_id").fetchall()
        return [self._load(KnowledgeArtifact, row["payload"]) for row in rows]

    def save_document_status(self, status: DocumentStatusRecord) -> DocumentStatusRecord:
        normalized = status.model_copy(update={"updated_at": datetime.now(UTC)})
        self._conn.execute(
            """
            INSERT INTO document_status (
                doc_id, source_id, location, content_hash, status, stage,
                attempts, error_message, updated_at, payload
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(doc_id) DO UPDATE SET
                source_id=excluded.source_id,
                location=excluded.location,
                content_hash=excluded.content_hash,
                status=excluded.status,
                stage=excluded.stage,
                attempts=excluded.attempts,
                error_message=excluded.error_message,
                updated_at=excluded.updated_at,
                payload=excluded.payload
            """,
            (
                normalized.doc_id,
                normalized.source_id,
                normalized.location,
                normalized.content_hash,
                normalized.status.value,
                str(normalized.stage.value if hasattr(normalized.stage, "value") else normalized.stage),
                normalized.attempts,
                normalized.error_message,
                normalized.updated_at.isoformat(),
                self._dump(normalized),
            ),
        )
        self._commit()
        return normalized

    def get_document_status(self, doc_id: int) -> DocumentStatusRecord | None:
        row = self._conn.execute("SELECT payload FROM document_status WHERE doc_id = ?", (doc_id,)).fetchone()
        return None if row is None else self._load(DocumentStatusRecord, row["payload"])

    def list_document_statuses(
        self,
        *,
        source_id: int | None = None,
        status: str | None = None,
    ) -> list[DocumentStatusRecord]:
        clauses: list[str] = []
        params: list[object] = []
        if source_id is not None:
            clauses.append("source_id = ?")
            params.append(source_id)
        if status is not None:
            clauses.append("status = ?")
            params.append(status)
        where_sql = "" if not clauses else f" WHERE {' AND '.join(clauses)}"
        rows = self._conn.execute(
            f"SELECT payload FROM document_status{where_sql} ORDER BY updated_at DESC, doc_id DESC",
            tuple(params),
        ).fetchall()
        return [self._load(DocumentStatusRecord, row["payload"]) for row in rows]

    def delete_document_status(self, doc_id: int) -> None:
        self._conn.execute("DELETE FROM document_status WHERE doc_id = ?", (doc_id,))
        self._commit()

    def save_cache_entry(self, entry: CacheEntry) -> CacheEntry:
        now = datetime.now(UTC)
        existing = self.get_cache_entry(entry.cache_key, namespace=entry.namespace)
        normalized = entry.model_copy(
            update={
                "created_at": existing.created_at if existing is not None else entry.created_at,
                "updated_at": now,
            }
        )
        self._conn.execute(
            """
            INSERT INTO cache_entries (
                namespace, cache_key, created_at, updated_at, expires_at, payload
            )
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(namespace, cache_key) DO UPDATE SET
                updated_at=excluded.updated_at,
                expires_at=excluded.expires_at,
                payload=excluded.payload
            """,
            (
                normalized.namespace,
                normalized.cache_key,
                normalized.created_at.isoformat(),
                normalized.updated_at.isoformat(),
                normalized.expires_at.isoformat() if normalized.expires_at is not None else None,
                self._dump(normalized),
            ),
        )
        self._commit()
        return normalized

    def get_cache_entry(self, cache_key: str, *, namespace: str = "default") -> CacheEntry | None:
        row = self._conn.execute(
            """
            SELECT payload
            FROM cache_entries
            WHERE namespace = ? AND cache_key = ?
            """,
            (namespace, cache_key),
        ).fetchone()
        return None if row is None else self._load(CacheEntry, row["payload"])

    def list_cache_entries(self, *, namespace: str | None = None) -> list[CacheEntry]:
        if namespace is None:
            rows = self._conn.execute(
                "SELECT payload FROM cache_entries ORDER BY updated_at DESC, namespace, cache_key"
            ).fetchall()
        else:
            rows = self._conn.execute(
                """
                SELECT payload
                FROM cache_entries
                WHERE namespace = ?
                ORDER BY updated_at DESC, cache_key
                """,
                (namespace,),
            ).fetchall()
        return [self._load(CacheEntry, row["payload"]) for row in rows]

    def delete_cache_entry(self, cache_key: str, *, namespace: str = "default") -> None:
        self._conn.execute("DELETE FROM cache_entries WHERE namespace = ? AND cache_key = ?", (namespace, cache_key))
        self._commit()

    def purge_expired_cache_entries(self, *, now: datetime | None = None) -> int:
        effective_now = (now or datetime.now(UTC)).isoformat()
        cursor = self._conn.execute(
            """
            DELETE FROM cache_entries
            WHERE expires_at IS NOT NULL AND expires_at <= ?
            """,
            (effective_now,),
        )
        self._commit()
        return int(cursor.rowcount)

    def close(self) -> None:
        self._conn.close()

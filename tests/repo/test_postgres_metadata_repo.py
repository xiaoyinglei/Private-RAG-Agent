from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

from rag.schema.core import AssetRecord, Document, ProcessingStateRecord, SectionRecord, Source, SourceType
from rag.storage.repositories.postgres_metadata_repo import PostgresMetadataRepo


class _Cursor:
    def __init__(self, row: dict[str, Any] | None = None, rows: list[dict[str, Any]] | None = None) -> None:
        self._row = row
        self._rows = rows or ([] if row is None else [row])
        self.rowcount = len(self._rows)

    def fetchone(self) -> dict[str, Any] | None:
        return self._row

    def fetchall(self) -> list[dict[str, Any]]:
        return list(self._rows)


class _Connection:
    def __init__(self, handler) -> None:
        self._handler = handler
        self.calls: list[tuple[str, tuple[object, ...]]] = []
        self.commit_count = 0

    def execute(self, sql: str, params: tuple[object, ...] = ()) -> _Cursor:
        placeholder_count = sql.count("%s")
        assert placeholder_count == len(params)
        self.calls.append((sql, params))
        return self._handler(sql, params)

    def commit(self) -> None:
        self.commit_count += 1

    def close(self) -> None:
        return None


def _build_repo(
    monkeypatch: pytest.MonkeyPatch,
    conn: _Connection,
    *,
    ensure_schema: bool = True,
) -> PostgresMetadataRepo:
    monkeypatch.setattr(PostgresMetadataRepo, "_connect", lambda self: conn)
    if not ensure_schema:
        monkeypatch.setattr(PostgresMetadataRepo, "_ensure_schema", lambda self: None)
    return PostgresMetadataRepo("postgresql://unit-test")


def test_postgres_metadata_repo_bootstraps_v1_schema(monkeypatch: pytest.MonkeyPatch) -> None:
    repo = _build_repo(monkeypatch, _Connection(lambda sql, params: _Cursor()))
    repo.close()

    ddl = "\n".join(sql for sql, _ in repo._conn.calls)  # type: ignore[attr-defined]
    assert "payload" not in ddl.lower()
    assert "legacy" not in ddl.lower()
    assert "CREATE TABLE IF NOT EXISTS public.sections" in ddl
    assert "source_id BIGINT NOT NULL REFERENCES public.sources(source_id) ON DELETE CASCADE" in ddl
    assert "CREATE TABLE IF NOT EXISTS public.assets" in ddl
    assert "section_id BIGINT REFERENCES public.sections(section_id) ON DELETE CASCADE" in ddl
    assert "CREATE TABLE IF NOT EXISTS public.layout_meta_cache" in ddl
    assert "CREATE TABLE IF NOT EXISTS public.processing_state" in ddl


def test_postgres_metadata_repo_saves_v1_records(monkeypatch: pytest.MonkeyPatch) -> None:
    now = datetime(2026, 4, 18, tzinfo=UTC)
    section_locator = {
        "visible_text_key": "objects/sections/21.md",
        "char_range_start": 0,
        "char_range_end": 256,
        "byte_range_start": 0,
        "byte_range_end": 512,
    }

    def handler(sql: str, params: tuple[object, ...]) -> _Cursor:
        if "INSERT INTO public.sources" in sql:
            return _Cursor(
                {
                    "source_id": 7,
                    "source_type": "markdown",
                    "location": "docs/spec.md",
                    "original_file_name": "spec.md",
                    "bucket": "raw",
                    "object_key": "docs/spec.md",
                    "content_hash": "source-hash",
                    "file_size_bytes": 1024,
                    "mime_type": "text/markdown",
                    "owner_id": "alice",
                    "ingest_version": 1,
                    "created_at": now,
                    "updated_at": now,
                    "metadata_json": {"channel": "design"},
                }
            )
        if "INSERT INTO public.documents" in sql:
            assert params[1] == 7
            return _Cursor(
                {
                    "doc_id": 11,
                    "source_id": 7,
                    "title": "Storage Blueprint",
                    "language": "zh",
                    "authors": ["alice"],
                    "file_hash": "doc-hash",
                    "version_group_id": 11,
                    "version_no": 1,
                    "doc_status": "published",
                    "effective_date": now,
                    "is_active": True,
                    "is_indexed": False,
                    "index_ready": False,
                    "index_priority": "high",
                    "storage_tier": "hot",
                    "reference_count": 1,
                    "page_count": 12,
                    "tenant_id": "tenant-a",
                    "department_id": "dept-a",
                    "auth_tag": "internal",
                    "embedding_model_id": "bge-v1",
                    "indexed_at": None,
                    "last_index_error": None,
                    "created_at": now,
                    "updated_at": now,
                    "metadata_json": {"pipeline": "v1"},
                }
            )
        if "INSERT INTO public.sections" in sql:
            assert params[2] == 7
            return _Cursor(
                {
                    "section_id": 21,
                    "doc_id": 11,
                    "source_id": 7,
                    "parent_section_id": None,
                    "toc_path": ["A", "B"],
                    "heading_level": 2,
                    "order_index": 0,
                    "anchor": "a-b",
                    "page_start": 1,
                    "page_end": 2,
                        "raw_locator": section_locator,
                        "char_range_start": 0,
                        "char_range_end": 256,
                        "byte_range_start": 0,
                        "byte_range_end": 512,
                    "visible_text_key": "objects/sections/21.md",
                    "section_kind": "body",
                    "content_hash": "section-hash",
                    "has_table": True,
                    "has_figure": False,
                    "neighbor_asset_count": 1,
                    "created_at": now,
                    "updated_at": now,
                    "metadata_json": {"lang": "zh"},
                }
            )
        if "INSERT INTO public.assets" in sql:
            assert params[2] == 7
            return _Cursor(
                {
                    "asset_id": 31,
                    "doc_id": 11,
                    "source_id": 7,
                    "section_id": 21,
                    "asset_type": "table",
                    "element_ref": "tbl-1",
                    "page_no": 2,
                    "bbox": {"x": 1, "y": 2, "w": 3, "h": 4},
                    "caption": "Summary Table",
                    "raw_locator": {"page": 2},
                    "neighbor_section_id": 21,
                    "content_hash": "asset-hash",
                    "storage_key": "objects/assets/31.json",
                    "created_at": now,
                    "updated_at": now,
                    "metadata_json": {"kind": "table"},
                }
            )
        if "INSERT INTO public.processing_state" in sql:
            return _Cursor(
                {
                    "doc_id": 11,
                    "source_id": 7,
                    "stage": "index",
                    "status": "processing",
                    "attempts": 1,
                    "priority": "high",
                    "worker_id": "worker-a",
                    "lease_expires_at": None,
                    "error_message": None,
                    "created_at": now,
                    "updated_at": now,
                    "metadata_json": {"partition": "hot"},
                }
            )
        return _Cursor()

    conn = _Connection(handler)
    repo = _build_repo(monkeypatch, conn, ensure_schema=False)

    source = repo.save_source(
        Source(
            source_type=SourceType.MARKDOWN,
            location="docs/spec.md",
            original_file_name="spec.md",
            bucket="raw",
            object_key="docs/spec.md",
            content_hash="source-hash",
            file_size_bytes=1024,
            mime_type="text/markdown",
            owner_id="alice",
            metadata_json={"channel": "design"},
            created_at=now,
            updated_at=now,
        )
    )
    document = repo.save_document(
        Document(
            source_id=source.source_id,
            title="Storage Blueprint",

            language="zh",
            authors=["alice"],
            file_hash="doc-hash",
            doc_status="published",
            effective_date=now,
            page_count=12,
            tenant_id="tenant-a",
            department_id="dept-a",
            auth_tag="internal",
            embedding_model_id="bge-v1",
            metadata_json={"pipeline": "v1"},
            created_at=now,
            updated_at=now,
        )
    )
    section = repo.save_section(
        SectionRecord(
            doc_id=document.doc_id,
            source_id=source.source_id,
            toc_path=["A", "B"],
            heading_level=2,
            order_index=0,
            anchor="a-b",
            page_start=1,
            page_end=2,
            raw_locator=section_locator,
            char_range_start=0,
            char_range_end=256,
            byte_range_start=0,
            byte_range_end=512,
            visible_text_key="objects/sections/21.md",
            section_kind="body",
            content_hash="section-hash",
            has_table=True,
            neighbor_asset_count=1,
            metadata_json={"lang": "zh"},
            created_at=now,
            updated_at=now,
        )
    )
    asset = repo.save_asset(
        AssetRecord(
            doc_id=document.doc_id,
            source_id=source.source_id,
            section_id=section.section_id,
            asset_type="table",
            element_ref="tbl-1",
            page_no=2,
            bbox={"x": 1, "y": 2, "w": 3, "h": 4},
            caption="Summary Table",
            raw_locator={"page": 2},
            neighbor_section_id=section.section_id,
            content_hash="asset-hash",
            storage_key="objects/assets/31.json",
            metadata_json={"kind": "table"},
            created_at=now,
            updated_at=now,
        )
    )
    state = repo.save_processing_state(
        ProcessingStateRecord(
            doc_id=document.doc_id,
            source_id=source.source_id,
            stage="index",
            status="processing",
            attempts=1,
            priority="high",
            worker_id="worker-a",
            metadata_json={"partition": "hot"},
            created_at=now,
            updated_at=now,
        )
    )

    assert source.source_id == 7
    assert document.doc_id == 11
    assert document.version_group_id == 11
    assert section.source_id == 7
    assert asset.source_id == 7
    assert state.stage == "index"
    assert conn.commit_count == 5

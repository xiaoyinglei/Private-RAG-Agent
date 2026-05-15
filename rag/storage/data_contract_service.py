from __future__ import annotations

import hashlib
import json
import logging
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from rag.schema.core import (
    AssetRecord,
    AssetSummaryRecord,
    DocSummaryRecord,
    Document,
    PartitionKey,
    ProcessingStateRecord,
    SectionRecord,
    SectionSummaryRecord,
    Source,
    SourceType,
)
from rag.schema.runtime import DataContractMetadataRepo, VectorSearchResult
from rag.storage.index_sync_service import IndexSyncService, StaleProcessingStateError

_PHONE_PATTERN = re.compile(r"\b1[3-9]\d{9}\b")
_SUMMARY_SPEC_VERSION = "v1"


@dataclass(frozen=True, slots=True)
class DocumentRegistrationResult:
    source: Source | None
    document: Document
    is_duplicate: bool


class SummaryIndexRepo(Protocol):
    def search(
        self,
        query: Iterable[float],
        *,
        limit: int = 10,
        doc_ids: list[str] | None = None,
        expr: str | None = None,
        embedding_space: str = "default",
        item_kind: str = "section_summary",
    ) -> list[VectorSearchResult]: ...

    def delete(self, *, expr: str, embedding_space: str | None = None, item_kind: str | None = None) -> int: ...

    def upsert_record(
        self,
        record: DocSummaryRecord | SectionSummaryRecord | AssetSummaryRecord,
        vector: Iterable[float],
        *,
        embedding_space: str = "default",
    ) -> None: ...

    def upsert_records(
        self,
        items: Sequence[tuple[DocSummaryRecord | SectionSummaryRecord | AssetSummaryRecord, Iterable[float]]],
        *,
        embedding_space: str = "default",
    ) -> None: ...


class DataContractService:
    def __init__(
        self,
        metadata_repo: DataContractMetadataRepo,
        milvus_repo: SummaryIndexRepo,
        *,
        embedder: object | None = None,
        embedding_space: str = "default",
        index_sync_service: IndexSyncService | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.metadata_repo = metadata_repo
        self.milvus_repo = milvus_repo
        self.embedder = embedder
        self.embedding_space = embedding_space
        self.index_sync_service = index_sync_service or (
            IndexSyncService(metadata_repo)
            if callable(getattr(metadata_repo, "save_processing_state", None))
            and callable(getattr(metadata_repo, "list_processing_states", None))
            else None
        )
        self.logger = logger or logging.getLogger(__name__)

    def compute_file_hash(self, file_bytes: bytes, *, algorithm: str = "sha256") -> str:
        hasher = hashlib.new(algorithm)
        hasher.update(file_bytes)
        return hasher.hexdigest()

    def mask_pii(self, text: str | None) -> str:
        if not text:
            return ""
        return _PHONE_PATTERN.sub("[手机号脱敏]", text)

    def register_document(
        self,
        *,
        source: Source,
        document: Document,
        file_bytes: bytes,
        hash_algorithm: str = "sha256",
        increment_reference: bool = True,
    ) -> DocumentRegistrationResult:
        file_hash = self.compute_file_hash(file_bytes, algorithm=hash_algorithm)
        existing = self.metadata_repo.find_document_by_hash(file_hash)
        if existing is not None:
            if increment_reference:
                existing = self.metadata_repo.increment_document_reference_count(existing.doc_id)
            return DocumentRegistrationResult(source=None, document=existing, is_duplicate=True)

        saved_source = self.metadata_repo.save_source(
            source.model_copy(
                update={
                    "content_hash": file_hash,
                    "original_file_name": self.mask_pii(source.original_file_name),
                }
            )
        )
        saved_document = self.metadata_repo.save_document(
            document.model_copy(
                update={
                    "source_id": saved_source.source_id,
                    "file_hash": file_hash,
                    "title": self._masked_optional_text(document.title),
                }
            )
        )
        return DocumentRegistrationResult(source=saved_source, document=saved_document, is_duplicate=False)

    def save_doc_summary(
        self,
        document: Document,
        *,
        source_type: SourceType | None,
        summary_text: str,
        is_urgent: bool = True,
    ) -> DocSummaryRecord:
        masked_summary, summary_contract = self._compile_doc_summary(
            document=document,
            source_type=source_type,
            raw_summary=summary_text,
        )
        updated_document = self.metadata_repo.save_document(
            document.model_copy(
                update={
                    "title": self._masked_optional_text(document.title),
                    "metadata_json": {
                        **document.metadata_json,
                        "summary_text": masked_summary,
                        "summary_contract": summary_contract,
                    },
                }
            )
        )
        record = DocSummaryRecord(
            doc_id=updated_document.doc_id,
            source_id=updated_document.source_id,
            version_group_id=updated_document.version_group_id,
            version_no=updated_document.version_no,
            doc_status=updated_document.doc_status,
            effective_date=updated_document.effective_date,
            updated_at=updated_document.updated_at,
            is_active=updated_document.is_active,
            index_ready=is_urgent,
            tenant_id=updated_document.tenant_id,
            department_id=updated_document.department_id,
            auth_tag=updated_document.auth_tag,
            source_type=source_type,
            embedding_model_id=updated_document.embedding_model_id,
            partition_key=self._partition_key(updated_document),
            title=self._masked_optional_text(updated_document.title),
            summary_text=masked_summary,
            metadata_json=updated_document.metadata_json,
        )
        self._maybe_index_record(updated_document, record, text=masked_summary, is_urgent=is_urgent)
        return record

    def save_section(
        self,
        document: Document,
        section: SectionRecord,
        *,
        source_type: SourceType | None,
        summary_text: str,
        is_urgent: bool = True,
    ) -> SectionRecord:
        saved_section = self.metadata_repo.save_section(section)
        masked_summary, summary_contract = self._compile_section_summary(
            section=saved_section,
            raw_summary=summary_text,
        )
        saved_section = self.metadata_repo.save_section(
            saved_section.model_copy(
                update={
                    "metadata_json": {
                        **saved_section.metadata_json,
                        "summary_text": masked_summary,
                        "summary_contract": summary_contract,
                    }
                }
            )
        )
        record = SectionSummaryRecord(
            section_id=saved_section.section_id,
            doc_id=saved_section.doc_id,
            source_id=saved_section.source_id,
            version_group_id=document.version_group_id,
            version_no=document.version_no,
            doc_status=document.doc_status,
            effective_date=document.effective_date,
            updated_at=saved_section.updated_at,
            is_active=document.is_active,
            index_ready=is_urgent,
            tenant_id=document.tenant_id,
            department_id=document.department_id,
            auth_tag=document.auth_tag,
            source_type=source_type,
            embedding_model_id=document.embedding_model_id,
            partition_key=self._partition_key(document),
            page_start=saved_section.page_start,
            page_end=saved_section.page_end,
            section_kind=saved_section.section_kind,
            toc_path=saved_section.toc_path,
            summary_text=masked_summary,
            metadata_json=saved_section.metadata_json,
        )
        self._maybe_index_record(document, record, text=masked_summary, is_urgent=is_urgent)
        return saved_section

    def save_asset(
        self,
        document: Document,
        asset: AssetRecord,
        *,
        summary_text: str,
        is_urgent: bool = True,
    ) -> AssetRecord:
        saved_asset = self.metadata_repo.save_asset(
            asset.model_copy(update={"caption": self._masked_optional_text(asset.caption)})
        )
        masked_summary, summary_contract = self._compile_asset_summary(asset=saved_asset, raw_summary=summary_text)
        saved_asset = self.metadata_repo.save_asset(
            saved_asset.model_copy(
                update={
                    "caption": self._masked_optional_text(saved_asset.caption),
                    "metadata_json": {
                        **saved_asset.metadata_json,
                        "summary_text": masked_summary,
                        "summary_contract": summary_contract,
                    },
                }
            )
        )
        record = AssetSummaryRecord(
            asset_id=saved_asset.asset_id,
            doc_id=saved_asset.doc_id,
            source_id=saved_asset.source_id,
            section_id=saved_asset.section_id,
            version_group_id=document.version_group_id,
            version_no=document.version_no,
            doc_status=document.doc_status,
            effective_date=document.effective_date,
            updated_at=saved_asset.updated_at,
            is_active=document.is_active,
            index_ready=is_urgent,
            tenant_id=document.tenant_id,
            department_id=document.department_id,
            auth_tag=document.auth_tag,
            embedding_model_id=document.embedding_model_id,
            partition_key=self._partition_key(document),
            asset_type=saved_asset.asset_type,
            page_no=saved_asset.page_no,
            caption=self._masked_optional_text(saved_asset.caption),
            summary_text=masked_summary,
            metadata_json=saved_asset.metadata_json,
        )
        self._maybe_index_record(document, record, text=masked_summary, is_urgent=is_urgent)
        return saved_asset

    def search(
        self,
        query_vector: list[float],
        *,
        item_kind: str = "section_summary",
        limit: int = 10,
        doc_ids: list[str] | None = None,
        expr: str | None = None,
    ) -> list[VectorSearchResult]:
        return self.milvus_repo.search(
            query_vector,
            limit=limit,
            doc_ids=doc_ids,
            expr=expr,
            embedding_space=self.embedding_space,
            item_kind=item_kind,
        )

    def deactivate_document(self, doc_id: int) -> Document:
        document = self.metadata_repo.deactivate_document(doc_id)
        self.metadata_repo.set_document_index_state(
            document.doc_id,
            is_indexed=False,
            index_ready=False,
            embedding_model_id=document.embedding_model_id,
            last_index_error=None,
        )
        try:
            self.milvus_repo.delete(expr=f"doc_id in [{doc_id}]")
        except Exception as exc:  # pragma: no cover - defensive logging path
            self._enqueue_sync(
                document=document,
                operation="delete_document",
                error_message=str(exc),
                metadata_json={"doc_id": document.doc_id},
            )
            self.metadata_repo.set_document_index_state(
                document.doc_id,
                is_indexed=False,
                index_ready=False,
                embedding_model_id=document.embedding_model_id,
                last_index_error=str(exc),
            )
            self.logger.error("failed to delete milvus vectors for doc_id=%s: %s", doc_id, exc)
        return document

    def _maybe_index_record(
        self,
        document: Document,
        record: DocSummaryRecord | SectionSummaryRecord | AssetSummaryRecord,
        *,
        text: str,
        is_urgent: bool,
    ) -> None:
        self.metadata_repo.set_document_index_state(
            document.doc_id,
            is_indexed=False,
            index_ready=False,
            embedding_model_id=document.embedding_model_id,
            last_index_error=None,
        )
        if not is_urgent:
            self._enqueue_sync(
                document=document,
                operation="upsert_summary",
                metadata_json={
                    "item_kind": self._record_kind(record),
                    "embedding_space": self.embedding_space,
                    "summary_digest": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                },
            )
            return
        try:
            vector = self._embed_text(text)
            self.milvus_repo.upsert_record(record, vector, embedding_space=self.embedding_space)
        except Exception as exc:
            self._enqueue_sync(
                document=document,
                operation="upsert_summary",
                error_message=str(exc),
                metadata_json={
                    "item_kind": self._record_kind(record),
                    "embedding_space": self.embedding_space,
                    "summary_digest": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                },
            )
            self.metadata_repo.set_document_index_state(
                document.doc_id,
                is_indexed=False,
                index_ready=False,
                embedding_model_id=document.embedding_model_id,
                last_index_error=str(exc),
            )
            raise
        self.metadata_repo.set_document_index_state(
            document.doc_id,
            is_indexed=True,
            index_ready=True,
            embedding_model_id=record.embedding_model_id,
            last_index_error=None,
        )
        self._mark_sync_completed(document.doc_id)

    def _embed_text(self, text: str) -> list[float]:
        embed = getattr(self.embedder, "embed", None)
        if not callable(embed):
            raise RuntimeError("embedding capability is required when is_urgent=True")
        vectors = embed([text])
        if not vectors:
            raise RuntimeError("embedding provider returned no vectors")
        return [float(value) for value in vectors[0]]

    def _masked_optional_text(self, text: str | None) -> str | None:
        masked = self.mask_pii(text)
        return masked or None

    def _enqueue_sync(
        self,
        *,
        document: Document,
        operation: str,
        metadata_json: dict[str, object] | None = None,
        error_message: str | None = None,
    ) -> None:
        if self.index_sync_service is None:
            return
        payload = dict(metadata_json or {})
        payload["commit_anchor"] = self._build_commit_anchor(
            document=document,
            operation=operation,
            metadata_json=payload,
        )
        self.index_sync_service.enqueue(
            doc_id=document.doc_id,
            source_id=document.source_id,
            operation=operation,
            priority=document.index_priority,
            metadata_json=payload,
            error_message=error_message,
        )

    def _mark_sync_completed(self, doc_id: int) -> None:
        if self.index_sync_service is None:
            return
        self.index_sync_service.mark_completed(doc_id)

    def sync_processing_state(self, state: ProcessingStateRecord) -> int:
        self._assert_fresh_commit_anchor(state)
        operation = str(state.metadata_json.get("operation", "") or "").strip().lower()
        if operation == "delete_document":
            self.milvus_repo.delete(expr=f"doc_id in [{state.doc_id}]")
            return 0
        if operation == "upsert_summary":
            embedding_space = str(state.metadata_json.get("embedding_space", "") or self.embedding_space)
            return self.sync_document_summaries(state.doc_id, embedding_space=embedding_space)
        raise RuntimeError(f"unsupported index sync operation: {operation or '<empty>'}")

    def sync_document_summaries(self, doc_id: int, *, embedding_space: str | None = None) -> int:
        get_document = getattr(self.metadata_repo, "get_document", None)
        get_source = getattr(self.metadata_repo, "get_source", None)
        list_sections = getattr(self.metadata_repo, "list_sections", None)
        list_assets = getattr(self.metadata_repo, "list_assets", None)
        if not callable(get_document) or not callable(get_source):
            raise RuntimeError("metadata repository is missing document/source lookup capabilities")
        if not callable(list_sections) or not callable(list_assets):
            raise RuntimeError("metadata repository is missing section/asset listing capabilities")

        document = get_document(doc_id)
        if document is None:
            self.milvus_repo.delete(expr=f"doc_id in [{doc_id}]")
            return 0

        if not self._document_is_indexable(document):
            self.milvus_repo.delete(expr=f"doc_id in [{doc_id}]")
            self.metadata_repo.set_document_index_state(
                document.doc_id,
                is_indexed=False,
                index_ready=False,
                embedding_model_id=document.embedding_model_id,
                last_index_error=None,
            )
            return 0

        source = get_source(document.source_id)
        source_type = None if source is None else source.source_type
        records = self._summary_records_for_document(
            document=document,
            source_type=source_type,
            sections=list_sections(doc_id=document.doc_id),
            assets=list_assets(doc_id=document.doc_id),
        )
        if not records:
            raise RuntimeError(f"no summary records available for doc_id={doc_id}")

        vectors = self._embed_many([record.summary_text for record in records])
        target_space = embedding_space or self.embedding_space
        self.milvus_repo.delete(expr=f"doc_id in [{doc_id}]")
        self.milvus_repo.upsert_records(list(zip(records, vectors, strict=True)), embedding_space=target_space)
        self.metadata_repo.set_document_index_state(
            document.doc_id,
            is_indexed=True,
            index_ready=True,
            embedding_model_id=document.embedding_model_id,
            last_index_error=None,
        )
        return len(records)

    @staticmethod
    def _record_kind(record: DocSummaryRecord | SectionSummaryRecord | AssetSummaryRecord) -> str:
        if isinstance(record, DocSummaryRecord):
            return "doc_summary"
        if isinstance(record, AssetSummaryRecord):
            return "asset_summary"
        return "section_summary"

    def _embed_many(self, texts: list[str]) -> list[list[float]]:
        embed = getattr(self.embedder, "embed", None)
        if not callable(embed):
            raise RuntimeError("embedding capability is required for index sync")
        vectors = list(embed(texts))
        if len(vectors) != len(texts):
            raise RuntimeError("embedding provider returned an unexpected number of vectors")
        return [[float(value) for value in vector] for vector in vectors]

    def _summary_records_for_document(
        self,
        *,
        document: Document,
        source_type: SourceType | None,
        sections: list[SectionRecord],
        assets: list[AssetRecord],
    ) -> list[DocSummaryRecord | SectionSummaryRecord | AssetSummaryRecord]:
        records: list[DocSummaryRecord | SectionSummaryRecord | AssetSummaryRecord] = []
        if summary_text := self._doc_summary_text(document=document, source_type=source_type):
            records.append(
                DocSummaryRecord(
                    doc_id=document.doc_id,
                    source_id=document.source_id,
                    version_group_id=document.version_group_id,
                    version_no=document.version_no,
                    doc_status=document.doc_status,
                    effective_date=document.effective_date,
                    updated_at=document.updated_at,
                    is_active=document.is_active,
                    index_ready=True,
                    tenant_id=document.tenant_id,
                    department_id=document.department_id,
                    auth_tag=document.auth_tag,
                    source_type=source_type,
                    embedding_model_id=document.embedding_model_id,
                    partition_key=self._partition_key(document),
                    title=self._masked_optional_text(document.title),
                    summary_text=summary_text,
                    metadata_json=document.metadata_json,
                )
            )
        for section in sections:
            if not (summary_text := self._section_summary_text(section)):
                continue
            records.append(
                SectionSummaryRecord(
                    section_id=section.section_id,
                    doc_id=section.doc_id,
                    source_id=section.source_id,
                    version_group_id=document.version_group_id,
                    version_no=document.version_no,
                    doc_status=document.doc_status,
                    effective_date=document.effective_date,
                    updated_at=section.updated_at,
                    is_active=document.is_active,
                    index_ready=True,
                    tenant_id=document.tenant_id,
                    department_id=document.department_id,
                    auth_tag=document.auth_tag,
                    source_type=source_type,
                    embedding_model_id=document.embedding_model_id,
                    partition_key=self._partition_key(document),
                    page_start=section.page_start,
                    page_end=section.page_end,
                    section_kind=section.section_kind,
                    toc_path=section.toc_path,
                    summary_text=summary_text,
                    metadata_json=section.metadata_json,
                )
            )
        for asset in assets:
            if not (summary_text := self._asset_summary_text(asset)):
                continue
            records.append(
                AssetSummaryRecord(
                    asset_id=asset.asset_id,
                    doc_id=asset.doc_id,
                    source_id=asset.source_id,
                    section_id=asset.section_id,
                    version_group_id=document.version_group_id,
                    version_no=document.version_no,
                    doc_status=document.doc_status,
                    effective_date=document.effective_date,
                    updated_at=asset.updated_at,
                    is_active=document.is_active,
                    index_ready=True,
                    tenant_id=document.tenant_id,
                    department_id=document.department_id,
                    auth_tag=document.auth_tag,
                    embedding_model_id=document.embedding_model_id,
                    partition_key=self._partition_key(document),
                    asset_type=asset.asset_type,
                    page_no=asset.page_no,
                    caption=self._masked_optional_text(asset.caption),
                    summary_text=summary_text,
                    metadata_json=asset.metadata_json,
                )
            )
        return records

    def _doc_summary_text(self, *, document: Document, source_type: SourceType | None) -> str | None:
        raw_summary = self._raw_summary_text(document.metadata_json)
        if raw_summary is None:
            return None
        normalized, _contract = self._compile_doc_summary(
            document=document,
            source_type=source_type,
            raw_summary=raw_summary,
        )
        return normalized

    def _section_summary_text(self, section: SectionRecord) -> str | None:
        raw_summary = self._raw_summary_text(section.metadata_json)
        if raw_summary is None:
            return None
        normalized, _contract = self._compile_section_summary(section=section, raw_summary=raw_summary)
        return normalized

    def _asset_summary_text(self, asset: AssetRecord) -> str | None:
        raw_summary = self._raw_summary_text(asset.metadata_json)
        if raw_summary is None:
            return None
        normalized, _contract = self._compile_asset_summary(asset=asset, raw_summary=raw_summary)
        return normalized

    @staticmethod
    def _partition_key(document: Document) -> PartitionKey:
        return PartitionKey.COLD if str(document.storage_tier).lower() == "cold" else PartitionKey.HOT

    @staticmethod
    def _raw_summary_text(metadata_json: dict[str, Any]) -> str | None:
        summary_text = metadata_json.get("summary_text")
        if not isinstance(summary_text, str):
            return None
        normalized = summary_text.strip()
        return normalized or None

    def _compile_doc_summary(
        self,
        *,
        document: Document,
        source_type: SourceType | None,
        raw_summary: str,
    ) -> tuple[str, dict[str, Any]]:
        return self._compile_summary_contract(
            kind="document",
            raw_summary=raw_summary,
            semantic_hint=document.title,
            fact_anchors=[
                f"doc_id={document.doc_id}",
                f"version_group={document.version_group_id}",
                "source_type="
                f"{source_type.value if isinstance(source_type, SourceType) else (source_type or 'unknown')}",
            ],
            structural_hints=[
                f"doc_status={document.doc_status}",
                f"title={self._masked_optional_text(document.title) or 'untitled'}",
            ],
        )

    def _compile_section_summary(
        self,
        *,
        section: SectionRecord,
        raw_summary: str,
    ) -> tuple[str, dict[str, Any]]:
        toc_text = " / ".join(section.toc_path)
        page_hint = (
            f"pages={section.page_start}-{section.page_end}"
            if section.page_start is not None and section.page_end is not None
            else "pages=unknown"
        )
        return self._compile_summary_contract(
            kind="section",
            raw_summary=raw_summary,
            semantic_hint=toc_text or section.anchor,
            fact_anchors=[
                f"section_id={section.section_id}",
                page_hint,
                f"anchor={section.anchor or 'none'}",
            ],
            structural_hints=[
                f"toc={toc_text or 'root'}",
                f"section_kind={section.section_kind}",
            ],
        )

    def _compile_asset_summary(
        self,
        *,
        asset: AssetRecord,
        raw_summary: str,
    ) -> tuple[str, dict[str, Any]]:
        return self._compile_summary_contract(
            kind="asset",
            raw_summary=raw_summary,
            semantic_hint=asset.caption,
            fact_anchors=[
                f"asset_id={asset.asset_id}",
                f"page={asset.page_no}",
                f"asset_type={asset.asset_type}",
            ],
            structural_hints=[
                f"section_id={asset.section_id or 'none'}",
                f"element_ref={asset.element_ref or 'none'}",
            ],
        )

    def _compile_summary_contract(
        self,
        *,
        kind: str,
        raw_summary: str,
        semantic_hint: str | None,
        fact_anchors: list[str],
        structural_hints: list[str],
    ) -> tuple[str, dict[str, Any]]:
        parsed = self._parse_summary_contract(raw_summary, expected_kind=kind)
        if parsed is not None:
            return parsed
        normalized = self.mask_pii(raw_summary).strip()
        if not normalized:
            normalized = self.mask_pii(semantic_hint).strip()
        semantic_core = self._semantic_core_text(normalized or kind)
        normalized_fact_anchors = [part for part in fact_anchors if part and part.strip()]
        normalized_structural_hints = [part for part in structural_hints if part and part.strip()]
        contract = {
            "spec_version": _SUMMARY_SPEC_VERSION,
            "kind": kind,
            "semantic_core": semantic_core,
            "fact_anchors": normalized_fact_anchors,
            "structural_hint": normalized_structural_hints,
        }
        compiled = "\n".join(
            [
                f"Semantic Core: {semantic_core}",
                f"Fact Anchors: {'; '.join(normalized_fact_anchors) if normalized_fact_anchors else 'none'}",
                f"Structural Hint: {'; '.join(normalized_structural_hints) if normalized_structural_hints else 'none'}",
            ]
        )
        return compiled, contract

    @staticmethod
    def _semantic_core_text(text: str) -> str:
        first_line = re.split(r"[\n\r]+", text.strip(), maxsplit=1)[0]
        collapsed = re.sub(r"\s+", " ", first_line).strip()
        return collapsed[:480] if len(collapsed) > 480 else collapsed

    def _parse_summary_contract(self, raw_summary: str, *, expected_kind: str) -> tuple[str, dict[str, Any]] | None:
        lines = [line.strip() for line in self.mask_pii(raw_summary).splitlines() if line.strip()]
        if len(lines) < 3:
            return None
        if not lines[0].startswith("Semantic Core: ") or not lines[1].startswith("Fact Anchors: "):
            return None
        if not lines[2].startswith("Structural Hint: "):
            return None
        semantic_core = lines[0].split(": ", 1)[1].strip()
        fact_anchors = [
            part.strip()
            for part in lines[1].split(": ", 1)[1].split(";")
            if part.strip() and part.strip() != "none"
        ]
        structural_hint = [
            part.strip()
            for part in lines[2].split(": ", 1)[1].split(";")
            if part.strip() and part.strip() != "none"
        ]
        contract = {
            "spec_version": _SUMMARY_SPEC_VERSION,
            "kind": expected_kind,
            "semantic_core": semantic_core,
            "fact_anchors": fact_anchors,
            "structural_hint": structural_hint,
        }
        compiled = "\n".join(lines[:3])
        return compiled, contract

    def _build_commit_anchor(
        self,
        *,
        document: Document,
        operation: str,
        metadata_json: dict[str, object],
    ) -> str:
        payload = {
            "doc_id": document.doc_id,
            "source_id": document.source_id,
            "file_hash": document.file_hash,
            "version_group_id": document.version_group_id,
            "updated_at": document.updated_at.isoformat(),
            "operation": operation,
            "metadata_json": metadata_json,
        }
        serialized = json.dumps(
            payload,
            sort_keys=True,
            ensure_ascii=False,
            default=str,
        ).encode("utf-8")
        return hashlib.sha256(serialized).hexdigest()

    def _assert_fresh_commit_anchor(self, state: ProcessingStateRecord) -> None:
        get_processing_state = getattr(self.metadata_repo, "get_processing_state", None)
        if not callable(get_processing_state):
            return
        claimed_anchor = str(state.metadata_json.get("commit_anchor", "") or "").strip()
        if not claimed_anchor:
            return
        current_state = get_processing_state(state.doc_id)
        if current_state is None:
            return
        current_anchor = str(current_state.metadata_json.get("commit_anchor", "") or "").strip()
        if current_anchor and current_anchor != claimed_anchor:
            raise StaleProcessingStateError(
                f"stale commit anchor for doc_id={state.doc_id}: claimed={claimed_anchor} current={current_anchor}"
            )

    @staticmethod
    def _document_is_indexable(document: Document) -> bool:
        if not document.is_active:
            return False
        status = str(document.doc_status or "").strip().lower()
        return status not in {"retired", "expired", "deleted", "inactive"}


__all__ = ["DataContractService", "DocumentRegistrationResult"]

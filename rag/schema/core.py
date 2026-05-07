from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, model_validator

from rag.schema.runtime import AccessPolicy


class SourceType(StrEnum):
    """数据的物理载体格式 (指挥 Parser 解析器)"""
    PDF = "pdf"
    MARKDOWN = "markdown"
    DOCX = "docx"
    PPTX = "pptx"
    XLSX = "xlsx"
    IMAGE = "image"
    WEB = "web"
    PLAIN_TEXT = "plain_text"
    PASTED_TEXT = "pasted_text"
    BROWSER_CLIP = "browser_clip"


class DocumentType(StrEnum):
    """知识的业务体裁 (指挥 L3 意图过滤与大模型 Prompt)"""
    POLICY = "policy"           # 规章制度 / 规范
    REPORT = "report"           # 财务报告 / 研报
    MANUAL = "manual"           # 操作手册 / 指南
    CONTRACT = "contract"       # 合同 / 协议
    ARTICLE = "article"         # 新闻 / 软文
    NOTE = "note"               # 会议记录 / 笔记
    UNKNOWN = "unknown"         # 未知/兜底体裁


class DocumentStatus(StrEnum):
    DRAFT = "draft"
    PUBLISHED = "published"
    RETIRED = "retired"


class PiiStatus(StrEnum):
    UNKNOWN = "unknown"
    CLEAN = "clean"
    MASKED = "masked"
    RESTRICTED = "restricted"


class IndexingMode(StrEnum):
    EAGER = "eager"
    LAZY = "lazy"


class StorageTier(StrEnum):
    HOT = "hot"
    COLD = "cold"


class PartitionKey(StrEnum):
    HOT = "hot"
    COLD = "cold"

class AssetRelationType(StrEnum):
    CAPTION_OF = "caption_of"
    TABLE_OF = "table_of"
    FIGURE_OF = "figure_of"
    REFERENCE_BY = "reference_by"

# ==========================================
# L0: 元数据与来源定义
# ==========================================

class Source(BaseModel):
    model_config = ConfigDict(frozen=True)

    source_id: int = 0
    source_type: SourceType
    location: str
    original_file_name: str | None = None
    bucket: str | None = None
    object_key: str | None = None
    content_hash: str
    file_size_bytes: int | None = None
    mime_type: str | None = None
    owner_id: str | None = None
    ingest_version: int = 1
    pii_status: PiiStatus = PiiStatus.UNKNOWN
    effective_access_policy: AccessPolicy = Field(default_factory=AccessPolicy.default)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metadata_json: dict[str, Any] = Field(default_factory=dict)


class Document(BaseModel):
    model_config = ConfigDict(frozen=True)

    doc_id: int = 0
    source_id: int
    title: str | None = None
    doc_type: DocumentType
    language: str | None = None
    authors: list[str] = Field(default_factory=list)
    file_hash: str
    version_group_id: int = 0
    version_no: int = 1
    doc_status: DocumentStatus | str = DocumentStatus.PUBLISHED
    effective_date: datetime | None = None
    is_active: bool = True
    is_indexed: bool = False
    index_ready: bool = False
    index_priority: str = "high"
    indexing_mode: IndexingMode = IndexingMode.EAGER
    storage_tier: StorageTier = StorageTier.HOT
    pii_status: PiiStatus = PiiStatus.UNKNOWN
    reference_count: int = 1
    page_count: int | None = None
    tenant_id: str | None = None
    department_id: str | None = None
    auth_tag: str | None = None
    embedding_model_id: str = "default"
    indexed_at: datetime | None = None
    last_index_error: str | None = None
    effective_access_policy: AccessPolicy = Field(default_factory=AccessPolicy.default)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metadata_json: dict[str, Any] = Field(default_factory=dict)


# ==========================================
# 解析器 (Parser) 输出契约
# ==========================================

class DocumentFeatures(BaseModel):
    model_config = ConfigDict(frozen=True)

    source_type: SourceType
    section_count: int
    word_count: int
    heading_count: int
    table_count: int
    figure_count: int
    caption_count: int
    ocr_region_count: int
    structure_depth: int
    has_dense_structure: bool
    metadata: dict[str, str] = Field(default_factory=dict)


@dataclass(frozen=True)
class ParsedSection:
    toc_path: tuple[str, ...]
    heading_level: int | None
    page_range: tuple[int, int] | None
    order_index: int
    text: str
    char_range_start: int 
    char_range_end: int
    anchor_hint: str | None = None
    metadata: dict[str, str] = field(default_factory=dict)
    
    
    @model_validator(mode="after")
    def validate_char_range(self):
        start = self.char_range_start
        end = self.char_range_end
        if start is None and end is None:
            return self
        if start is None or end is None:
            raise ValueError("char_range_start and char_range_end must both be set")
        if start < 0:
            raise ValueError("char_range_start must be >= 0")
        if end <= start:
            raise ValueError("char_range_end must be > char_range_start")
        return self

@dataclass(frozen=True)
class ParsedElement:
    element_id: str
    kind: str
    text: str
    toc_path: tuple[str, ...] = ()
    heading_level: int | None = None
    page_no: int | None = None
    bbox: tuple[float, float, float, float] | None = None
    parent_ref: str | None = None
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class ParsedDocument:
    title: str
    source_type: SourceType
    doc_type: DocumentType
    authors: list[str]
    language: str
    sections: list[ParsedSection]
    visible_text: str
    visual_semantics: str | None = None
    elements: list[ParsedElement] = field(default_factory=list)
    page_count: int | None = None
    doc_model: Any | None = None
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class OcrRegion:
    text: str
    bbox: tuple[int, int, int, int] | None = None


@dataclass(frozen=True)
class OcrResult:
    visible_text: str
    visual_semantics: str
    regions: list[OcrRegion] = field(default_factory=list)


# ==========================================
# L1: 物理拆解与原文记录 (PostgreSQL 存储)
# ==========================================
class SectionLocatorRecord(BaseModel):
    """
    Section 正式物理定位锚点。

    这版设计是“强约束版”：
    - visible_text_key 必填
    - char range 必填
    - byte range 必填
    - 不允许半套 locator
    - 不允许非法区间
    """

    model_config = ConfigDict(frozen=True)

    locator_version: Literal["v1"] = "v1"
    text_basis: Literal["visible_text"] = "visible_text"
    text_encoding: Literal["utf-8"] = "utf-8"

    # object store 中“规范化 visible_text”对象的 key
    visible_text_key: str

    # 业务主锚点：精确逻辑范围
    char_range_start: int = Field(ge=0)
    char_range_end: int = Field(gt=0)

    # 存储主锚点：高效物理范围
    byte_range_start: int = Field(ge=0)
    byte_range_end: int = Field(gt=0)

    def model_post_init(self, __context: object) -> None:
        if not self.visible_text_key.strip():
            raise ValueError("visible_text_key must not be empty")

        if self.char_range_end <= self.char_range_start:
            raise ValueError(
                "char_range_end must be greater than char_range_start"
            )

        if self.byte_range_end <= self.byte_range_start:
            raise ValueError(
                "byte_range_end must be greater than byte_range_start"
            )


class SectionRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    section_id: int = 0
    doc_id: int
    source_id: int

    parent_section_id: int | None = None
    toc_path: list[str] = Field(default_factory=list)
    heading_level: int | None = None
    order_index: int
    anchor: str | None = None

    page_start: int | None = None
    page_end: int | None = None

    # 原始文件 / 大块内容对象
    content_storage_key: str | None = None

    # 规范化 visible_text 对象
    visible_text_key: str

    # 正式主真相：必须存在
    raw_locator: SectionLocatorRecord

    # 冗余字段：便于 SQL 过滤 / 调试 / 审计
    char_range_start: int = Field(ge=0)
    char_range_end: int = Field(gt=0)
    byte_range_start: int = Field(ge=0)
    byte_range_end: int = Field(gt=0)

    section_kind: str
    content_hash: str

    has_table: bool = False
    has_figure: bool = False
    neighbor_asset_count: int = 0

    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    metadata_json: dict[str, Any] = Field(default_factory=dict)

    def model_post_init(self, __context: object) -> None:
        if not self.visible_text_key.strip():
            raise ValueError("visible_text_key must not be empty")

        if self.page_start is not None and self.page_end is not None:
            if self.page_end < self.page_start:
                raise ValueError(
                    "page_end must be greater than or equal to page_start"
                )

        if self.char_range_end <= self.char_range_start:
            raise ValueError(
                "char_range_end must be greater than char_range_start"
            )

        if self.byte_range_end <= self.byte_range_start:
            raise ValueError(
                "byte_range_end must be greater than byte_range_start"
            )

        if self.raw_locator.visible_text_key != self.visible_text_key:
            raise ValueError(
                "raw_locator.visible_text_key must match SectionRecord.visible_text_key"
            )

        if self.raw_locator.char_range_start != self.char_range_start:
            raise ValueError(
                "raw_locator.char_range_start must match SectionRecord.char_range_start"
            )

        if self.raw_locator.char_range_end != self.char_range_end:
            raise ValueError(
                "raw_locator.char_range_end must match SectionRecord.char_range_end"
            )

        if self.raw_locator.byte_range_start != self.byte_range_start:
            raise ValueError(
                "raw_locator.byte_range_start must match SectionRecord.byte_range_start"
            )

        if self.raw_locator.byte_range_end != self.byte_range_end:
            raise ValueError(
                "raw_locator.byte_range_end must match SectionRecord.byte_range_end"
            )

    def can_precisely_recall_visible_text(self) -> bool:
        return True

class AssetRecord(BaseModel):
    model_config = ConfigDict(frozen=True, populate_by_name=True, serialize_by_alias=True)

    asset_id: int = 0
    doc_id: int
    source_id: int
    section_id: int | None = None
    relation_type: AssetRelationType | None = None

    asset_type: str
    element_ref: str | None = None
    page_no: int
    bbox: dict[str, Any] = Field(default_factory=dict)
    caption: str | None = None
    raw_locator: dict[str, Any] = Field(default_factory=dict)
    neighbor_section_id: int | None = None
    sheet_name: str | None = None
    row_count: int | None = None
    column_count: int | None = None
    sample_rows: list[dict[str, Any]] = Field(default_factory=list)
    table_schema: list[dict[str, Any]] = Field(
        default_factory=list,
        validation_alias=AliasChoices("schema", "table_schema"),
        serialization_alias="schema",
    )
    content_hash: str
    storage_key: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metadata_json: dict[str, Any] = Field(default_factory=dict)

    @property
    def schema(self) -> list[dict[str, Any]]:
        return self.table_schema


# ==========================================
# L2: 索引与摘要 (Milvus 存储)
# ==========================================



class DocSummaryRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    doc_id: int
    source_id: int
    version_group_id: int
    version_no: int = 1
    doc_status: DocumentStatus | str = DocumentStatus.PUBLISHED
    effective_date: datetime | None = None
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    is_active: bool = True
    index_ready: bool = True
    tenant_id: str | None = None
    department_id: str | None = None
    auth_tag: str | None = None
    source_type: SourceType | None = None
    embedding_model_id: str = "default"
    partition_key: PartitionKey = PartitionKey.HOT
    title: str | None = None
    summary_text: str
    metadata_json: dict[str, Any] = Field(default_factory=dict)


class SectionSummaryRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    section_id: int
    doc_id: int
    source_id: int
    version_group_id: int
    version_no: int = 1
    doc_status: DocumentStatus | str = DocumentStatus.PUBLISHED
    effective_date: datetime | None = None
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    is_active: bool = True
    index_ready: bool = True
    tenant_id: str | None = None
    department_id: str | None = None
    auth_tag: str | None = None
    source_type: SourceType | None = None
    embedding_model_id: str = "default"
    partition_key: PartitionKey = PartitionKey.HOT
    page_start: int | None = None
    page_end: int | None = None
    section_kind: str
    toc_path: list[str] = Field(default_factory=list)
    summary_text: str
    metadata_json: dict[str, Any] = Field(default_factory=dict)


class AssetSummaryRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    asset_id: int
    doc_id: int
    source_id: int
    section_id: int | None = None
    version_group_id: int
    version_no: int = 1
    doc_status: DocumentStatus | str = DocumentStatus.PUBLISHED
    effective_date: datetime | None = None
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    is_active: bool = True
    index_ready: bool = True
    tenant_id: str | None = None
    department_id: str | None = None
    auth_tag: str | None = None
    embedding_model_id: str = "default"
    partition_key: PartitionKey = PartitionKey.HOT
    asset_type: str
    page_no: int | None = None
    caption: str | None = None
    summary_text: str
    metadata_json: dict[str, Any] = Field(default_factory=dict)

# ==========================================
# 基础设施: 缓存与状态监控
# ==========================================

class LayoutMetaCacheRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    cache_id: int = 0
    source_id: int
    doc_id: int | None = None
    content_hash: str
    object_key: str | None = None
    layout_json: dict[str, Any] = Field(default_factory=dict)
    layout_version: str = "v1"
    page_count: int | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ProcessingStateRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    doc_id: int
    source_id: int
    stage: str
    status: str
    attempts: int = 0
    priority: str = "normal"
    worker_id: str | None = None
    lease_expires_at: datetime | None = None
    error_message: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metadata_json: dict[str, Any] = Field(default_factory=dict)


# ==========================================
# Ingest 阶段输出: 终极包裹
# ==========================================

class DocumentProcessingPackage(BaseModel):
    """
    更新后的包裹：完全基于 L1 (Record) 和 L2 (Summary) 的装载舱。
    Ingest Pipeline 跑完后，将产生这个包裹，并交由 Storage 写入数据库。
    """
    model_config = ConfigDict(frozen=True)

    source: Source
    document: Document
    analysis: DocumentFeatures
    
    # L1 物理数据
    sections: list[SectionRecord]
    assets: list[AssetRecord]
    
    # L2 索引数据
    doc_summary: DocSummaryRecord | None = None
    section_summaries: list[SectionSummaryRecord] = Field(default_factory=list)
    asset_summaries: list[AssetSummaryRecord] = Field(default_factory=list)
    
    # Infra 缓存与状态 (补齐防断链标志)
    layout_cache: LayoutMetaCacheRecord | None = None
    processing_state: ProcessingStateRecord | None = None
    metadata_summary: dict[str, Any] = Field(default_factory=dict)


__all__ = [
    "Document",
    "DocumentFeatures",
    "DocumentStatus",
    "DocumentProcessingPackage",
    "DocumentType",
    "IndexingMode",
    "OcrRegion",
    "OcrResult",
    "ParsedDocument",
    "ParsedElement",
    "ParsedSection",
    "PartitionKey",
    "PiiStatus",
    "Source",
    "SourceType",
    "StorageTier",

    "SectionRecord",
    "AssetRecord",
    "DocSummaryRecord",
    "SectionSummaryRecord",
    "AssetSummaryRecord",
    "LayoutMetaCacheRecord",
    "ProcessingStateRecord",
]

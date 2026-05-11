from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class TaskType(StrEnum):
    LOOKUP = "lookup"
    SINGLE_DOC_QA = "single_doc_qa"
    COMPARISON = "comparison"
    SYNTHESIS = "synthesis"
    TIMELINE = "timeline"
    RESEARCH = "research"




class PageRangeConstraint(BaseModel):
    model_config = ConfigDict(frozen=True)

    start: int
    end: int


class StructureConstraints(BaseModel):
    model_config = ConfigDict(frozen=True)

    match_strategy: str = "none"
    requires_structure_match: bool = False
    prefer_heading_match: bool = False
    focus_terms: list[str] = Field(default_factory=list)

    def has_constraints(self) -> bool:
        return any(
            (
                self.match_strategy != "none",
                self.requires_structure_match,
                self.prefer_heading_match,
                bool(self.focus_terms),
            )
        )


class MetadataFilters(BaseModel):
    model_config = ConfigDict(frozen=True)

    page_numbers: list[int] = Field(default_factory=list)
    page_ranges: list[PageRangeConstraint] = Field(default_factory=list)
    source_types: list[str] = Field(default_factory=list)
    document_titles: list[str] = Field(default_factory=list)
    file_names: list[str] = Field(default_factory=list)

    def has_constraints(self) -> bool:
        return any(
            (
                bool(self.page_numbers),
                bool(self.page_ranges),
                bool(self.source_types),
                bool(self.document_titles),
                bool(self.file_names),
            )
        )


class PolicyHints(BaseModel):
    model_config = ConfigDict(frozen=True)

    disable_external_retrieval: bool = False
    local_only: bool = False
    source_type_scope: list[str] = Field(default_factory=list)

    def has_hints(self) -> bool:
        return any((self.disable_external_retrieval, self.local_only, bool(self.source_type_scope)))


class RetrievalSignals(BaseModel):
    """RAG 检索层消费的结构化信号，由 Agent 或上层调用方产出。"""
    model_config = ConfigDict(frozen=True)

    metadata_filters: MetadataFilters = Field(default_factory=MetadataFilters)
    structure_constraints: StructureConstraints = Field(default_factory=StructureConstraints)
    special_targets: list[str] = Field(default_factory=list)
    quoted_terms: list[str] = Field(default_factory=list)
    allow_graph_expansion: bool = False

    def has_constraints(self) -> bool:
        return any((
            self.metadata_filters.has_constraints(),
            self.structure_constraints.has_constraints(),
            bool(self.special_targets),
            bool(self.quoted_terms),
            self.allow_graph_expansion,
        ))

    @staticmethod
    def from_query_understanding(qu: QueryUnderstanding) -> RetrievalSignals:
        return RetrievalSignals(
            metadata_filters=qu.metadata_filters,
            structure_constraints=qu.structure_constraints,
            special_targets=qu.special_targets,
            quoted_terms=qu.quoted_terms,
            allow_graph_expansion=qu.needs_graph_expansion,
        )


class QueryUnderstanding(BaseModel):
    model_config = ConfigDict(frozen=True)

    task_type: TaskType = TaskType.LOOKUP

    needs_special: bool = False
    needs_structure: bool = False
    needs_metadata: bool = False
    needs_graph_expansion: bool = False

    structure_constraints: StructureConstraints = Field(default_factory=StructureConstraints)
    metadata_filters: MetadataFilters = Field(default_factory=MetadataFilters)

    special_targets: list[str] = Field(default_factory=list)
    source_scope_hints: list[str] = Field(default_factory=list)
    quoted_terms: list[str] = Field(default_factory=list)
    policy_hints: PolicyHints = Field(default_factory=PolicyHints)

    def has_explicit_constraints(self) -> bool:
        return any(
            (
                self.needs_special,
                self.needs_structure,
                self.needs_metadata,
                self.needs_graph_expansion,
                self.structure_constraints.has_constraints(),
                self.metadata_filters.has_constraints(),
                bool(self.special_targets),
                bool(self.source_scope_hints),
                bool(self.quoted_terms),
                self.policy_hints.has_hints(),
            )
        )


class AnswerCitation(BaseModel):
    model_config = ConfigDict(frozen=True)

    citation_id: str
    file_name: str | None = None
    section_path: list[str] = Field(default_factory=list)
    page_start: int | None = None
    page_end: int | None = None
    
    evidence_id: str
    record_type: str
    citation_anchor: str | None = None
    
    doc_id: int | None = None
    benchmark_doc_id: str | None = None
    source_id: int | None = None
    source_type: str | None = None


class AnswerEvidenceLink(BaseModel):
    model_config = ConfigDict(frozen=True)

    link_id: str
    answer_section_id: str
    answer_excerpt: str
    evidence_id: str
    citation_id: str | None = None
    support_score: float = Field(default=0.0, ge=0.0, le=1.0)


class AnswerSection(BaseModel):
    model_config = ConfigDict(frozen=True)

    section_id: str
    title: str
    text: str
    citation_ids: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)


class GroundedAnswer(BaseModel):
    model_config = ConfigDict(frozen=True)

    answer_text: str
    answer_sections: list[AnswerSection] = Field(default_factory=list)
    citations: list[AnswerCitation] = Field(default_factory=list)
    evidence_links: list[AnswerEvidenceLink] = Field(default_factory=list)
    groundedness_flag: bool
    insufficient_evidence_flag: bool


class GroundingTarget(BaseModel):
    model_config = ConfigDict(frozen=True)

    kind: str
    doc_id: int
    source_id: int | None = None
    section_id: int | None = None
    asset_id: int | None = None
    page_start: int | None = None
    page_end: int | None = None
    section_path: list[str] = Field(default_factory=list)
    raw_locator: dict[str, str] = Field(default_factory=dict)


class EvidenceItem(BaseModel):
    model_config = ConfigDict(frozen=True)

    evidence_id: str
    doc_id: int
    benchmark_doc_id: str | None = None
    source_id: int | None = None
    
    citation_anchor: str
    text: str
    score: float
    evidence_kind: str = "internal"
    
    record_type: str | None = None
    
    file_name: str | None = None
    section_path: list[str] = Field(default_factory=list)
    page_start: int | None = None
    page_end: int | None = None
    source_type: str | None = None
    retrieval_channels: list[str] = Field(default_factory=list)
    retrieval_family: str | None = None
    grounding_target: GroundingTarget | None = None


class ArtifactStatus(StrEnum):
    SUGGESTED = "suggested"
    APPROVED = "approved"
    STALE = "stale"
    CONFLICTED = "conflicted"
    ARCHIVED = "archived"


class KnowledgeArtifact(BaseModel):
    model_config = ConfigDict(frozen=True)

    artifact_id: str
    # 🚨 解除 UI 强绑定，降级为普通字符串
    artifact_type: str
    title: str
    supported_evidence_ids: list[str]
    confidence: float | None = None
    status: ArtifactStatus
    last_reviewed_at: datetime
    body_markdown: str
    source_scope: list[str] = Field(default_factory=list)
    metadata: dict[str, str] = Field(default_factory=dict)


def _rebuild_runtime_schema_refs() -> None:
    import rag.schema.runtime as _runtime_schema

    _runtime_schema.RetrievalDiagnostics.model_rebuild(
        _types_namespace={
            "QueryUnderstanding": QueryUnderstanding,
            "RetrievalSignals": RetrievalSignals,
        }
    )


_rebuild_runtime_schema_refs()
EvidenceItem.model_rebuild(_types_namespace={"GroundingTarget": GroundingTarget})

__all__ = [
    "AnswerCitation",
    "AnswerEvidenceLink",
    "AnswerSection",
    "ArtifactStatus",
    "EvidenceItem",
    "GroundedAnswer",
    "GroundingTarget",
    "KnowledgeArtifact",
    "MetadataFilters",
    "PageRangeConstraint",
    "PolicyHints",
    "QueryUnderstanding",
    "RetrievalSignals",
    "StructureConstraints",
    "TaskType",
]

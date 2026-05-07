from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict, Field

from rag.schema.query import EvidenceItem, GroundedAnswer, GroundingTarget
from rag.schema.runtime import AccessPolicy, ExecutionLocationPreference, ProviderAttempt, RetrievalDiagnostics

if TYPE_CHECKING:
    from rag.retrieval.analysis import RoutingDecision
    from rag.retrieval.evidence import EvidenceBundle, SelfCheckResult


class RetrievalProfile(StrEnum):
    BYPASS = "bypass"
    FAST = "fast"
    AUTO = "auto"
    DEEP = "deep"
    ASSET = "asset"


def normalize_retrieval_profile(profile: RetrievalProfile | str | None) -> RetrievalProfile:
    if profile is None:
        return RetrievalProfile.AUTO
    if isinstance(profile, RetrievalProfile):
        return profile
    return RetrievalProfile(profile)


@dataclass(frozen=True, slots=True)
class QueryOptions:
    retrieval_profile: Literal["bypass", "fast", "auto", "deep", "asset"] = "auto"
    user_id: str | None = None
    source_scope: tuple[str, ...] = ()
    access_policy: AccessPolicy = field(default_factory=AccessPolicy.default)
    execution_location_preference: ExecutionLocationPreference = ExecutionLocationPreference.LOCAL_FIRST
    max_context_tokens: int = 12000
    max_evidence_items: int | None = None
    evidence_top_k: int | None = None
    answer_context_top_k: int | None = None
    top_k: int = 8
    response_type: str = "Multiple Paragraphs"
    user_prompt: str | None = None
    conversation_history: tuple[tuple[str, str], ...] = ()
    enable_rerank: bool = True
    retrieval_pool_k: int | None = None
    rerank_pool_k: int | None = None

    @property
    def resolved_retrieval_profile(self) -> RetrievalProfile:
        return normalize_retrieval_profile(self.retrieval_profile)

    @property
    def resolved_max_evidence_items(self) -> int:
        return max(self.max_evidence_items if self.max_evidence_items is not None else self.top_k, 1)

    @property
    def resolved_candidate_top_k(self) -> int:
        candidate_top_k = self.evidence_top_k if self.evidence_top_k is not None else self.top_k
        return max(candidate_top_k, 1)


class ContextEvidence(BaseModel):
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
    section_path: list[str] = Field(default_factory=list)
    file_name: str | None = None
    page_start: int | None = None
    page_end: int | None = None
    source_type: str | None = None
    retrieval_channels: list[str] = Field(default_factory=list)
    retrieval_family: str | None = None
    grounding_target: GroundingTarget | None = None
    token_count: int
    selected_token_count: int
    truncated: bool = False

    def as_evidence_item(self) -> EvidenceItem:
        return EvidenceItem(
            evidence_id=self.evidence_id,
            doc_id=self.doc_id,
            benchmark_doc_id=self.benchmark_doc_id,
            source_id=self.source_id,
            citation_anchor=self.citation_anchor,
            text=self.text,
            score=self.score,
            evidence_kind=self.evidence_kind,
            record_type=self.record_type,
            file_name=self.file_name,
            section_path=self.section_path,
            page_start=self.page_start,
            page_end=self.page_end,
            source_type=self.source_type,
            retrieval_channels=list(self.retrieval_channels),
            retrieval_family=self.retrieval_family,
            grounding_target=self.grounding_target,
        )


class BuiltContext(BaseModel):
    model_config = ConfigDict(frozen=True)

    evidence: list[ContextEvidence] = Field(default_factory=list)
    token_budget: int
    token_count: int
    truncated_count: int = 0
    grounded_candidate: str
    prompt: str


class RetrievalResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    decision: RoutingDecision
    evidence: EvidenceBundle
    self_check: SelfCheckResult
    reranked_evidence_ids: list[str] = Field(default_factory=list)
    reranked_benchmark_doc_ids: list[str] = Field(default_factory=list)
    graph_expanded: bool = False
    diagnostics: RetrievalDiagnostics = Field(default_factory=RetrievalDiagnostics)


class RAGQueryResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    query: str
    retrieval_profile: str
    answer: GroundedAnswer
    retrieval: RetrievalResult
    context: BuiltContext
    generation_provider: str | None = None
    generation_model: str | None = None
    generation_attempts: list[ProviderAttempt] = Field(default_factory=list)


class PublicQueryResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    query: str
    retrieval_profile: str
    answer: GroundedAnswer
    context: BuiltContext
    routing_decision: dict[str, object] = Field(default_factory=dict)
    retrieval_diagnostics: RetrievalDiagnostics = Field(default_factory=RetrievalDiagnostics)
    retrieval_self_check: dict[str, object] = Field(default_factory=dict)
    generation_provider: str | None = None
    generation_model: str | None = None
    generation_attempts: list[ProviderAttempt] = Field(default_factory=list)

def _rebuild_retrieval_result_refs() -> None:
    from rag.retrieval.analysis import RoutingDecision
    from rag.retrieval.evidence import EvidenceBundle, SelfCheckResult

    RetrievalResult.model_rebuild(
        _types_namespace={
            "EvidenceBundle": EvidenceBundle,
            "RoutingDecision": RoutingDecision,
            "SelfCheckResult": SelfCheckResult,
        }
    )


_rebuild_retrieval_result_refs()

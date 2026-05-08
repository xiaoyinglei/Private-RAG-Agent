from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict, Field

from rag.retrieval.evidence import CandidateLike
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


@dataclass
class FusedCandidateView(CandidateLike):
    evidence_id: str
    doc_id: str
    text: str
    citation_anchor: str
    score: float
    rank: int
    source_kind: str
    source_id: str | None
    section_path: Sequence[str]
    benchmark_doc_id: str | None = None
    effective_access_policy: AccessPolicy | None = None
    metadata: dict[str, str] | None = None
    record_type: str | None = None
    retrieval_channels: list[str] | None = None
    dense_score: float | None = None
    sparse_score: float | None = None
    special_score: float | None = None
    structure_score: float | None = None
    metadata_score: float | None = None
    fusion_score: float | None = None
    rrf_score: float | None = None
    unified_rank: int | None = None
    grounding_target: GroundingTarget | None = None

    @property
    def item_id(self) -> str:
        return self.evidence_id


@dataclass(frozen=True, slots=True)
class RankPipelineResult:
    candidates: list[CandidateLike]
    candidate_count: int
    collapsed_candidate_count: int
    pre_rerank_count: int
    post_cleanup_count: int
    top1_confidence: float | None
    exit_decision: str | None


class ContextEvidence(EvidenceItem):
    model_config = ConfigDict(frozen=True)

    token_count: int
    selected_token_count: int
    truncated: bool = False

    def as_evidence_item(self) -> EvidenceItem:
        return EvidenceItem.model_validate(
            self.model_dump(exclude={"token_count", "selected_token_count", "truncated"})
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

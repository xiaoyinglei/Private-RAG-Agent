from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Coroutine
from dataclasses import dataclass, field
from typing import Any, TypeVar, cast

from pydantic import BaseModel, ConfigDict, Field

from rag.retrieval.evidence import CandidateLike, EvidenceBundle, SelfCheckResult
from rag.retrieval.models import RetrievalResult
from rag.schema.query import RetrievalSignals
from rag.schema.runtime import ProviderAttempt, RetrievalDiagnostics, RuntimeMode

T = TypeVar("T")


class RoutingDecision(BaseModel):
    model_config = ConfigDict(frozen=True)

    runtime_mode: RuntimeMode
    source_scope: list[str] = Field(default_factory=list)
    web_search_allowed: bool = False
    graph_expansion_allowed: bool = False
    rerank_required: bool = True


@dataclass(frozen=True, slots=True)
class CoreRetrievalPayload:
    decision: RoutingDecision
    evidence: EvidenceBundle
    self_check: SelfCheckResult
    clean_items: list[CandidateLike]
    reranked_benchmark_doc_ids: list[str]
    graph_expanded: bool = False
    retrieval_profile: str | None = None
    branch_hits: dict[str, int] = field(default_factory=dict)
    branch_limits: dict[str, int] = field(default_factory=dict)
    planning_complexity_gate: str | None = None
    semantic_route: str | None = None
    target_collections: list[str] = field(default_factory=list)
    predicate_strategy: str | None = None
    predicate_expression: str | None = None
    version_gate_applied: bool = False
    operator_plan: list[str] = field(default_factory=list)
    rewritten_query: str | None = None
    sparse_query: str | None = None
    embedding_provider: str | None = None
    rerank_provider: str | None = None
    rerank_skipped: bool = False
    attempts: list[ProviderAttempt] = field(default_factory=list)
    fusion_strategy: str | None = None
    fusion_alpha: float | None = None
    fusion_input_count: int = 0
    fused_count: int = 0
    retrieval_signals: RetrievalSignals | None = None
    retrieval_signals_debug: dict[str, object] = field(default_factory=dict)
    pre_rerank_count: int = 0
    post_cleanup_count: int = 0
    top1_confidence: float | None = None
    exit_decision: str | None = None
    fallback_triggered: list[str] = field(default_factory=list)
    collapsed_candidate_count: int = 0


def to_retrieval_result(payload: CoreRetrievalPayload) -> RetrievalResult:
    diagnostics = build_retrieval_diagnostics(payload)
    return RetrievalResult(
        decision=payload.decision,
        evidence=payload.evidence,
        self_check=payload.self_check,
        reranked_evidence_ids=[candidate.item_id for candidate in payload.clean_items],
        reranked_benchmark_doc_ids=list(payload.reranked_benchmark_doc_ids or []),
        graph_expanded=bool(payload.graph_expanded),
        diagnostics=diagnostics,
    )


def build_retrieval_diagnostics(payload: CoreRetrievalPayload) -> RetrievalDiagnostics:
    diagnostics = RetrievalDiagnostics(
        retrieval_profile=payload.retrieval_profile,
        branch_hits=dict(payload.branch_hits or {}),
        branch_limits=dict(payload.branch_limits or {}),
        planning_complexity_gate=payload.planning_complexity_gate,
        semantic_route=payload.semantic_route,
        target_collections=list(payload.target_collections or []),
        predicate_strategy=payload.predicate_strategy,
        predicate_expression=payload.predicate_expression,
        version_gate_applied=bool(payload.version_gate_applied),
        operator_plan=list(payload.operator_plan or []),
        rewritten_query=payload.rewritten_query,
        sparse_query=payload.sparse_query,
        reranked_evidence_ids=[candidate.item_id for candidate in payload.clean_items],
        reranked_benchmark_doc_ids=list(payload.reranked_benchmark_doc_ids or []),
        embedding_provider=payload.embedding_provider,
        rerank_provider=payload.rerank_provider,
        rerank_skipped=payload.rerank_skipped,
        attempts=list(payload.attempts or []),
        fusion_strategy=payload.fusion_strategy,
        fusion_alpha=payload.fusion_alpha,
        fusion_input_count=payload.fusion_input_count,
        fused_count=payload.fused_count,
        graph_expanded=bool(payload.graph_expanded),
        retrieval_signals=payload.retrieval_signals,
        retrieval_signals_debug=dict(payload.retrieval_signals_debug or {}),
        pre_rerank_count=payload.pre_rerank_count,
        post_cleanup_count=payload.post_cleanup_count,
        top1_confidence=payload.top1_confidence,
        exit_decision=payload.exit_decision,
        fallback_triggered=list(payload.fallback_triggered or []),
        collapsed_candidate_count=payload.collapsed_candidate_count,
    )
    return diagnostics


class RuntimeCoordinator:
    def run_sync(self, awaitable: Awaitable[T]) -> T:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(cast(Coroutine[Any, Any, T], awaitable))
        raise RuntimeError("synchronous runtime bridge cannot run inside an active event loop; call the async path")


__all__ = ["CoreRetrievalPayload", "RuntimeCoordinator", "build_retrieval_diagnostics", "to_retrieval_result"]

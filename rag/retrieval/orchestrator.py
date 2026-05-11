from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, fields, replace
from typing import Any, Protocol

from rag.retrieval.analysis import (
    QueryUnderstandingService,
    RoutingDecision,
    RoutingService,
    narrow_access_policy_for_query,
)
from rag.retrieval.evidence import (
    CandidateLike,
    EvidenceService,
    EvidenceThresholds,
)
from rag.retrieval.graph import GraphExpansionService
from rag.retrieval.l3_l4_engine import L3L4RetrievalEngine
from rag.retrieval.models import (
    FusedCandidateView,
    QueryOptions,
    RankPipelineResult,
    RetrievalProfile,
    RetrievalResult,
)
from rag.retrieval.planning_graph import PlanningGraph, PlanningState
from rag.retrieval.rerank_service import IndustrialRerankService
from rag.retrieval.retrieval_adapter import RetrievalAdapter
from rag.retrieval.runtime_coordinator import (
    CoreRetrievalPayload,
    RuntimeCoordinator,
    to_retrieval_result,
)
from rag.schema.model_protocols import Reranker as ModelReranker
from rag.schema.query import RetrievalSignals
from rag.schema.runtime import AccessPolicy, ExecutionLocationPreference, ProviderAttempt
from rag.utils.telemetry import TelemetryService


class RetrievalExecutor(Protocol):
    def retrieve(
        self,
        query: str,
        *,
        access_policy: AccessPolicy,
        source_scope: Sequence[str] = (),
        execution_location_preference: ExecutionLocationPreference | None = None,
        query_options: QueryOptions | None = None,
    ) -> RetrievalResult: ...


class RetrieverFn(Protocol):
    def __call__(
        self,
        query: str,
        source_scope: list[str],
        retrieval_signals: RetrievalSignals,
    ) -> Sequence[CandidateLike]: ...


class GraphExpander(Protocol):
    def __call__(
        self,
        query: str,
        source_scope: list[str],
        evidence: list[CandidateLike],
    ) -> Sequence[CandidateLike]: ...


@dataclass(slots=True)
class BranchRetrieverRegistry:
    vector_retriever: RetrieverFn
    section_retriever: RetrieverFn
    special_retriever: RetrieverFn
    metadata_retriever: RetrieverFn
    local_retriever: RetrieverFn
    global_retriever: RetrieverFn
    web_retriever: RetrieverFn

    def collect_web(
        self,
        *,
        query: str,
        source_scope: list[str],
        retrieval_signals: RetrievalSignals,
    ) -> list[CandidateLike]:
        return list(self.web_retriever(query, source_scope, retrieval_signals))

    def get(self, branch: str) -> RetrieverFn:
        return {
            "vector": self.vector_retriever,
            "section": self.section_retriever,
            "special": self.special_retriever,
            "metadata": self.metadata_retriever,
            "local": self.local_retriever,
            "global": self.global_retriever,
        }.get(branch, self.vector_retriever)


@dataclass(slots=True)
class UnifiedReranker:
    reranker: ModelReranker | None = None

    @property
    def enabled(self) -> bool:
        return self.reranker is not None

    def rerank(self, query: str, documents: Sequence[str], **kwargs: object) -> list[float]:
        if self.reranker is None:
            raise RuntimeError("Reranker is not configured")
        rerank = getattr(self.reranker, "rerank", None)
        if callable(rerank):
            return [float(score) for score in rerank(query, documents, **kwargs)]
        if callable(self.reranker):
            return [float(score) for score in self.reranker(query, documents, **kwargs)]  # type: ignore[misc]
        raise RuntimeError("Reranker does not implement rerank(query, documents)")


@dataclass(frozen=True)
class FusedCandidate:
    candidate: CandidateLike
    fused_score: float
    rank: int
    supporting_branches: int
    branch_scores: dict[str, float]


@dataclass(frozen=True, slots=True)
class RetrievalServiceConfig:
    vector_retriever: RetrieverFn | None = None
    local_retriever: RetrieverFn | None = None
    global_retriever: RetrieverFn | None = None
    section_retriever: RetrieverFn | None = None
    special_retriever: RetrieverFn | None = None
    metadata_retriever: RetrieverFn | None = None
    graph_expander: GraphExpander | None = None
    web_retriever: RetrieverFn | None = None
    reranker: ModelReranker | None = None
    routing_service: RoutingService | None = None
    query_understanding_service: QueryUnderstandingService | None = None
    evidence_service: EvidenceService | None = None
    graph_expansion_service: GraphExpansionService | None = None
    telemetry_service: TelemetryService | None = None
    evidence_thresholds: EvidenceThresholds | None = None
    metadata_scope_resolver: object | None = None
    planning_graph: PlanningGraph | None = None
    retrieval_adapter: RetrievalAdapter | None = None
    rerank_service: IndustrialRerankService | None = None
    fusion_alpha: float = 0.65


@dataclass(slots=True)
class ReciprocalRankFusion:
    rank_constant: int = 60
    alpha: float = 0.65

    def fuse(
        self,
        *,
        query: str,
        retrieval_profile: object,
        branches: Sequence[tuple[str, Sequence[CandidateLike]]],
        alpha: float | None = None,
    ) -> list[CandidateLike]:
        del query, retrieval_profile
        blend = self._normalized_alpha(alpha)
        branch_weights = self._branch_weights(branches, alpha=blend)
        fused: dict[str, FusedCandidate] = {}
        for branch_name, branch in branches:
            branch_weight = branch_weights.get(branch_name, 1.0)
            for index, candidate in enumerate(branch, start=1):
                score = branch_weight * (1.0 / (self.rank_constant + index))
                candidate_id = candidate.item_id
                existing = fused.get(candidate_id)
                branch_scores = {branch_name: max(float(candidate.score), 0.0)}
                if existing is None:
                    fused[candidate_id] = FusedCandidate(
                        candidate=candidate,
                        fused_score=score,
                        rank=index,
                        supporting_branches=1,
                        branch_scores=branch_scores,
                    )
                    continue
                merged_scores = dict(existing.branch_scores)
                merged_scores.update(branch_scores)
                fused[candidate_id] = FusedCandidate(
                    candidate=existing.candidate,
                    fused_score=existing.fused_score + score,
                    rank=min(existing.rank, index),
                    supporting_branches=existing.supporting_branches + 1,
                    branch_scores=merged_scores,
                )

        ordered = sorted(
            fused.values(),
            key=lambda item: (-item.fused_score, -item.supporting_branches, item.rank, item.candidate.item_id),
        )
        return [self._to_view(item, index) for index, item in enumerate(ordered, start=1)]

    def _normalized_alpha(self, alpha: float | None) -> float:
        if alpha is None:
            alpha = self.alpha
        return max(0.0, min(float(alpha), 1.0))

    @classmethod
    def _branch_weights(
        cls,
        branches: Sequence[tuple[str, Sequence[CandidateLike]]],
        *,
        alpha: float,
    ) -> dict[str, float]:
        branch_names = [branch_name for branch_name, branch in branches if branch]
        semantic = [
            branch_name
            for branch_name in branch_names
            if branch_name in {"vector", "local", "global", "special"}
        ]
        lexical = [branch_name for branch_name in branch_names if branch_name in {"section", "metadata"}]
        if semantic and lexical:
            weights: dict[str, float] = {}
            semantic_weight = max(alpha, 0.05) / len(semantic)
            lexical_weight = max(1.0 - alpha, 0.05) / len(lexical)
            for branch_name in semantic:
                weights[branch_name] = semantic_weight
            for branch_name in lexical:
                weights[branch_name] = lexical_weight
            return weights
        return {branch_name: 1.0 for branch_name in branch_names}

    @staticmethod
    def _to_view(item: FusedCandidate, unified_rank: int) -> FusedCandidateView:
        final_score = item.fused_score
        return FusedCandidateView(
            evidence_id=item.candidate.item_id,
            doc_id=item.candidate.doc_id,
            benchmark_doc_id=getattr(item.candidate, "benchmark_doc_id", None),
            text=item.candidate.text,
            citation_anchor=item.candidate.citation_anchor,
            score=final_score,
            rank=item.rank,
            source_kind=item.candidate.source_kind,
            source_id=item.candidate.source_id,
            section_path=tuple(item.candidate.section_path),
            effective_access_policy=getattr(item.candidate, "effective_access_policy", None),
            metadata=getattr(item.candidate, "metadata", None),
            record_type=getattr(item.candidate, "record_type", None),
            retrieval_channels=sorted(
                    {*item.branch_scores, *(getattr(item.candidate, "retrieval_channels", None) or ())}
                ),
            dense_score=item.branch_scores.get("vector"),
            sparse_score=item.branch_scores.get("section") or item.branch_scores.get("metadata"),
            special_score=item.branch_scores.get("special"),
            structure_score=item.branch_scores.get("section"),
            metadata_score=item.branch_scores.get("metadata"),
            fusion_score=final_score,
            rrf_score=final_score,
            unified_rank=unified_rank,
            grounding_target=getattr(item.candidate, "grounding_target", None),
        )

class RetrievalService:
    @staticmethod
    def _normalize_config(
        config: RetrievalServiceConfig | None,
        overrides: dict[str, Any],
    ) -> RetrievalServiceConfig:
        config = config or RetrievalServiceConfig()
        if not overrides:
            return config
        allowed = {field.name for field in fields(RetrievalServiceConfig)}
        unexpected = sorted(set(overrides) - allowed)
        if unexpected:
            names = ", ".join(unexpected)
            raise TypeError(f"Unexpected RetrievalService option(s): {names}")
        return replace(config, **overrides)

    def __init__(
        self,
        config: RetrievalServiceConfig | None = None,
        **overrides: Any,
    ) -> None:
        config = self._normalize_config(config, overrides)
        self.config = config
        self._vector_retriever: RetrieverFn = config.vector_retriever or (lambda _query, _scope, _understanding: [])
        self._local_retriever: RetrieverFn = config.local_retriever or (lambda _query, _scope, _understanding: [])
        self._global_retriever: RetrieverFn = config.global_retriever or (lambda _query, _scope, _understanding: [])
        self._section_retriever: RetrieverFn = config.section_retriever or (lambda _query, _scope, _understanding: [])
        self._special_retriever: RetrieverFn = config.special_retriever or (lambda _query, _scope, _understanding: [])
        self._metadata_retriever: RetrieverFn = config.metadata_retriever or (lambda _query, _scope, _understanding: [])
        self._graph_expander: GraphExpander = config.graph_expander or (lambda _query, _scope, _evidence: [])
        self._web_retriever: RetrieverFn = config.web_retriever or (lambda _query, _scope, _understanding: [])
        self._reranker = config.reranker
        self._routing_service = config.routing_service or RoutingService()
        self._query_understanding_service = config.query_understanding_service or QueryUnderstandingService()
        self._evidence_service = config.evidence_service or EvidenceService(config.evidence_thresholds)
        self._graph_expansion_service = config.graph_expansion_service or GraphExpansionService()
        self._telemetry_service = config.telemetry_service
        self._fusion = ReciprocalRankFusion(alpha=config.fusion_alpha)
        self._unified_reranker = UnifiedReranker(reranker=self._reranker)
        self.last_result: RetrievalResult | None = None
        self.last_payload: CoreRetrievalPayload | None = None
        self._branch_registry = BranchRetrieverRegistry(
            vector_retriever=self._vector_retriever,
            local_retriever=self._local_retriever,
            global_retriever=self._global_retriever,
            section_retriever=self._section_retriever,
            special_retriever=self._special_retriever,
            metadata_retriever=self._metadata_retriever,
            web_retriever=self._web_retriever,
        )
        self._planning_graph = config.planning_graph or PlanningGraph(
            metadata_scope_resolver=config.metadata_scope_resolver
        )
        self._retrieval_adapter = config.retrieval_adapter or RetrievalAdapter(
            branch_registry=self._branch_registry,
            evidence_service=self._evidence_service,
            telemetry_service=self._telemetry_service,
        )
        self._rerank_service = config.rerank_service or IndustrialRerankService()
        self._runtime_coordinator = RuntimeCoordinator()
        self._l3_l4_engine = L3L4RetrievalEngine(
            branch_registry=self._branch_registry,
            routing_service=self._routing_service,
            query_understanding_service=self._query_understanding_service,
            evidence_service=self._evidence_service,
            graph_expansion_service=self._graph_expansion_service,
            telemetry_service=self._telemetry_service,
            planning_graph=self._planning_graph,
            retrieval_adapter=self._retrieval_adapter,
            rerank_service=self._rerank_service,
            fusion=self._fusion,
            reranker=self._unified_reranker,
            graph_expander=self._graph_expander,
        )
        self.branch_registry = self._branch_registry
        self.routing_service = self._routing_service
        self.query_understanding_service = self._query_understanding_service
        self.evidence_service = self._evidence_service
        self.graph_expansion_service = self._graph_expansion_service
        self.fusion = self._fusion
        self.reranker = self._unified_reranker
        self.graph_expander = self._graph_expander
        self.telemetry_service = self._telemetry_service
        self.planning_graph = self._planning_graph
        self.retrieval_adapter = self._retrieval_adapter
        self.rerank_service = self._rerank_service
        self.runtime_coordinator = self._runtime_coordinator
        self.l3_l4_engine = self._l3_l4_engine

    @staticmethod
    def _benchmark_doc_ids(candidates: Sequence[CandidateLike]) -> list[str]:
        ranked_doc_ids: list[str] = []
        seen: set[str] = set()
        for candidate in candidates:
            benchmark_doc_id = getattr(candidate, "benchmark_doc_id", None)
            if not isinstance(benchmark_doc_id, str):
                continue
            normalized = benchmark_doc_id.strip()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            ranked_doc_ids.append(normalized)
        return ranked_doc_ids

    def retrieve_payload(
        self,
        query: str,
        *,
        access_policy: AccessPolicy,
        source_scope: Sequence[str] = (),
        execution_location_preference: ExecutionLocationPreference | None = None,
        query_options: QueryOptions | None = None,
    ) -> CoreRetrievalPayload:
        payload = self.runtime_coordinator.run_sync(
            self.l3_l4_engine.arun(
                query,
                access_policy=access_policy,
                source_scope=source_scope,
                execution_location_preference=execution_location_preference,
                query_options=query_options,
            )
        )
        self.last_payload = payload
        return payload

    async def aretrieve_payload(
        self,
        query: str,
        *,
        access_policy: AccessPolicy,
        source_scope: Sequence[str] = (),
        execution_location_preference: ExecutionLocationPreference | None = None,
        query_options: QueryOptions | None = None,
    ) -> CoreRetrievalPayload:
        payload = await self.l3_l4_engine.arun(
            query,
            access_policy=access_policy,
            source_scope=source_scope,
            execution_location_preference=execution_location_preference,
            query_options=query_options,
        )
        self.last_payload = payload
        return payload

    def run(
        self,
        query: str,
        *,
        access_policy: AccessPolicy,
        source_scope: Sequence[str] = (),
        execution_location_preference: ExecutionLocationPreference | None = None,
        query_options: QueryOptions | None = None,
    ) -> RetrievalResult:
        return to_retrieval_result(
            self.retrieve_payload(
                query,
                access_policy=access_policy,
                source_scope=source_scope,
                execution_location_preference=execution_location_preference,
                query_options=query_options,
            )
        )

    async def arun(
        self,
        query: str,
        *,
        access_policy: AccessPolicy,
        source_scope: Sequence[str] = (),
        execution_location_preference: ExecutionLocationPreference | None = None,
        query_options: QueryOptions | None = None,
    ) -> RetrievalResult:
        return to_retrieval_result(
            await self.aretrieve_payload(
                query,
                access_policy=access_policy,
                source_scope=source_scope,
                execution_location_preference=execution_location_preference,
                query_options=query_options,
            )
        )

    def plan_query(
        self,
        *,
        query: str,
        access_policy: AccessPolicy,
        source_scope: Sequence[str] = (),
        execution_location_preference: ExecutionLocationPreference | None = None,
        query_options: QueryOptions | None = None,
    ) -> tuple[RetrievalSignals, AccessPolicy, RoutingDecision, PlanningState]:
        scope = list(source_scope)
        retrieval_profile = (
            query_options.resolved_retrieval_profile
            if query_options is not None
            else RetrievalProfile.AUTO
        )
        query_understanding = self.query_understanding_service.analyze(
            query,
            access_policy=access_policy,
            execution_location_preference=(
                execution_location_preference or ExecutionLocationPreference.LOCAL_FIRST
            ),
        )
        retrieval_signals = RetrievalSignals.from_query_understanding(query_understanding)
        effective_access_policy = narrow_access_policy_for_query(access_policy, query_understanding)
        decision = self.routing_service.route(
            query,
            retrieval_signals=retrieval_signals,
            source_scope=scope,
            access_policy=effective_access_policy,
        )
        plan = self.runtime_coordinator.run_sync(
            self.planning_graph.aplan(
                query,
                source_scope=scope,
                access_policy=effective_access_policy,
                retrieval_signals=retrieval_signals,
                resolved_retrieval_profile=retrieval_profile,
                query_options=query_options,
            )
        )
        return retrieval_signals, effective_access_policy, decision, plan

    def collect_internal_branches(
        self,
        *,
        plan: PlanningState,
        source_scope: Sequence[str],
        access_policy: AccessPolicy,
        runtime_mode: ExecutionLocationPreference | str | object,
        retrieval_signals: RetrievalSignals,
    ) -> object:
        return self.runtime_coordinator.run_sync(
            self.retrieval_adapter.acollect_internal(
                plan=plan,
                source_scope=list(source_scope),
                access_policy=access_policy,
                runtime_mode=runtime_mode,
                retrieval_signals=retrieval_signals,
            )
        )

    def collect_branch_candidates(
        self,
        *,
        branch: str,
        plan: PlanningState,
        retrieval_signals: RetrievalSignals,
        source_scope: Sequence[str],
        access_policy: AccessPolicy,
        runtime_mode: object,
        limit: int,
    ) -> list[CandidateLike]:
        lexical_branches = {"section", "metadata"}
        branch_query = plan.sparse_query if branch in lexical_branches else plan.rewritten_query
        retriever = self.branch_registry.get(branch)
        candidates = list(
            self.retrieval_adapter._call_branch(
                retriever=retriever,
                query=branch_query,
                source_scope=list(source_scope),
                retrieval_signals=retrieval_signals,
                plan=plan,
            )
        )
        filtered = self.evidence_service.filter_candidates(
            candidates,
            source_scope=source_scope,
            access_policy=access_policy,
            runtime_mode=runtime_mode,
            retrieval_signals=retrieval_signals,
        )
        return filtered[:limit]

    def rank_plan_branches(
        self,
        *,
        query: str,
        plan: PlanningState,
        branches: list[tuple[str, list[CandidateLike]]],
        query_options: QueryOptions | None,
        rerank_required: bool,
    ) -> RankPipelineResult:
        return self.runtime_coordinator.run_sync(
            self.l3_l4_engine._rank_branches(
                query=query,
                plan=plan,
                branches=branches,
                query_options=query_options,
                rerank_required=rerank_required,
            )
        )

    def _embedding_provider(self) -> str | None:
        for retriever in (
            self.branch_registry.vector_retriever,
            self.branch_registry.special_retriever,
        ):
            provider = getattr(retriever, "last_provider", None)
            if isinstance(provider, str) and provider:
                return provider
        return None

    def _provider_attempts(self) -> list[ProviderAttempt]:
        attempts: list[ProviderAttempt] = []
        for retriever in (
            self.branch_registry.vector_retriever,
            self.branch_registry.special_retriever,
        ):
            attempts.extend(
                attempt
                for attempt in getattr(retriever, "last_attempts", [])
                if isinstance(attempt, ProviderAttempt)
            )
        attempts.extend(
            attempt
            for attempt in getattr(self.reranker.reranker, "last_attempts", [])
            if isinstance(attempt, ProviderAttempt)
        )
        return attempts

    def retrieve(
        self,
        query: str,
        *,
        access_policy: AccessPolicy,
        source_scope: Sequence[str] = (),
        execution_location_preference: ExecutionLocationPreference | None = None,
        query_options: QueryOptions | None = None,
    ) -> RetrievalResult:
        scope = list(source_scope)
        self._prepare_retriever_policies(
            access_policy=access_policy,
            execution_location_preference=execution_location_preference,
        )
        result = self.run(
            query,
            access_policy=access_policy,
            source_scope=scope,
            execution_location_preference=execution_location_preference,
            query_options=query_options,
        )
        self.last_result = result
        return result

    async def aretrieve(
        self,
        query: str,
        *,
        access_policy: AccessPolicy,
        source_scope: Sequence[str] = (),
        execution_location_preference: ExecutionLocationPreference | None = None,
        query_options: QueryOptions | None = None,
    ) -> RetrievalResult:
        scope = list(source_scope)
        self._prepare_retriever_policies(
            access_policy=access_policy,
            execution_location_preference=execution_location_preference,
        )
        result = await self.arun(
            query,
            access_policy=access_policy,
            source_scope=scope,
            execution_location_preference=execution_location_preference,
            query_options=query_options,
        )
        self.last_result = result
        return result

    def _prepare_retriever_policies(
        self,
        *,
        access_policy: AccessPolicy,
        execution_location_preference: ExecutionLocationPreference | None,
    ) -> None:
        for retriever in (
            self._vector_retriever,
            self._local_retriever,
            self._global_retriever,
            self._special_retriever,
        ):
            prepare_for_policy = getattr(retriever, "prepare_for_policy", None)
            if callable(prepare_for_policy):
                prepare_for_policy(
                    access_policy=access_policy,
                    execution_location_preference=execution_location_preference,
                )


__all__ = [
    "BranchRetrieverRegistry",
    "GraphExpander",
    "ReciprocalRankFusion",
    "RetrievalService",
    "RetrievalServiceConfig",
    "RetrieverFn",
    "UnifiedReranker",
]

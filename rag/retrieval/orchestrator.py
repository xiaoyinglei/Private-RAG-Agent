from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, fields, replace
from typing import Any, Protocol

from rag.retrieval.runtime_coordinator import RoutingDecision
from rag.retrieval.evidence import (
    CandidateLike,
    EvidenceService,
    EvidenceThresholds,
)
from rag.retrieval.fusion import ReciprocalRankFusion
from rag.retrieval.graph import GraphExpansionService
from rag.retrieval.l3_l4_engine import L3L4RetrievalEngine
from rag.retrieval.models import (
    QueryOptions,
    RetrievalResult,
)
from rag.retrieval.planning_graph import PlanningGraph
from rag.retrieval.rerank_service import IndustrialRerankService
from rag.retrieval.retrieval_adapter import RetrievalAdapter
from rag.retrieval.runtime_coordinator import (
    CoreRetrievalPayload,
    RuntimeCoordinator,
    to_retrieval_result,
)
from rag.schema.model_protocols import Reranker as ModelReranker
from rag.schema.query import RetrievalSignals
from rag.schema.runtime import AccessPolicy, RuntimeMode
from rag.utils.telemetry import TelemetryService


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
        mapping = {
            "vector": self.vector_retriever,
            "section": self.section_retriever,
            "special": self.special_retriever,
            "metadata": self.metadata_retriever,
            "local": self.local_retriever,
            "global": self.global_retriever,
            "web": self.web_retriever,
        }
        if branch not in mapping:
            raise KeyError(f"Unknown branch {branch!r}. Valid: {sorted(mapping)}")
        return mapping[branch]


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
    evidence_service: EvidenceService | None = None
    graph_expansion_service: GraphExpansionService | None = None
    telemetry_service: TelemetryService | None = None
    evidence_thresholds: EvidenceThresholds | None = None
    metadata_scope_resolver: object | None = None
    planning_graph: PlanningGraph | None = None
    retrieval_adapter: RetrievalAdapter | None = None
    rerank_service: IndustrialRerankService | None = None
    fusion_alpha: float = 0.65


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
        self._evidence_service = config.evidence_service or EvidenceService(config.evidence_thresholds)
        self._graph_expansion_service = config.graph_expansion_service or GraphExpansionService()
        self._telemetry_service = config.telemetry_service
        self._fusion = ReciprocalRankFusion(alpha=config.fusion_alpha)  # imported from rag.retrieval.fusion
        self._unified_reranker = UnifiedReranker(reranker=self._reranker)
        # ── debug-only — main path must not depend on mutable instance state ──
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

    def retrieve_payload(
        self,
        query: str,
        *,
        access_policy: AccessPolicy,
        source_scope: Sequence[str] = (),
        query_options: QueryOptions | None = None,
    ) -> CoreRetrievalPayload:
        payload = self.runtime_coordinator.run_sync(
            self.aretrieve_payload(
                query,
                access_policy=access_policy,
                source_scope=source_scope,
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
        query_options: QueryOptions | None = None,
    ) -> CoreRetrievalPayload:
        signals = query_options.retrieval_signals if query_options else None
        retrieval_signals = signals or RetrievalSignals()
        payload = await self.l3_l4_engine.arun(
            query,
            access_policy=access_policy,
            retrieval_signals=retrieval_signals,
            decision=RoutingDecision(
                runtime_mode=RuntimeMode.FAST,
                rerank_required=True,
            ),
            source_scope=source_scope,
            query_options=query_options,
        )
        self.last_payload = payload
        return payload

    # Backward-compatible alias. Prefer retrieve()/aretrieve().
    def run(
        self,
        query: str,
        *,
        access_policy: AccessPolicy,
        source_scope: Sequence[str] = (),
        query_options: QueryOptions | None = None,
    ) -> RetrievalResult:
        return to_retrieval_result(
            self.retrieve_payload(
                query,
                access_policy=access_policy,
                source_scope=source_scope,
                query_options=query_options,
            )
        )

    # Backward-compatible alias. Prefer retrieve()/aretrieve().
    async def arun(
        self,
        query: str,
        *,
        access_policy: AccessPolicy,
        source_scope: Sequence[str] = (),
        query_options: QueryOptions | None = None,
    ) -> RetrievalResult:
        return to_retrieval_result(
            await self.aretrieve_payload(
                query,
                access_policy=access_policy,
                source_scope=source_scope,
                query_options=query_options,
            )
        )

    def retrieve(
        self,
        query: str,
        *,
        access_policy: AccessPolicy,
        source_scope: Sequence[str] = (),
        query_options: QueryOptions | None = None,
    ) -> RetrievalResult:
        result = self.run(
            query,
            access_policy=access_policy,
            source_scope=list(source_scope),
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
        query_options: QueryOptions | None = None,
    ) -> RetrievalResult:
        result = await self.arun(
            query,
            access_policy=access_policy,
            source_scope=list(source_scope),
            query_options=query_options,
        )
        self.last_result = result
        return result


__all__ = [
    "BranchRetrieverRegistry",
    "GraphExpander",
    "ReciprocalRankFusion",
    "RetrievalService",
    "RetrievalServiceConfig",
    "RetrieverFn",
    "UnifiedReranker",
]

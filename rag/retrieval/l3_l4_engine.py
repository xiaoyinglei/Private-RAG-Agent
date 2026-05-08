from __future__ import annotations

from collections.abc import Sequence

from rag.retrieval.analysis import (
    QueryUnderstandingService,
    RoutingDecision,
    RoutingService,
    narrow_access_policy_for_query,
)
from rag.retrieval.evidence import (
    CandidateLike,
    EvidenceBundle,
    EvidenceService,
    SelfCheckResult,
)
from rag.retrieval.graph import GraphExpansionService
from rag.retrieval.models import FusedCandidateView, QueryOptions, RankPipelineResult, RetrievalProfile
from rag.retrieval.planning_graph import PlanningGraph, PlanningState
from rag.retrieval.rerank_service import IndustrialRerankService
from rag.retrieval.retrieval_adapter import RetrievalAdapter
from rag.retrieval.runtime_coordinator import CoreRetrievalPayload
from rag.schema.query import QueryUnderstanding
from rag.schema.runtime import AccessPolicy, ExecutionLocationPreference, ProviderAttempt
from rag.utils.telemetry import TelemetryService


class L3L4RetrievalEngine:
    def __init__(
        self,
        *,
        branch_registry: object,
        routing_service: RoutingService,
        query_understanding_service: QueryUnderstandingService,
        evidence_service: EvidenceService,
        graph_expansion_service: GraphExpansionService,
        telemetry_service: TelemetryService | None,
        planning_graph: PlanningGraph,
        retrieval_adapter: RetrievalAdapter,
        rerank_service: IndustrialRerankService,
        fusion: object,
        reranker: object,
        graph_expander: object,
    ) -> None:
        self.branch_registry = branch_registry
        self.routing_service = routing_service
        self.query_understanding_service = query_understanding_service
        self.evidence_service = evidence_service
        self.graph_expansion_service = graph_expansion_service
        self.telemetry_service = telemetry_service
        self.planning_graph = planning_graph
        self.retrieval_adapter = retrieval_adapter
        self.rerank_service = rerank_service
        self.fusion = fusion
        self.reranker = reranker
        self.graph_expander = graph_expander

    async def arun(
        self,
        query: str,
        *,
        access_policy: AccessPolicy,
        source_scope: Sequence[str] = (),
        execution_location_preference: ExecutionLocationPreference | None = None,
        query_options: QueryOptions | None = None,
    ) -> CoreRetrievalPayload:
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
        effective_access_policy = narrow_access_policy_for_query(access_policy, query_understanding)
        decision = self.routing_service.route(
            query,
            query_understanding=query_understanding,
            source_scope=scope,
            access_policy=effective_access_policy,
        )
        if retrieval_profile is RetrievalProfile.BYPASS:
            return self._run_bypass_mode(
                query=query,
                decision=decision,
                query_understanding=query_understanding,
            )
        plan = await self.planning_graph.aplan(
            query,
            source_scope=scope,
            access_policy=effective_access_policy,
            query_understanding=query_understanding,
            resolved_retrieval_profile=retrieval_profile,
            query_options=query_options,
        )
        return await self._execute_mode_async(
            query=query,
            source_scope=scope,
            access_policy=effective_access_policy,
            decision=decision,
            query_understanding=query_understanding,
            query_options=query_options,
            plan=plan,
        )

    async def _execute_mode_async(
        self,
        *,
        query: str,
        source_scope: list[str],
        access_policy: AccessPolicy,
        decision: RoutingDecision,
        query_understanding: QueryUnderstanding,
        query_options: QueryOptions | None,
        plan: PlanningState,
    ) -> CoreRetrievalPayload:
        collection = await self.retrieval_adapter.acollect_internal(
            plan=plan,
            source_scope=source_scope,
            access_policy=access_policy,
            runtime_mode=decision.runtime_mode,
            query_understanding=query_understanding,
        )
        rank_result = await self._rank_branches(
            query=query,
            plan=plan,
            branches=collection.branches,
            query_options=query_options,
            rerank_required=decision.rerank_required,
            query_understanding=query_understanding,
        )
        reranked_candidates = rank_result.candidates
        evidence = self.evidence_service.assemble_bundle(reranked_candidates)
        executed_fallbacks: set[tuple[str, ...]] = set()
        fallback_triggered: list[str] = []
        while True:
            supplemental_branches = self._supplemental_branches(
                plan=plan,
                collection=collection,
                rank_result=rank_result,
            )
            if not supplemental_branches:
                break
            fingerprint = tuple(sorted(supplemental_branches))
            if fingerprint in executed_fallbacks:
                break
            executed_fallbacks.add(fingerprint)
            fallback_triggered.extend(branch for branch in supplemental_branches if branch not in fallback_triggered)
            supplemental = await self.retrieval_adapter.acollect_selected_branches(
                plan=plan,
                branch_names=supplemental_branches,
                source_scope=source_scope,
                access_policy=access_policy,
                runtime_mode=decision.runtime_mode,
                query_understanding=query_understanding,
                skip_absorbed_sparse=False,
            )
            collection = self._merge_branch_collection(collection, supplemental)
            rank_result = await self._rank_branches(
                query=query,
                plan=plan,
                branches=collection.branches,
                query_options=query_options,
                rerank_required=decision.rerank_required,
                query_understanding=query_understanding,
            )
            reranked_candidates = rank_result.candidates
            evidence = self.evidence_service.assemble_bundle(reranked_candidates)

        self_check = self.evidence_service.evaluate_self_check(
            bundle=evidence,
            task_type=decision.task_type,
            runtime_mode=decision.runtime_mode,
        )

        web_candidates: list[CandidateLike] = []
        if (
            plan.allow_web
            and plan.web_limit > 0
            and decision.web_search_allowed
            and access_policy.external_retrieval.value == "allow"
            and self_check.retrieve_more
        ):
            web_candidates = await self.retrieval_adapter.acollect_web(
                plan=plan,
                source_scope=source_scope,
                access_policy=access_policy,
                runtime_mode=decision.runtime_mode,
                query_understanding=query_understanding,
                limit=plan.web_limit,
            )
            collection.branch_hits["web"] = len(web_candidates)
            collection.branch_limits["web"] = plan.web_limit
            if web_candidates:
                rank_result = await self._rank_branches(
                    query=query,
                    plan=plan,
                    branches=[*collection.branches, ("web", web_candidates)],
                    query_options=query_options,
                    rerank_required=decision.rerank_required,
                    query_understanding=query_understanding,
                )
                reranked_candidates = rank_result.candidates
                evidence = self.evidence_service.assemble_bundle(reranked_candidates)

        graph_expanded = False
        if plan.allow_graph_expansion and decision.graph_expansion_allowed:
            internal_candidates = [
                candidate for candidate in reranked_candidates if candidate.source_kind == "internal"
            ]
            graph_candidates = self.graph_expansion_service.expand(
                query=query,
                source_scope=source_scope,
                evidence=evidence,
                graph_candidates=self.graph_expander(query, source_scope, internal_candidates),
                access_policy=access_policy,
            )
            if plan.graph_limit > 0:
                graph_candidates = graph_candidates[: plan.graph_limit]
            if graph_candidates:
                graph_expanded = True
                if self.telemetry_service is not None:
                    self.telemetry_service.record_graph_expansion(
                        seed_count=len(internal_candidates),
                        added_count=len(graph_candidates),
                    )
                graph_items = self.evidence_service.assemble_bundle(graph_candidates).graph
                evidence = EvidenceBundle(
                    internal=evidence.internal,
                    external=evidence.external,
                    graph=[*evidence.graph, *graph_items],
                )

        self_check = self.evidence_service.evaluate_self_check(
            bundle=evidence,
            task_type=decision.task_type,
            runtime_mode=decision.runtime_mode,
        )
        reranked_benchmark_doc_ids = self._benchmark_doc_ids(reranked_candidates)

        return CoreRetrievalPayload(
            decision=decision,
            evidence=evidence,
            self_check=self_check,
            clean_items=reranked_candidates,
            reranked_benchmark_doc_ids=reranked_benchmark_doc_ids,
            graph_expanded=graph_expanded,
            retrieval_profile=plan.retrieval_profile.value,
            branch_hits=collection.branch_hits,
            branch_limits=collection.branch_limits,
            planning_complexity_gate=plan.complexity_gate.value,
            semantic_route=plan.semantic_route,
            target_collections=list(plan.target_collections),
            predicate_strategy=plan.predicate_plan.strategy,
            predicate_expression=plan.predicate_plan.expression,
            version_gate_applied=plan.version_gate_enabled,
            operator_plan=[step.name for step in plan.operator_plan],
            rewritten_query=plan.rewritten_query,
            sparse_query=plan.sparse_query,
            embedding_provider=self._embedding_provider(),
            rerank_provider=getattr(getattr(self.reranker, "reranker", None), "last_provider", None),
            attempts=self._provider_attempts(),
            fusion_strategy=plan.fusion_strategy,
            fusion_alpha=plan.fusion_alpha,
            fusion_input_count=rank_result.candidate_count + len(web_candidates),
            fused_count=len(reranked_candidates),
            query_understanding=query_understanding,
            query_understanding_debug=self.query_understanding_service.diagnostics_payload(),
            pre_rerank_count=rank_result.pre_rerank_count,
            post_cleanup_count=rank_result.post_cleanup_count,
            top1_confidence=rank_result.top1_confidence,
            exit_decision=rank_result.exit_decision,
            fallback_triggered=fallback_triggered,
            collapsed_candidate_count=rank_result.collapsed_candidate_count,
        )

    @staticmethod
    def _merge_branch_collection(
        current: object,
        supplemental: object,
    ) -> object:
        current_branches = list(getattr(current, "branches", []) or [])
        current_hits = dict(getattr(current, "branch_hits", {}) or {})
        current_limits = dict(getattr(current, "branch_limits", {}) or {})
        for branch_name, items in getattr(supplemental, "branches", []) or []:
            current_branches.append((branch_name, list(items)))
        current_hits.update(dict(getattr(supplemental, "branch_hits", {}) or {}))
        current_limits.update(dict(getattr(supplemental, "branch_limits", {}) or {}))
        return type(current)(
            branches=current_branches,
            branch_hits=current_hits,
            branch_limits=current_limits,
        )

    @staticmethod
    def _supplemental_branches(
        *,
        plan: PlanningState,
        collection: object,
        rank_result: RankPipelineResult,
    ) -> tuple[str, ...]:
        available = {
            step.trigger: step.branch
            for step in tuple(getattr(plan, "fallback_plan", ()) or ())
            if getattr(step, "branch", None) not in set(getattr(collection, "branch_limits", {}).keys())
        }
        if rank_result.exit_decision == "asset_fallback":
            branch = available.get("asset_fallback")
            return () if branch is None else (branch,)
        if rank_result.exit_decision == "empty_response":
            branches = [
                branch_name
                for trigger_name in ("asset_fallback", "empty_response")
                if (branch_name := available.get(trigger_name)) is not None
            ]
            return tuple(dict.fromkeys(branches))
        return ()

    @staticmethod
    def _run_bypass_mode(
        *,
        query: str,
        decision: RoutingDecision,
        query_understanding: QueryUnderstanding,
    ) -> CoreRetrievalPayload:
        del query
        return CoreRetrievalPayload(
            decision=decision.model_copy(
                update={
                    "runtime_mode": decision.runtime_mode,
                    "web_search_allowed": False,
                    "graph_expansion_allowed": False,
                }
            ),
            evidence=EvidenceBundle(),
            self_check=SelfCheckResult(
                retrieve_more=False,
                evidence_sufficient=False,
                claim_supported=False,
            ),
            clean_items=[],
            reranked_benchmark_doc_ids=[],
            graph_expanded=False,
            retrieval_profile=RetrievalProfile.BYPASS.value,
            branch_hits={},
            branch_limits={},
            fusion_input_count=0,
            fused_count=0,
            query_understanding=query_understanding,
            query_understanding_debug={},
            collapsed_candidate_count=0,
        )

    async def _rank_branches(
        self,
        *,
        query: str,
        plan: PlanningState,
        branches: list[tuple[str, list[CandidateLike]]],
        query_options: QueryOptions | None,
        rerank_required: bool,
        query_understanding: QueryUnderstanding | None = None,
    ) -> RankPipelineResult:
        candidate_count = sum(len(branch) for _, branch in branches)
        fused_candidates = self.fusion.fuse(
            query=query,
            retrieval_profile=plan.retrieval_profile,
            branches=branches,
            alpha=plan.fusion_alpha,
        )
        if self.telemetry_service is not None:
            self.telemetry_service.record_rrf_fusion(
                branch_count=len(branches),
                candidate_count=candidate_count,
                fused_count=len(fused_candidates),
                duplicate_count=max(0, candidate_count - len(fused_candidates)),
            )

        rerank_result = await self.rerank_service.arank(
            query=query,
            fused_candidates=fused_candidates,
            reranker=self.reranker if getattr(self.reranker, "enabled", False) else None,
            rerank_required=rerank_required and (query_options is None or query_options.enable_rerank),
            rerank_pool_k=(query_options.rerank_pool_k if query_options is not None else None),
            allow_asset_fallback=plan.semantic_route in {"asset_first", "text_plus_asset"},
            query_understanding=query_understanding,
            min_output_candidates=query_options.resolved_candidate_top_k if query_options is not None else None,
        )
        reranked_candidates = rerank_result.ranked_candidates
        collapsed_candidate_count = 0
        collapsed_candidates: list[CandidateLike] = []
        seen_keys: set[tuple[str, str]] = set()
        for candidate in reranked_candidates:
            if candidate.source_kind != "internal":
                collapsed_candidates.append(candidate)
                continue
            dedupe_key = self._candidate_dedupe_key(candidate)
            if dedupe_key in seen_keys:
                collapsed_candidate_count += 1
                continue
            seen_keys.add(dedupe_key)
            collapsed_candidates.append(candidate)
        if query_options is None:
            reranked_candidates = collapsed_candidates
        else:
            limit = query_options.resolved_candidate_top_k
            reranked_candidates = [] if limit <= 0 else collapsed_candidates[:limit]
        if self.telemetry_service is not None:
            fused_ids = [candidate.item_id for candidate in fused_candidates]
            reranked_ids = [candidate.item_id for candidate in reranked_candidates]
            self.telemetry_service.record_rerank_effectiveness(
                input_count=len(fused_candidates),
                output_count=len(reranked_candidates),
                reordered=fused_ids != reranked_ids,
                top1_changed=(fused_ids[:1] != reranked_ids[:1]),
            )
        return RankPipelineResult(
            candidates=reranked_candidates,
            candidate_count=candidate_count,
            collapsed_candidate_count=collapsed_candidate_count,
            pre_rerank_count=rerank_result.diagnostics.input_count,
            post_cleanup_count=rerank_result.diagnostics.output_count,
            top1_confidence=rerank_result.top1_confidence,
            exit_decision=rerank_result.exit_decision,
        )

    @staticmethod
    def _candidate_dedupe_key(candidate: CandidateLike) -> tuple[str, str]:
        target = getattr(candidate, "grounding_target", None)
        if target is not None and getattr(target, "asset_id", None) is not None:
            return ("asset", str(target.asset_id))
        if target is not None and getattr(target, "section_id", None) is not None:
            return ("section", str(target.section_id))
        if target is not None and getattr(target, "doc_id", None) is not None:
            return ("document", str(target.doc_id))
        return ("evidence", candidate.item_id)

    def _embedding_provider(self) -> str | None:
        for retriever in (
            getattr(self.branch_registry, "vector_retriever", None),
            getattr(self.branch_registry, "special_retriever", None),
        ):
            provider = getattr(retriever, "last_provider", None)
            if isinstance(provider, str) and provider:
                return provider
        return None

    def _provider_attempts(self) -> list[ProviderAttempt]:
        attempts: list[ProviderAttempt] = []
        for retriever in (
            getattr(self.branch_registry, "vector_retriever", None),
            getattr(self.branch_registry, "special_retriever", None),
        ):
            attempts.extend(
                attempt
                for attempt in getattr(retriever, "last_attempts", [])
                if isinstance(attempt, ProviderAttempt)
            )
        attempts.extend(
            attempt
            for attempt in getattr(getattr(self.reranker, "reranker", None), "last_attempts", [])
            if isinstance(attempt, ProviderAttempt)
        )
        return attempts

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


__all__ = ["FusedCandidateView", "L3L4RetrievalEngine", "RankPipelineResult"]

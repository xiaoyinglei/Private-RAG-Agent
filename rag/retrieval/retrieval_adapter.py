from __future__ import annotations

import inspect
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from rag.retrieval.evidence import CandidateLike, EvidenceService
from rag.retrieval.planning_graph import FallbackStep, PlanningState, QueryVariant, RetrievalPath
from rag.schema.query import QueryUnderstanding
from rag.schema.runtime import AccessPolicy, RuntimeMode
from rag.utils.telemetry import TelemetryService


class BranchRetriever(Protocol):
    def __call__(
        self,
        query: str,
        source_scope: list[str],
        query_understanding: QueryUnderstanding,
    ) -> Sequence[CandidateLike]: ...


class PlanAwareRetriever(Protocol):
    def retrieve_with_plan(
        self,
        *,
        query: str,
        source_scope: list[str],
        query_understanding: QueryUnderstanding,
        plan: PlanningState,
    ) -> Sequence[CandidateLike]: ...


class BranchRetrieverRegistry(Protocol):
    def get(self, branch: str) -> BranchRetriever: ...

    def collect_web(
        self,
        *,
        query: str,
        source_scope: list[str],
        query_understanding: QueryUnderstanding,
    ) -> list[CandidateLike]: ...


@dataclass(frozen=True, slots=True)
class BranchCollectionResult:
    branches: list[tuple[str, list[CandidateLike]]]
    branch_hits: dict[str, int]
    branch_limits: dict[str, int]


class RetrievalAdapter:
    def __init__(
        self,
        *,
        branch_registry: BranchRetrieverRegistry,
        evidence_service: EvidenceService,
        telemetry_service: TelemetryService | None = None,
    ) -> None:
        self._branch_registry = branch_registry
        self._evidence_service = evidence_service
        self._telemetry_service = telemetry_service

    def collect_internal(
        self,
        *,
        plan: PlanningState,
        source_scope: list[str],
        access_policy: AccessPolicy,
        runtime_mode: RuntimeMode,
        query_understanding: QueryUnderstanding,
    ) -> BranchCollectionResult:
        return self._collect_sync(
            plan=plan,
            source_scope=source_scope,
            access_policy=access_policy,
            runtime_mode=runtime_mode,
            query_understanding=query_understanding,
            branch_names=None,
            skip_absorbed_sparse=True,
        )

    async def acollect_internal(
        self,
        *,
        plan: PlanningState,
        source_scope: list[str],
        access_policy: AccessPolicy,
        runtime_mode: RuntimeMode,
        query_understanding: QueryUnderstanding,
    ) -> BranchCollectionResult:
        return await self._collect_async(
            plan=plan,
            source_scope=source_scope,
            access_policy=access_policy,
            runtime_mode=runtime_mode,
            query_understanding=query_understanding,
            branch_names=None,
            skip_absorbed_sparse=True,
        )

    async def acollect_selected_branches(
        self,
        *,
        plan: PlanningState,
        branch_names: Sequence[str],
        source_scope: list[str],
        access_policy: AccessPolicy,
        runtime_mode: RuntimeMode,
        query_understanding: QueryUnderstanding,
        skip_absorbed_sparse: bool = False,
    ) -> BranchCollectionResult:
        return await self._collect_async(
            plan=plan,
            source_scope=source_scope,
            access_policy=access_policy,
            runtime_mode=runtime_mode,
            query_understanding=query_understanding,
            branch_names=branch_names,
            skip_absorbed_sparse=skip_absorbed_sparse,
        )

    def _collect_sync(
        self,
        *,
        plan: PlanningState,
        source_scope: list[str],
        access_policy: AccessPolicy,
        runtime_mode: RuntimeMode,
        query_understanding: QueryUnderstanding,
        branch_names: Sequence[str] | None,
        skip_absorbed_sparse: bool,
    ) -> BranchCollectionResult:
        branches: list[tuple[str, list[CandidateLike]]] = []
        branch_hits: dict[str, int] = {}
        selected_paths, branch_limits, skipped_branches = self._collection_inputs(
            plan=plan,
            branch_names=branch_names,
            skip_absorbed_sparse=skip_absorbed_sparse,
        )
        for path in selected_paths:
            if path.branch in skipped_branches:
                branch_hits[path.branch] = 0
                continue
            branch_query = plan.rewritten_query if path.query_variant is QueryVariant.DENSE else plan.sparse_query
            retriever = self._branch_registry.get(path.branch)
            raw_candidates = list(
                self._call_branch(
                    retriever=retriever,
                    query=branch_query,
                    source_scope=source_scope,
                    query_understanding=query_understanding,
                    plan=plan,
                )
            )
            self._record_branch_candidates(
                path=path,
                raw_candidates=raw_candidates,
                source_scope=source_scope,
                access_policy=access_policy,
                runtime_mode=runtime_mode,
                query_understanding=query_understanding,
                branches=branches,
                branch_hits=branch_hits,
            )
        return BranchCollectionResult(branches=branches, branch_hits=branch_hits, branch_limits=branch_limits)

    async def _collect_async(
        self,
        *,
        plan: PlanningState,
        source_scope: list[str],
        access_policy: AccessPolicy,
        runtime_mode: RuntimeMode,
        query_understanding: QueryUnderstanding,
        branch_names: Sequence[str] | None,
        skip_absorbed_sparse: bool,
    ) -> BranchCollectionResult:
        branches: list[tuple[str, list[CandidateLike]]] = []
        branch_hits: dict[str, int] = {}
        selected_paths, branch_limits, skipped_branches = self._collection_inputs(
            plan=plan,
            branch_names=branch_names,
            skip_absorbed_sparse=skip_absorbed_sparse,
        )
        for path in selected_paths:
            if path.branch in skipped_branches:
                branch_hits[path.branch] = 0
                continue
            branch_query = plan.rewritten_query if path.query_variant is QueryVariant.DENSE else plan.sparse_query
            retriever = self._branch_registry.get(path.branch)
            raw_candidates = list(
                await self._acall_branch(
                    retriever=retriever,
                    query=branch_query,
                    source_scope=source_scope,
                    query_understanding=query_understanding,
                    plan=plan,
                )
            )
            self._record_branch_candidates(
                path=path,
                raw_candidates=raw_candidates,
                source_scope=source_scope,
                access_policy=access_policy,
                runtime_mode=runtime_mode,
                query_understanding=query_understanding,
                branches=branches,
                branch_hits=branch_hits,
            )
        return BranchCollectionResult(branches=branches, branch_hits=branch_hits, branch_limits=branch_limits)

    def _collection_inputs(
        self,
        *,
        plan: PlanningState,
        branch_names: Sequence[str] | None,
        skip_absorbed_sparse: bool,
    ) -> tuple[tuple[RetrievalPath, ...], dict[str, int], set[str]]:
        selected_paths = self._selected_paths(plan, branch_names)
        branch_limits = {path.branch: path.limit for path in selected_paths}
        skipped_branches = self._skipped_branches(plan) if skip_absorbed_sparse else set()
        return selected_paths, branch_limits, skipped_branches

    def _record_branch_candidates(
        self,
        *,
        path: RetrievalPath,
        raw_candidates: Sequence[CandidateLike],
        source_scope: list[str],
        access_policy: AccessPolicy,
        runtime_mode: RuntimeMode,
        query_understanding: QueryUnderstanding,
        branches: list[tuple[str, list[CandidateLike]]],
        branch_hits: dict[str, int],
    ) -> None:
        filtered = self._evidence_service.filter_candidates(
            raw_candidates,
            source_scope=source_scope,
            access_policy=access_policy,
            runtime_mode=runtime_mode,
            query_understanding=query_understanding,
        )
        limited = filtered[: path.limit]
        branch_hits[path.branch] = len(limited)
        if self._telemetry_service is not None:
            self._telemetry_service.record_branch_usage(
                branch=path.branch,
                hit_count=len(limited),
                runtime_mode=runtime_mode.value,
            )
        if limited:
            branches.append((path.branch, limited))

    def collect_web(
        self,
        *,
        plan: PlanningState,
        source_scope: list[str],
        access_policy: AccessPolicy,
        runtime_mode: RuntimeMode,
        query_understanding: QueryUnderstanding,
        limit: int,
    ) -> list[CandidateLike]:
        filtered = self._evidence_service.filter_candidates(
            self._branch_registry.collect_web(
                query=plan.rewritten_query,
                source_scope=self._effective_scope(source_scope, plan=plan, supports_plan=False),
                query_understanding=query_understanding,
            ),
            source_scope=source_scope,
            access_policy=access_policy,
            runtime_mode=runtime_mode,
            query_understanding=query_understanding,
        )
        limited = filtered[:limit]
        if self._telemetry_service is not None:
            self._telemetry_service.record_branch_usage(
                branch="web",
                hit_count=len(limited),
                runtime_mode=runtime_mode.value,
            )
        return limited

    async def acollect_web(
        self,
        *,
        plan: PlanningState,
        source_scope: list[str],
        access_policy: AccessPolicy,
        runtime_mode: RuntimeMode,
        query_understanding: QueryUnderstanding,
        limit: int,
    ) -> list[CandidateLike]:
        return self.collect_web(
            plan=plan,
            source_scope=source_scope,
            access_policy=access_policy,
            runtime_mode=runtime_mode,
            query_understanding=query_understanding,
            limit=limit,
        )

    def _call_branch(
        self,
        *,
        retriever: BranchRetriever,
        query: str,
        source_scope: list[str],
        query_understanding: QueryUnderstanding,
        plan: PlanningState,
    ) -> Sequence[CandidateLike]:
        retrieve_with_plan = getattr(retriever, "retrieve_with_plan", None)
        supports_plan = callable(retrieve_with_plan)
        effective_scope = self._effective_scope(source_scope, plan=plan, supports_plan=supports_plan)
        if supports_plan:
            return retrieve_with_plan(
                query=query,
                source_scope=effective_scope,
                query_understanding=query_understanding,
                plan=plan,
            )
        return retriever(query, effective_scope, query_understanding)

    async def _acall_branch(
        self,
        *,
        retriever: BranchRetriever,
        query: str,
        source_scope: list[str],
        query_understanding: QueryUnderstanding,
        plan: PlanningState,
    ) -> Sequence[CandidateLike]:
        aretrieve_with_plan = getattr(retriever, "aretrieve_with_plan", None)
        retrieve_with_plan = getattr(retriever, "retrieve_with_plan", None)
        supports_plan = callable(aretrieve_with_plan) or callable(retrieve_with_plan)
        effective_scope = self._effective_scope(source_scope, plan=plan, supports_plan=supports_plan)
        if callable(aretrieve_with_plan):
            return await aretrieve_with_plan(
                query=query,
                source_scope=effective_scope,
                query_understanding=query_understanding,
                plan=plan,
            )
        if callable(retrieve_with_plan):
            result = retrieve_with_plan(
                query=query,
                source_scope=effective_scope,
                query_understanding=query_understanding,
                plan=plan,
            )
            if inspect.isawaitable(result):
                return await result
            return result
        result = retriever(query, effective_scope, query_understanding)
        if inspect.isawaitable(result):
            return await result
        return result

    def _skipped_branches(self, plan: PlanningState) -> set[str]:
        sparse_absorbed = False
        for path in plan.retrieval_paths:
            if path.query_variant is QueryVariant.SPARSE:
                continue
            try:
                retriever = self._branch_registry.get(path.branch)
            except Exception:
                continue
            if bool(getattr(retriever, "absorbs_sparse_branch", False)):
                sparse_absorbed = True
                break
        if not sparse_absorbed:
            return set()
        return {path.branch for path in plan.retrieval_paths if path.query_variant is QueryVariant.SPARSE}

    @staticmethod
    def _effective_scope(
        source_scope: list[str],
        *,
        plan: PlanningState,
        supports_plan: bool,
    ) -> list[str]:
        if plan.predicate_plan.strategy == "doc_id_whitelist":
            return list(plan.predicate_plan.doc_ids)
        if plan.predicate_plan.strategy == "attribute_filter" and supports_plan:
            return []
        return list(source_scope)

    @staticmethod
    def _selected_paths(plan: PlanningState, branch_names: Sequence[str] | None) -> tuple[RetrievalPath, ...]:
        if branch_names is None:
            return plan.retrieval_paths
        requested = tuple(dict.fromkeys(branch_names))
        mapped = {path.branch: path for path in plan.retrieval_paths}
        fallback_steps = {
            step.branch: step
            for step in tuple(getattr(plan, "fallback_plan", ()) or ())
            if isinstance(step, FallbackStep)
        }
        selected: list[RetrievalPath] = []
        for branch in requested:
            existing = mapped.get(branch)
            if existing is not None:
                selected.append(existing)
                continue
            fallback = fallback_steps.get(branch)
            if fallback is None:
                continue
            selected.append(
                RetrievalPath(
                    branch=branch,
                    limit=max(fallback.limit, 1),
                    query_variant=QueryVariant.DENSE,
                )
            )
        return tuple(selected)


__all__ = ["BranchCollectionResult", "RetrievalAdapter"]

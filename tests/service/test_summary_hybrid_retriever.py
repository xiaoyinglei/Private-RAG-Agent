from __future__ import annotations

import asyncio
from dataclasses import dataclass

from rag.retrieval.graph import MilvusSummaryHybridRetriever, RetrievedCandidate, SearchBackedRetrievalFactory
from rag.retrieval.models import RetrievalProfile
from rag.retrieval.planning_graph import (
    BranchStagePlan,
    CollectionStage,
    ComplexityGate,
    PlanningState,
    PredicatePlan,
    QuerySubTask,
    RetrievalPath,
)
from rag.schema.runtime import VectorSearchResult


@dataclass
class _Binding:
    provider_name: str = "fake-embedding"
    space: str = "default"
    location: str = "local"

    def embed_query(self, texts: list[str]) -> list[list[float]]:
        assert texts
        return [[0.1, 0.2]]


class _Factory:
    def build_summary_candidates_from_vector_results(
        self,
        query: str,
        results: list[VectorSearchResult],
        source_scope: list[str],
        retrieval_channels: list[str] | None = None,
    ) -> list[RetrievedCandidate]:
        del query, source_scope, retrieval_channels
        return [
            RetrievedCandidate(
                evidence_id=f"{result.item_kind}-{result.item_id}",
                doc_id=result.doc_id,
                text=result.text or result.item_kind,
                citation_anchor=f"#{result.item_id}",
                score=result.score,
                rank=index,
                metadata=dict(result.metadata),
            )
            for index, result in enumerate(results, start=1)
        ]


class _Repo:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def supports_hybrid_search(self) -> bool:
        return True

    def count_vectors(
        self,
        *,
        embedding_space: str | None = None,
        item_kind: str | None = None,
        distinct_records: bool = False,
    ) -> int:
        del embedding_space, distinct_records
        return 1 if item_kind in {"section_summary", "doc_summary"} else 0

    async def hybrid_search_async(
        self,
        *,
        query_vector,
        sparse_query: str,
        limit: int = 10,
        doc_ids: list[str] | None = None,
        expr: str | None = None,
        embedding_space: str = "default",
        item_kind: str = "section_summary",
        fusion_strategy: str = "rrf",
        alpha: float | None = None,
    ) -> list[VectorSearchResult]:
        del query_vector, sparse_query, limit, doc_ids, expr, embedding_space, fusion_strategy, alpha
        self.calls.append(item_kind)
        if item_kind == "section_summary":
            return []
        return [
            VectorSearchResult(
                item_id="7",
                item_kind="doc_summary",
                doc_id="42",
                text="fallback doc summary",
                score=0.91,
            )
        ]


@dataclass
class _SparseBinding:
    provider_name: str = "fake-bge-m3"
    space: str = "default"
    location: str = "local"
    dense_queries: list[str] | None = None
    sparse_queries: list[str] | None = None

    def __post_init__(self) -> None:
        if self.dense_queries is None:
            self.dense_queries = []
        if self.sparse_queries is None:
            self.sparse_queries = []

    def embed_query(self, texts: list[str]) -> list[list[float]]:
        self.dense_queries.extend(texts)
        return [[0.1, 0.2] for _ in texts]

    def supports_sparse_embedding(self) -> bool:
        return True

    def embed_query_sparse(self, texts: list[str]) -> list[dict[int, float]]:
        self.sparse_queries.extend(texts)
        return [{1: 0.4, 7: 0.9} for _ in texts]


class _SparseRepo:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def supports_hybrid_search(self) -> bool:
        return True

    def count_vectors(
        self,
        *,
        embedding_space: str | None = None,
        item_kind: str | None = None,
        distinct_records: bool = False,
    ) -> int:
        del distinct_records
        return 1 if embedding_space == "default" and item_kind == "section_summary" else 0

    async def hybrid_search_async(
        self,
        *,
        query_vector,
        sparse_query: str,
        sparse_query_vector: dict[int, float] | None = None,
        limit: int = 10,
        doc_ids: list[str] | None = None,
        expr: str | None = None,
        embedding_space: str = "default",
        item_kind: str = "section_summary",
        fusion_strategy: str = "rrf",
        alpha: float | None = None,
    ) -> list[VectorSearchResult]:
        del query_vector, limit, doc_ids, expr, embedding_space, item_kind, fusion_strategy, alpha
        self.calls.append(
            {
                "sparse_query": sparse_query,
                "sparse_query_vector": sparse_query_vector,
            }
        )
        return []


def test_summary_hybrid_retriever_falls_back_from_section_to_doc_when_stage_is_weak() -> None:
    retriever = MilvusSummaryHybridRetriever(
        factory=_Factory(),
        vector_repo=_Repo(),
        bindings=[_Binding()],
        item_kinds=("section_summary", "doc_summary"),
        branch_name="vector",
    )
    plan = PlanningState(
        original_query="alpha",
        rewritten_query="alpha",
        sparse_query="alpha",
        retrieval_profile=RetrievalProfile.AUTO,
        complexity_gate=ComplexityGate.STANDARD,
        semantic_route="text_first",
        target_collections=("section_summary", "doc_summary"),
        predicate_plan=PredicatePlan(),
        retrieval_paths=(RetrievalPath("vector", 6),),
        allow_web=False,
        allow_graph_expansion=False,
        web_limit=0,
        graph_limit=0,
        branch_stage_plans=(
            BranchStagePlan(
                branch="vector",
                stages=(
                    CollectionStage(collection="section_summary", limit=6, min_hits=2, trigger="always"),
                    CollectionStage(collection="doc_summary", limit=4, min_hits=2, trigger="if_insufficient"),
                ),
            ),
        ),
        fusion_strategy="weighted_rrf",
        fusion_alpha=0.7,
    )

    results = asyncio.run(
        retriever.aretrieve_with_plan(
            query="alpha",
            source_scope=["42"],
            plan=plan,
        )
    )

    assert retriever._vector_repo.calls == ["section_summary", "doc_summary"]  # type: ignore[attr-defined]
    assert [item.evidence_id for item in results] == ["doc_summary-7"]


def test_summary_hybrid_retriever_uses_query_subtasks_and_sparse_vectors() -> None:
    binding = _SparseBinding(space="bge-m3")
    repo = _SparseRepo()
    retriever = MilvusSummaryHybridRetriever(
        factory=_Factory(),
        vector_repo=repo,
        bindings=[binding],
        item_kinds=("section_summary",),
        branch_name="vector",
    )
    plan = PlanningState(
        original_query="Compare Alpha and Beta",
        rewritten_query="Compare Alpha and Beta",
        sparse_query="Compare Alpha and Beta",
        retrieval_profile=RetrievalProfile.AUTO,
        complexity_gate=ComplexityGate.COMPLEX,
        semantic_route="text_first",
        target_collections=("section_summary",),
        predicate_plan=PredicatePlan(),
        retrieval_paths=(RetrievalPath("vector", 6),),
        allow_web=False,
        allow_graph_expansion=False,
        web_limit=0,
        graph_limit=0,
        branch_stage_plans=(
            BranchStagePlan(
                branch="vector",
                stages=(CollectionStage(collection="section_summary", limit=6, min_hits=2, trigger="always"),),
            ),
        ),
        query_subtasks=(
            QuerySubTask(prompt="Alpha", purpose="collect_left_side_evidence"),
            QuerySubTask(prompt="Beta", purpose="collect_right_side_evidence"),
        ),
        fusion_strategy="weighted_rrf",
        fusion_alpha=0.7,
    )

    results = asyncio.run(
        retriever.aretrieve_with_plan(
            query="Compare Alpha and Beta",
            source_scope=["42"],
            plan=plan,
        )
    )

    assert results == []
    assert binding.dense_queries[-3:] == ["Compare Alpha and Beta", "Alpha", "Beta"]
    assert binding.sparse_queries[-3:] == ["Compare Alpha and Beta", "Alpha", "Beta"]
    assert [call["sparse_query"] for call in repo.calls] == ["Compare Alpha and Beta", "Alpha", "Beta"]
    assert all(call["sparse_query_vector"] == {1: 0.4, 7: 0.9} for call in repo.calls)


def test_summary_candidate_builder_emits_grounding_target_contract() -> None:
    factory = SearchBackedRetrievalFactory(
        metadata_repo=object(),
        graph_repo=object(),
    )

    candidates = factory.build_summary_candidates_from_vector_results(
        "alpha",
        [
            VectorSearchResult(
                item_id="7",
                item_kind="section_summary",
                doc_id="42",
                source_id="9",
                text="Section summary",
                score=0.91,
                metadata={
                    "section_id": "7",
                    "toc_path_text": "Architecture / Alpha",
                    "page_start": "2",
                    "page_end": "3",
                    "source_type": "markdown",
                },
            )
        ],
        ["42"],
    )

    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.evidence_id == "summary:section_summary:7"
    assert candidate.text == "Section summary"
    assert candidate.citation_anchor == "Architecture / Alpha"
    assert candidate.grounding_target is not None
    assert candidate.grounding_target.kind == "section"
    assert candidate.grounding_target.section_id == 7
    assert candidate.grounding_target.page_start == 2
    assert candidate.grounding_target.page_end == 3
    assert candidate.grounding_target.section_path == ["Architecture", "Alpha"]


def test_summary_candidate_builder_drops_pg_inactive_documents_even_if_milvus_returns_them() -> None:
    @dataclass
    class _MetadataRepo:
        def get_document(self, doc_id):
            del doc_id
            return type("Doc", (), {"is_active": False, "index_ready": False, "doc_status": "retired"})()

    factory = SearchBackedRetrievalFactory(
        metadata_repo=_MetadataRepo(),
        graph_repo=object(),
    )

    candidates = factory.build_summary_candidates_from_vector_results(
        "alpha",
        [
            VectorSearchResult(
                item_id="7",
                item_kind="section_summary",
                doc_id="42",
                source_id="9",
                text="Section summary",
                score=0.91,
                metadata={"section_id": "7"},
            )
        ],
        [],
    )

    assert candidates == []

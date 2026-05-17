from __future__ import annotations

from dataclasses import dataclass, field

from rag.retrieval.reranker import CandidatePoolPolicy, IndustrialRerankService


@dataclass(frozen=True)
class _FakeCandidate:
    evidence_id: str
    doc_id: str
    text: str
    citation_anchor: str
    score: float
    rank: int
    source_kind: str = "internal"
    source_id: str | None = None
    section_path: tuple[str, ...] = ()
    benchmark_doc_id: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def item_id(self) -> str:
        return self.evidence_id


class _CapturingReranker:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def rerank(self, query: str, candidates: list[str]) -> list[float]:
        del query
        self.calls.append(list(candidates))
        return [1.0 - index * 0.001 for index, _candidate in enumerate(candidates)]


def test_legacy_rerank_service_import_path_exports_service() -> None:
    from rag.retrieval.rerank_service import IndustrialRerankService as LegacyIndustrialRerankService

    assert LegacyIndustrialRerankService is IndustrialRerankService


def test_industrial_rerank_service_applies_hard_cap_before_model_rerank() -> None:
    reranker = _CapturingReranker()
    service = IndustrialRerankService(
        max_model_candidates=50,
        candidate_pool_policy=CandidatePoolPolicy(family_keep_limits={"unknown": 80}),
    )

    result = service.rank(
        query="Alpha",
        fused_candidates=[
            _FakeCandidate(
                evidence_id=f"evidence-{index}",
                doc_id="doc-a",
                text=f"candidate {index}",
                citation_anchor=f"#c{index}",
                score=1.0 - index * 0.001,
                rank=index + 1,
                metadata={"section_id": index},
            )
            for index in range(80)
        ],
        reranker=reranker,
        rerank_required=True,
        rerank_pool_k=None,
        allow_asset_fallback=False,
    )

    assert len(reranker.calls) == 1
    assert len(reranker.calls[0]) == 50
    assert result.diagnostics.output_count == 50
    assert len(result.ranked_candidates) == 50


def test_industrial_rerank_service_keeps_requested_candidate_top_k_before_final_slice() -> None:
    service = IndustrialRerankService()

    result = service.rank(
        query="Alpha",
        fused_candidates=[
            _FakeCandidate(
                evidence_id=f"evidence-{index}",
                doc_id=f"doc-{index}",
                text=f"candidate {index}",
                citation_anchor=f"#c{index}",
                score=1.0 - index * 0.001,
                rank=index + 1,
                metadata={"section_id": index, "source_family": "dense"},
            )
            for index in range(20)
        ],
        reranker=None,
        rerank_required=False,
        rerank_pool_k=None,
        allow_asset_fallback=False,
        min_output_candidates=20,
    )

    assert result.diagnostics.family_truncated_count == 0
    assert len(result.ranked_candidates) == 20

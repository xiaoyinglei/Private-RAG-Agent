from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from rag.retrieval.models import QueryOptions
from rag.retrieval.orchestrator import RetrievalService, RetrievalServiceConfig
from rag.schema.query import GroundingTarget, MetadataFilters, RetrievalSignals, StructureConstraints
from rag.schema.runtime import AccessPolicy


@dataclass(frozen=True)
class _Candidate:
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
    metadata: dict[str, Any] = field(default_factory=dict)
    record_type: str | None = None
    retrieval_channels: list[str] = field(default_factory=list)
    grounding_target: GroundingTarget | None = None

    @property
    def item_id(self) -> str:
        return self.evidence_id


@dataclass
class _CapturingRetriever:
    branch: str
    candidates: list[_Candidate]
    calls: list[dict[str, Any]] = field(default_factory=list)

    def __call__(
        self,
        query: str,
        source_scope: list[str],
        retrieval_signals: RetrievalSignals,
    ) -> list[_Candidate]:
        self.calls.append(
            {
                "query": query,
                "source_scope": list(source_scope),
                "retrieval_signals": retrieval_signals,
            }
        )
        return self.candidates


def _empty_retriever(
    query: str,
    source_scope: list[str],
    retrieval_signals: RetrievalSignals,
) -> list[_Candidate]:
    del query, source_scope, retrieval_signals
    return []


def test_retrieval_service_prioritizes_table_asset_path_and_preserves_metadata() -> None:
    vector = _CapturingRetriever(
        branch="vector",
        candidates=[
            _Candidate(
                evidence_id="summary:section_summary:12",
                doc_id="7",
                source_id="3",
                text="报表说明：开票量按区域和季度汇总，详见关联表格。",
                citation_anchor="开票量报表 / 说明",
                score=0.72,
                rank=1,
                section_path=("开票量报表", "说明"),
                metadata={"section_id": "12", "source_type": "xlsx"},
                record_type="section_summary",
                retrieval_channels=["vector"],
                grounding_target=GroundingTarget(kind="section", doc_id=7, source_id=3, section_id=12),
            )
        ],
    )
    special = _CapturingRetriever(
        branch="special",
        candidates=[
            _Candidate(
                evidence_id="summary:asset_summary:88",
                doc_id="7",
                source_id="3",
                text="表格列包括 区域、季度、开票量。可用于按区域过滤和聚合开票量。",
                citation_anchor="开票量报表 / Sheet1",
                score=0.65,
                rank=1,
                section_path=("开票量报表", "Sheet1"),
                metadata={
                    "asset_id": "88",
                    "asset_type": "table",
                    "source_type": "xlsx",
                    "page_no": "1",
                },
                record_type="asset_summary",
                retrieval_channels=["special"],
                grounding_target=GroundingTarget(kind="asset", doc_id=7, source_id=3, section_id=12, asset_id=88),
            )
        ],
    )
    service = RetrievalService(
        RetrievalServiceConfig(
            vector_retriever=vector,
            special_retriever=special,
            local_retriever=_empty_retriever,
            global_retriever=_empty_retriever,
            section_retriever=_empty_retriever,
            metadata_retriever=_empty_retriever,
            web_retriever=_empty_retriever,
        )
    )
    signals = RetrievalSignals(
        special_targets=["table"],
        quoted_terms=["开票量"],
        metadata_filters=MetadataFilters(source_types=["xlsx"], page_numbers=[1]),
        structure_constraints=StructureConstraints(
            match_strategy="heading",
            requires_structure_match=True,
            prefer_heading_match=True,
            focus_terms=["开票量报表"],
        ),
    )

    payload = asyncio.run(
        service.aretrieve_payload(
            "从开票量报表里找华东区域 Q1 开票量最高的记录",
            access_policy=AccessPolicy.default(),
            source_scope=("7",),
            query_options=QueryOptions(
                retrieval_profile="asset",
                top_k=2,
                retrieval_signals=signals,
                retrieval_signals_debug={"signals_source": "test"},
            ),
        )
    )

    assert vector.calls and special.calls
    assert special.calls[0]["retrieval_signals"] is signals
    assert payload.retrieval_profile == "asset"
    assert payload.semantic_route == "text_plus_asset"
    assert "asset_summary" in payload.target_collections
    assert "AssetSearch" in payload.operator_plan
    assert payload.branch_hits["vector"] == 1
    assert payload.branch_hits["special"] == 1
    assert payload.predicate_strategy == "doc_id_whitelist"
    assert payload.retrieval_signals is signals
    assert payload.retrieval_signals_debug == {"signals_source": "test"}

    assert payload.clean_items[0].item_id == "summary:asset_summary:88"
    asset_evidence = payload.evidence.internal[0]
    assert asset_evidence.evidence_id == "summary:asset_summary:88"
    assert asset_evidence.record_type == "asset_summary"
    assert asset_evidence.retrieval_family == "multimodal"
    assert asset_evidence.grounding_target is not None
    assert asset_evidence.grounding_target.asset_id == 88

from __future__ import annotations

from rag.retrieval.evidence import EvidenceBundle, SelfCheckResult
from rag.retrieval.runtime_coordinator import CoreRetrievalPayload, RoutingDecision, build_retrieval_diagnostics
from rag.schema.runtime import RuntimeMode


def test_retrieval_diagnostics_marks_rerank_skipped() -> None:
    payload = CoreRetrievalPayload(
        decision=RoutingDecision(runtime_mode=RuntimeMode.FAST, rerank_required=True),
        evidence=EvidenceBundle(),
        self_check=SelfCheckResult(
            retrieve_more=False,
            evidence_sufficient=False,
            claim_supported=False,
        ),
        clean_items=[],
        reranked_benchmark_doc_ids=[],
        operator_plan=["HybridFusion", "PreRerankCleanup", "Rerank", "ConfidenceAudit"],
        rerank_skipped=True,
    )

    diagnostics = build_retrieval_diagnostics(payload)

    assert diagnostics.rerank_skipped is True

from __future__ import annotations

from rag.retrieval.orchestrator import RetrievalService, RetrievalServiceConfig
from rag.schema.query import RetrievalSignals


def _empty_retriever(query: str, source_scope: list[str], retrieval_signals: RetrievalSignals) -> list[object]:
    del query, source_scope, retrieval_signals
    return []


def test_retrieval_service_accepts_config_object() -> None:
    service = RetrievalService(RetrievalServiceConfig(vector_retriever=_empty_retriever))

    assert service.branch_registry.vector_retriever is _empty_retriever


def test_retrieval_service_keeps_legacy_keyword_compatibility() -> None:
    service = RetrievalService(vector_retriever=_empty_retriever)

    assert service.branch_registry.vector_retriever is _empty_retriever

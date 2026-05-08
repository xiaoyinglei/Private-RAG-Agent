from __future__ import annotations

from typing import get_type_hints


def test_query_pipeline_lives_outside_runtime_module() -> None:
    from rag.query_pipeline import _QueryPipeline
    from rag.runtime import RAGRuntime

    assert _QueryPipeline.__module__ == "rag.query_pipeline"
    assert get_type_hints(RAGRuntime)["query_pipeline"] is _QueryPipeline

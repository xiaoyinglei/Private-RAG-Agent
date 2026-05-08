from __future__ import annotations

import json
import time

import pytest

from rag.assembly import ChatCapabilityBinding
from rag.retrieval.analysis import QueryUnderstandingService


class FakeQueryUnderstandingBackend:
    chat_model_name = "fake-query-understanding"

    def chat(self, prompt: str) -> str:
        time.sleep(0.001)
        query = prompt.split("Query: ", 1)[1].rsplit("\nJSON only.", 1)[0]
        payload = {
            "这个项目做什么？": {
                "task_type": "lookup",
                "query_type": "lookup",
            },
            "第2页讲了什么风险？": {
                "task_type": "single_doc_qa",
                "query_type": "scoped_lookup",
                "needs_metadata": True,
                "metadata_filters": {"page_numbers": [2]},
            },
            "讲系统分层的那一部分在哪一节？": {
                "task_type": "single_doc_qa",
                "query_type": "section_lookup",
                "needs_structure": True,
                "structure_constraints": {
                    "match_strategy": "heading",
                    "requires_structure_match": True,
                    "prefer_heading_match": True,
                    "focus_terms": ["architecture", "系统架构", "那一部分", "哪一节"],
                },
                "preferred_section_terms": ["系统架构"],
            },
            "系统架构分为哪几层？": {
                "task_type": "lookup",
                "query_type": "structure_lookup",
                "needs_structure": True,
                "structure_constraints": {
                    "match_strategy": "semantic",
                    "requires_structure_match": True,
                    "focus_terms": ["architecture", "系统架构"],
                },
                "preferred_section_terms": ["系统架构"],
            },
            "总结一下这个系统的核心能力。": {
                "task_type": "synthesis",
                "query_type": "summary",
            },
            "这个系统的处理流程是怎样的？": {
                "task_type": "research",
                "query_type": "process",
                "needs_graph_expansion": True,
            },
            "比较 Alpha 和 Beta 的检索链路差异。": {
                "task_type": "comparison",
                "query_type": "comparison",
            },
            "解释一下这个公式表达了什么": {
                "task_type": "lookup",
                "query_type": "special_lookup",
                "needs_special": True,
                "special_targets": ["formula"],
            },
            "pdf 第2到4页的表格指标是什么？": {
                "task_type": "single_doc_qa",
                "query_type": "special_lookup",
                "needs_special": True,
                "needs_metadata": True,
                "metadata_filters": {
                    "source_types": ["pdf"],
                    "page_ranges": [{"start": 2, "end": 4}],
                },
                "special_targets": ["table"],
            },
            "比较 pptx 和 xlsx 里的表格有什么区别": {
                "task_type": "comparison",
                "query_type": "comparison",
                "needs_special": True,
                "needs_metadata": True,
                "metadata_filters": {"source_types": ["pptx", "xlsx"]},
                "special_targets": ["table"],
            },
        }.get(query, {"task_type": "lookup", "query_type": "lookup"})
        return json.dumps(payload, ensure_ascii=False)


def _service() -> QueryUnderstandingService:
    binding = ChatCapabilityBinding(backend=FakeQueryUnderstandingBackend(), location="local")
    return QueryUnderstandingService(chat_bindings=(binding,))


@pytest.mark.parametrize(
    ("query", "query_type"),
    [
        ("这个项目做什么？", "lookup"),
        ("第2页讲了什么风险？", "scoped_lookup"),
        ("讲系统分层的那一部分在哪一节？", "section_lookup"),
        ("系统架构分为哪几层？", "structure_lookup"),
        ("总结一下这个系统的核心能力。", "summary"),
        ("这个系统的处理流程是怎样的？", "process"),
        ("比较 Alpha 和 Beta 的检索链路差异。", "comparison"),
        ("解释一下这个公式表达了什么", "special_lookup"),
    ],
)
def test_query_understanding_service_classifies_coarse_query_types(query: str, query_type: str) -> None:
    result = _service().analyze(query)

    assert result.query_type == query_type


def test_query_understanding_service_extracts_structure_constraints_from_explicit_section_query() -> None:
    result = _service().analyze("讲系统分层的那一部分在哪一节？")

    assert result.needs_structure is True
    assert result.structure_constraints.requires_structure_match is True
    assert result.structure_constraints.prefer_heading_match is True
    assert "architecture" in result.structure_constraints.focus_terms
    assert "系统架构" in result.structure_constraints.focus_terms


def test_query_understanding_service_extracts_metadata_and_special_constraints() -> None:
    result = _service().analyze("pdf 第2到4页的表格指标是什么？")

    assert result.needs_metadata is True
    assert result.needs_special is True
    assert result.metadata_filters.source_types == ["pdf"]
    assert [(item.start, item.end) for item in result.metadata_filters.page_ranges] == [(2, 4)]
    assert result.special_targets == ["table"]


def test_query_understanding_service_extracts_multiple_source_types() -> None:
    result = _service().analyze("比较 pptx 和 xlsx 里的表格有什么区别")

    assert result.query_type == "comparison"
    assert result.metadata_filters.source_types == ["pptx", "xlsx"]
    assert result.special_targets == ["table"]
    assert result.needs_metadata is True


def test_query_understanding_service_marks_process_queries_for_graph_expansion() -> None:
    result = _service().analyze("这个系统的处理流程是怎样的？")

    assert result.query_type == "process"
    assert result.needs_graph_expansion is True


def test_query_understanding_service_records_llm_latency_diagnostics() -> None:
    service = _service()

    service.analyze("这个项目做什么？")

    diagnostics = service.diagnostics_payload()
    assert diagnostics["llm_provider"] == "fakequeryunderstandingbackend"
    assert diagnostics["llm_model"] == "fake-query-understanding"
    assert diagnostics["llm_latency_ms"] is not None
    assert diagnostics["llm_latency_ms"] >= 0

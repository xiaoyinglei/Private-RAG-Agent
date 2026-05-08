from __future__ import annotations

import pytest

from rag.agent.builtin.research import RESEARCH_AGENT
from rag.agent.tools.builtin_registry import create_builtin_tool_registry
from rag.agent.tools.llm_tools import LLMTextOutput


def test_builtin_tool_registry_contains_rag_and_llm_specs() -> None:
    registry = create_builtin_tool_registry()
    names = {tool.name for tool in registry.list_all()}

    assert names == {
        "vector_search",
        "keyword_search",
        "grounding",
        "rerank",
        "graph_expand",
        "llm_generate",
        "llm_summarize",
        "llm_compare",
    }


def test_builtin_tool_registry_satisfies_research_agent_allowlist() -> None:
    registry = create_builtin_tool_registry()
    names = {tool.name for tool in registry.list_all()}

    assert set(RESEARCH_AGENT.allowed_tools) <= names


def test_builtin_tool_registry_has_no_default_runners() -> None:
    registry = create_builtin_tool_registry()

    assert all(not registry.has_runner(tool.name) for tool in registry.list_all())


@pytest.mark.anyio
async def test_builtin_tool_registry_accepts_explicit_runners() -> None:
    registry = create_builtin_tool_registry(
        runners={
            "llm_summarize": lambda payload: LLMTextOutput(
                text=f"summary:{payload.task}",
                evidence_ids=payload.evidence_ids,
                citation_ids=payload.citation_ids,
            )
        }
    )

    result = await registry.run(
        "llm_summarize",
        {"task": "Explain policy", "evidence_ids": ["ev1"], "citation_ids": ["cit1"]},
    )

    assert result == LLMTextOutput(
        text="summary:Explain policy",
        evidence_ids=["ev1"],
        citation_ids=["cit1"],
    )

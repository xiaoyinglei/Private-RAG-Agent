from __future__ import annotations

import pytest

from rag.agent.builtin.research import RESEARCH_AGENT
from rag.agent.builtin_registry import create_builtin_tool_registry
from rag.agent.tools.builtin_registry import (
    create_builtin_tool_registry as legacy_create_builtin_tool_registry,
)
from rag.agent.tools.llm_tools import LLMTextOutput


def test_builtin_tool_registry_contains_rag_and_llm_specs() -> None:
    registry = create_builtin_tool_registry()
    names = {tool.name for tool in registry.list_all()}

    # Standard RAG and LLM tools, including the ordinary grounded-answer tool.
    assert {
        "vector_search", "keyword_search", "grounding", "rerank", "graph_expand",
        "llm_generate", "llm_summarize", "llm_compare",
        "rag_search_answer",
    } <= names
    # agent-as-tool specs (registered without runners)
    assert {
        "agent_research", "agent_compare", "agent_factcheck", "agent_synthesize",
    } <= names
    # PrimitiveOps tools
    assert {
        "list_files", "read_file", "structured_probe", "write_file", "run_python",
    } <= names


def test_builtin_tool_registry_satisfies_research_agent_allowlist() -> None:
    registry = create_builtin_tool_registry()
    names = {tool.name for tool in registry.list_all()}

    assert set(RESEARCH_AGENT.allowed_tools) <= names


def test_legacy_builtin_registry_import_remains_compatible() -> None:
    assert legacy_create_builtin_tool_registry is create_builtin_tool_registry


def test_builtin_tool_execution_contracts_are_safe() -> None:
    registry = create_builtin_tool_registry()

    for spec in registry.list_all():
        if spec.max_retries > 0:
            assert spec.idempotent, f"{spec.name} retries without idempotency"
        if spec.concurrency_safe:
            assert not (
                spec.permissions.write_db
                or spec.permissions.kg_mutation
                or spec.permissions.write_fs
                or spec.permissions.execute_code
            ), f"{spec.name} allows unsafe concurrent mutation"


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

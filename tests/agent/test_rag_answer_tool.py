from __future__ import annotations

import ast
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from rag.agent.core.context import AgentRunConfig, RunRegistry
from rag.agent.tools.executor import ToolExecutor
from rag.agent.tools.integrations import knowledge as knowledge_module
from rag.agent.tools.integrations.knowledge import create_knowledge_tools
from rag.agent.tools.permissions import ToolExecutionContext as FinalToolExecutionContext
from rag.agent.tools.rag_answer_tools import (
    RAGSearchAnswerInput,
    RAGSearchAnswerRunner,
)
from rag.agent.tools.registry import ToolExecutionContext
from rag.agent.tools.tool import Tool, ToolCall, ToolCallOrigin
from rag.providers.llm_gateway import current_llm_budget_ledger
from rag.schema.query import (
    AnswerCitation,
    EvidenceItem,
    GroundedAnswer,
    RetrievalSignals,
)
from rag.schema.runtime import AccessPolicy


@pytest.mark.anyio
async def test_rag_search_answer_runner_uses_fast_runtime_query_and_preserves_contract() -> None:
    evidence = EvidenceItem(
        evidence_id="ev-runtime",
        doc_id=7,
        text="Runtime evidence",
        score=0.8,
        citation_anchor="runtime#1",
    )
    citation = AnswerCitation(
        citation_id="cit-runtime",
        evidence_id="ev-runtime",
        record_type="section",
        citation_anchor="runtime#1",
        doc_id=7,
    )

    class _Runtime:
        def __init__(self) -> None:
            self.calls: list[tuple[str, object]] = []
            self.active_ledgers: list[object | None] = []

        def query_public(self, query: str, *, options: object) -> object:
            self.calls.append((query, options))
            self.active_ledgers.append(current_llm_budget_ledger())
            return SimpleNamespace(
                answer=GroundedAnswer(
                    answer_text="Runtime fast answer",
                    citations=[citation],
                    groundedness_flag=True,
                    insufficient_evidence_flag=False,
                ),
                context=SimpleNamespace(evidence=[evidence]),
            )

    run_config = AgentRunConfig(
        run_id="rag-answer-test",
        thread_id="rag-answer-test",
        llm_budget_total=10_000,
        max_depth=2,
        access_policy=AccessPolicy.default(),
    )
    RunRegistry.remove(run_config.run_id)
    RunRegistry.get_or_create(run_config)
    runtime = _Runtime()

    output = await RAGSearchAnswerRunner(runtime=runtime).answer(
        RAGSearchAnswerInput(
            query="runtime query",
            top_k=5,
            retrieval_signals=RetrievalSignals(
                special_targets=["table"],
                quoted_terms=["runtime"],
            ),
        ),
        ToolExecutionContext(run_config=run_config),
    )

    assert output.text == "Runtime fast answer"
    assert output.evidence == [evidence]
    assert output.citations == [citation]
    assert output.groundedness_flag is True
    assert output.insufficient_evidence is False
    [(query, options)] = runtime.calls
    assert query == "runtime query"
    assert options.retrieval_profile == "fast"
    assert options.top_k == 5
    assert options.retrieval_signals.quoted_terms == ["runtime"]
    assert options.retrieval_signals_debug["special_targets"] == ["table"]
    assert runtime.active_ledgers == [
        RunRegistry.get(run_config.run_id).llm_budget_ledger
    ]
    RunRegistry.remove(run_config.run_id)


def test_knowledge_tool_is_installed_only_for_explicit_configuration() -> None:
    assert create_knowledge_tools(None) == ()

    tools = create_knowledge_tools(
        lambda _arguments: {"results": []},
        execution_revision="knowledge-v2",
    )

    assert len(tools) == 1
    assert isinstance(tools[0], Tool)
    assert tools[0].definition.name == "search_knowledge"
    assert tools[0].execution_revision.endswith(":knowledge-v2")


@pytest.mark.anyio
async def test_knowledge_runner_is_projected_into_canonical_tool_output() -> None:
    calls: list[Mapping[str, Any]] = []

    async def search(arguments: Mapping[str, Any]) -> Mapping[str, Any]:
        calls.append(arguments)
        return {
            "results": [
                {
                    "evidence_id": "ev-1",
                    "doc_id": "doc-7",
                    "citation_anchor": "doc-7#p1",
                    "text": "Canonical knowledge evidence",
                    "score": 0.91,
                    "source_type": "document",
                    "file_name": "report.pdf",
                }
            ],
            "answer_text": "Grounded answer",
            "citations": ["doc-7#p1"],
            "groundedness_flag": True,
            "insufficient_evidence": False,
            "total_found": 1,
        }

    [tool] = create_knowledge_tools(search)
    call = ToolCall(
        tool_call_id="call_knowledge",
        tool_name="search_knowledge",
        arguments={"query": "What is canonical?", "top_k": 4},
        origin=ToolCallOrigin(
            request_id="req_knowledge",
            toolset_revision="tools_knowledge_v1",
            exposed_tool_names=("search_knowledge",),
        ),
    )
    execution = await ToolExecutor({"search_knowledge": tool}).execute(
        call,
        context=FinalToolExecutionContext(),
    )

    assert execution.result.is_error is False
    assert calls[0]["query"] == "What is canonical?"
    assert calls[0]["top_k"] == 4
    assert execution.result.structured_content is not None
    assert execution.result.structured_content["results"][0]["evidence_id"] == "ev-1"
    assert execution.result.structured_content["citations"] == ("doc-7#p1",)


def test_knowledge_integration_does_not_own_retrieval_or_ingestion_lifecycle() -> None:
    module_path = Path(knowledge_module.__file__ or "")
    source = module_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }

    assert not any(
        module.startswith(("rag.retrieval", "rag.ingestion"))
        for module in imports
    )
    assert "VectorStore" not in source
    assert "ingest" not in source.lower()

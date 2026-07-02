from __future__ import annotations

from types import SimpleNamespace

import pytest

from rag.agent.core.context import AgentRunConfig, RunRegistry
from rag.agent.tools.rag_answer_tools import (
    RAGSearchAnswerInput,
    RAGSearchAnswerRunner,
)
from rag.agent.tools.registry import ToolExecutionContext
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

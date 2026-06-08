from __future__ import annotations

from types import SimpleNamespace

import pytest

from rag.agent.core.context import AgentRunConfig, RunRegistry
from rag.agent.core.definition import AgentDefinition
from rag.agent.graphs.base import build_agent_graph
from rag.agent.graphs.nodes.execute import run_tools_raw
from rag.agent.state import AgentState, ThinkOutput, ToolCallPlan
from rag.agent.tools.builtin_registry import create_builtin_tool_registry
from rag.agent.tools.rag_answer_tools import (
    RAGSearchAnswerInput,
    RAGSearchAnswerOutput,
    RAGSearchAnswerRunner,
)
from rag.agent.tools.registry import ToolExecutionContext
from rag.providers.llm_gateway import current_llm_budget_ledger
from rag.schema.query import AnswerCitation, EvidenceItem, GroundedAnswer, RetrievalSignals
from rag.schema.runtime import AccessPolicy


class _RetrievalHintProvider:
    def hint(self, state: AgentState) -> dict[str, object]:
        del state
        return {
            "decision_reason": "simple_lookup",
            "retrieval_signals": RetrievalSignals(quoted_terms=["policy"]),
        }


class _RAGSearchDecisionProvider:
    def __init__(self) -> None:
        self.calls = 0

    async def decide(self, state: AgentState, **_: object) -> ThinkOutput:
        self.calls += 1
        signals = state["retrieval_signals"]
        return ThinkOutput(
            action="execute",
            thought="use retrieval hint to fetch grounded answer",
            tool_calls=[
                ToolCallPlan.create(
                    "rag_search_answer",
                    {
                        "query": state["task"],
                        "top_k": 8,
                        "retrieval_signals": signals.model_dump(mode="json"),
                    },
                )
            ],
        )


def _definition() -> AgentDefinition:
    return AgentDefinition(
        agent_type="rag_answer_test",
        description="Fast path test",
        system_prompt="Answer simple questions.",
        allowed_tools=["rag_search_answer", "vector_search"],
    )


def _state() -> AgentState:
    run_config = AgentRunConfig(
        run_id="rag-answer-test",
        thread_id="rag-answer-test",
        budget_total=10000,
        max_depth=2,
        access_policy=AccessPolicy.default(),
    )
    RunRegistry.remove(run_config.run_id)
    RunRegistry.get_or_create(run_config)
    return {
        "messages": [],
        "evidence": [],
        "citations": [],
        "tool_results": [],
        "task": "What is the policy?",
        "retrieval_signals": RetrievalSignals(),
        "retrieval_signals_debug": None,
        "run_config": run_config,
        "iteration": 0,
        "status": "running",
        "decision_reason": None,
        "stop_reason": None,
        "needs_user_input": None,
        "pending_tool_calls": [],
        "approved_tool_call_ids": [],
        "denied_tool_call_ids": [],
        "user_decision": None,
        "user_message": None,
        "human_input_request": None,
        "human_input_response": None,
        "working_summary": None,
        "extracted_facts": [],
        "context_budget": None,
        "final_answer": None,
        "groundedness_flag": False,
        "insufficient_evidence_flag": False,
    }


@pytest.mark.anyio
async def test_model_decision_can_select_rag_search_answer_from_retrieval_hint() -> None:
    calls: list[RAGSearchAnswerInput] = []
    evidence = EvidenceItem(
        evidence_id="ev-fast",
        doc_id=1,
        text="Policy evidence",
        score=0.9,
        citation_anchor="doc#policy",
    )
    citation = AnswerCitation(
        citation_id="cit-fast",
        evidence_id="ev-fast",
        record_type="section",
        citation_anchor="doc#policy",
        doc_id=1,
    )

    def rag_search_answer(payload: RAGSearchAnswerInput) -> RAGSearchAnswerOutput:
        calls.append(payload)
        return RAGSearchAnswerOutput(
            text="Fast path answer",
            evidence=[evidence],
            citations=[citation],
            groundedness_flag=True,
        )

    def vector_search(_: object) -> object:
        raise AssertionError("model decision should not call vector_search")

    provider = _RAGSearchDecisionProvider()
    graph = build_agent_graph(
        definition=_definition(),
        tool_registry=create_builtin_tool_registry(
            runners={
                "rag_search_answer": rag_search_answer,
                "vector_search": vector_search,
            }
        ),
        retrieval_hint_provider=_RetrievalHintProvider(),
        tool_decision_provider=provider,
    )

    result = await graph.ainvoke(
        _state(),
        config={"configurable": {"thread_id": "rag-answer-test"}},
    )

    assert result["status"] == "done"
    assert result["final_answer"] == "Fast path answer"
    assert result["evidence"] == [evidence]
    assert result["citations"] == [citation]
    assert result["groundedness_flag"] is True
    assert result["insufficient_evidence_flag"] is False
    assert [tool_result.tool_name for tool_result in result["tool_results"]] == [
        "rag_search_answer"
    ]
    assert calls[0].query == "What is the policy?"
    assert calls[0].retrieval_signals is not None
    assert calls[0].retrieval_signals.quoted_terms == ["policy"]
    assert provider.calls == 1
    RunRegistry.remove("rag-answer-test")


@pytest.mark.anyio
async def test_run_tools_raw_does_not_create_tool_call_from_legacy_execution_mode() -> None:
    calls: list[object] = []
    state = _state()
    state["execution_mode"] = "fast_path"  # type: ignore[typeddict-unknown-key]

    update = await run_tools_raw(
        state,
        tool_registry=create_builtin_tool_registry(
            runners={"vector_search": lambda payload: calls.append(payload)}
        ),
        allowed_tools=frozenset({"vector_search"}),
    )

    assert update == {}
    assert calls == []
    RunRegistry.remove("rag-answer-test")


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

    runtime = _Runtime()
    runner = RAGSearchAnswerRunner(runtime=runtime)
    state = _state()

    output = await runner.answer(
        RAGSearchAnswerInput(
            query="runtime query",
            top_k=5,
            retrieval_signals=RetrievalSignals(
                special_targets=["table"],
                quoted_terms=["runtime"],
            ),
        ),
        ToolExecutionContext(run_config=state["run_config"]),
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
    assert options.retrieval_signals is not None
    assert options.retrieval_signals.special_targets == []
    assert options.retrieval_signals.quoted_terms == ["runtime"]
    assert options.retrieval_signals_debug["special_targets"] == ["table"]
    assert options.retrieval_signals_debug["answer_path_special_targets_skipped"] == ["table"]
    assert runtime.active_ledgers == [
        RunRegistry.get("rag-answer-test").budget_ledger
    ]
    RunRegistry.remove("rag-answer-test")

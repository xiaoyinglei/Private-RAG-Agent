from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from rag.agent.builtin.research import RESEARCH_AGENT
from rag.agent.core.definition import AgentDefinition
from rag.agent.graphs.nodes.synthesize import SynthesisRunResult
from rag.agent.service import AgentRunRequest, AgentService
from rag.agent.state import AgentState, ThinkOutput, ToolCallPlan
from rag.agent.tools.builtin_registry import create_builtin_tool_registry
from rag.agent.tools.llm_tools import LLMTextOutput
from rag.agent.tools.rag_tools import SearchInput, SearchOutput
from rag.agent.tools.spec import ToolResult
from rag.schema.query import AnswerCitation, EvidenceItem, RetrievalSignals


class _TableRetrievalHintProvider:
    def hint(self, state: AgentState) -> dict[str, object]:
        del state
        return {
            "decision_reason": "complex_table_question",
            "retrieval_signals": RetrievalSignals(
                special_targets=["table"],
                quoted_terms=["开票量"],
                allow_graph_expansion=False,
            ),
            "retrieval_signals_debug": {
                "signals_source": "test_structured_route",
                "special_targets": ["table"],
                "quoted_terms": ["开票量"],
            },
        }


class _TwoStepDecisionProvider:
    def decide(
        self,
        state: AgentState,
        *,
        definition: AgentDefinition,
        budget_remaining: int,
        context: object,
    ) -> ThinkOutput:
        del definition, budget_remaining, context
        tool_results = state.get("tool_results", [])
        ok_tool_names = [result.tool_name for result in tool_results if result.status == "ok"]
        if "vector_search" not in ok_tool_names:
            return ThinkOutput(
                action="execute",
                thought="retrieve table evidence first",
                tool_calls=[
                    ToolCallPlan.create(
                        "vector_search",
                        {"query": state["task"], "top_k": 4},
                    )
                ],
            )
        if "llm_summarize" not in ok_tool_names:
            vector_result = next(result for result in tool_results if result.tool_name == "vector_search")
            assert isinstance(vector_result.output, SearchOutput)
            evidence_ids = [
                str(item["evidence_id"])
                for item in vector_result.output.items
                if "evidence_id" in item
            ]
            return ThinkOutput(
                action="execute",
                thought="synthesize retrieved evidence",
                tool_calls=[
                    ToolCallPlan.create(
                        "llm_summarize",
                        {
                            "task": state["task"],
                            "context_sections": [
                                str(item["text"]) for item in vector_result.output.items
                            ],
                            "evidence_ids": evidence_ids,
                            "citation_ids": ["cit-table"],
                        },
                    )
                ],
            )
        return ThinkOutput(
            action="synthesize",
            thought="all required tools completed",
            stop_reason="complex_loop_complete",
        )


@dataclass
class _CapturingSearchRunner:
    seen_payloads: list[SearchInput] = field(default_factory=list)

    def __call__(self, payload: SearchInput) -> SearchOutput:
        self.seen_payloads.append(payload)
        return SearchOutput(
            items=[
                {
                    "evidence_id": "ev-table",
                    "doc_id": 7,
                    "citation_anchor": "开票量报表 / 华东",
                    "text": "华东区域 2026 年 Q1 开票量为 1280 单。",
                    "score": 0.94,
                    "record_type": "asset_summary",
                    "retrieval_channels": ["vector", "special"],
                }
            ]
        )


@dataclass
class _CapturingSummarizeRunner:
    seen_contexts: list[list[str]] = field(default_factory=list)

    def __call__(self, payload: Any) -> LLMTextOutput:
        self.seen_contexts.append(list(payload.context_sections))
        return LLMTextOutput(
            text="华东区域 2026 年 Q1 开票量为 1280 单。",
            evidence_ids=list(payload.evidence_ids),
            citation_ids=list(payload.citation_ids),
        )


@dataclass
class _SynthesisResult:
    status: str = "done"
    final_answer: str | None = None
    stop_reason: str | None = None
    tool_results: list[ToolResult] = field(default_factory=list)
    evidence: list[EvidenceItem] = field(default_factory=list)
    citations: list[AnswerCitation] = field(default_factory=list)
    groundedness_flag: bool = True
    insufficient_evidence_flag: bool = False


class _ToolAwareSynthesisRunner:
    def run_synthesis(self, *, parent_state: AgentState) -> SynthesisRunResult:
        tool_names = [result.tool_name for result in parent_state.get("tool_results", [])]
        assert tool_names == ["vector_search", "llm_summarize"]
        summarize = next(result for result in parent_state["tool_results"] if result.tool_name == "llm_summarize")
        assert isinstance(summarize.output, LLMTextOutput)
        return _SynthesisResult(
            final_answer=summarize.output.text,
            tool_results=list(parent_state["tool_results"]),
            groundedness_flag=True,
            insufficient_evidence_flag=False,
        )


@pytest.mark.anyio
async def test_agent_loop_routes_table_query_injects_rag_signals_and_synthesizes() -> None:
    search_runner = _CapturingSearchRunner()
    summarize_runner = _CapturingSummarizeRunner()
    service = AgentService(
        definition=RESEARCH_AGENT,
        tool_registry=create_builtin_tool_registry(
            runners={
                "vector_search": search_runner,
                "llm_summarize": summarize_runner,
            }
        ),
        retrieval_hint_provider=_TableRetrievalHintProvider(),
        tool_decision_provider=_TwoStepDecisionProvider(),
        synthesis_runner=_ToolAwareSynthesisRunner(),
    )

    result = await service.run(
        AgentRunRequest(
            task="请基于开票量表格回答：华东区域 2026 年 Q1 开票量是多少？",
            run_id="complex-agent-rag-loop",
            thread_id="complex-agent-rag-loop",
        )
    )

    assert result.status == "done"
    assert result.final_answer == "华东区域 2026 年 Q1 开票量为 1280 单。"
    assert [tool.tool_name for tool in result.tool_results] == ["vector_search", "llm_summarize"]
    assert search_runner.seen_payloads[0].retrieval_signals is not None
    assert search_runner.seen_payloads[0].retrieval_signals.special_targets == ["table"]
    assert search_runner.seen_payloads[0].retrieval_signals.quoted_terms == ["开票量"]
    assert summarize_runner.seen_contexts == [["华东区域 2026 年 Q1 开票量为 1280 单。"]]

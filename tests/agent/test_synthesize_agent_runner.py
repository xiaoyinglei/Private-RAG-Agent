from __future__ import annotations

import pytest

from rag.agent.builtin.synthesize import SYNTHESIZE_AGENT
from rag.agent.core.agent_service_factory import AgentServiceFactory
from rag.agent.core.context import AgentRunConfig, RuntimeRegistry
from rag.agent.core.registry import AgentRegistry
from rag.agent.core.subagent_runner import BuiltinSynthesisRunner
from rag.agent.core.task import SubTaskNode, SubTaskResult, SubTaskStatus
from rag.agent.graphs.nodes.synthesize import synthesize_node
from rag.agent.state import AgentState
from rag.agent.tools.builtin_registry import create_builtin_tool_registry
from rag.agent.tools.fast_path_tools import RAGSearchAnswerOutput
from rag.agent.tools.llm_tools import LLMGenerateInput, LLMTextOutput
from rag.agent.tools.spec import ToolResult
from rag.schema.query import AnswerCitation, EvidenceItem, RetrievalSignals
from rag.schema.runtime import AccessPolicy


def _state() -> AgentState:
    evidence = EvidenceItem(
        evidence_id="ev-child",
        doc_id=1,
        text="Grounded child evidence",
        score=0.9,
        citation_anchor="doc#1",
    )
    citation = AnswerCitation(
        citation_id="cit-child",
        evidence_id="ev-child",
        record_type="section",
        citation_anchor="doc#1",
    )
    subtask = SubTaskNode(
        subtask_id="s1",
        agent_type="research",
        prompt="Research child evidence",
        priority=1,
    )
    run_config = AgentRunConfig(
        run_id="synthesis-parent",
        thread_id="synthesis-parent",
        budget_total=10000,
        max_depth=2,
        access_policy=AccessPolicy.default(),
    )
    RuntimeRegistry.remove(run_config.run_id)
    RuntimeRegistry.get_or_create(run_config)
    return {
        "messages": [],
        "evidence": [evidence],
        "citations": [citation],
        "tool_results": [],
        "task": "Write final answer",
        "retrieval_signals": RetrievalSignals(),
        "retrieval_signals_debug": None,
        "run_config": run_config,
        "plan": None,
        "iteration": 0,
        "status": "done",
        "route_reason": None,
        "stop_reason": "all_subtasks_terminal",
        "needs_user_input": None,
        "pending_tool_calls": [],
        "approved_tool_call_ids": [],
        "denied_tool_call_ids": [],
        "user_decision": None,
        "user_message": None,
        "human_input_request": None,
        "human_input_response": None,
        "next_subtasks": None,
        "working_summary": None,
        "extracted_facts": [],
        "context_budget": None,
        "subtask_results": {
            "s1": SubTaskResult(
                subtask=subtask,
                status=SubTaskStatus.COMPLETED,
                findings=["Child finding"],
                evidence=[evidence],
                citations=[citation],
            )
        },
        "terminal_subtasks": {"s1"},
        "successful_subtasks": {"s1"},
        "final_answer": None,
        "groundedness_flag": False,
        "insufficient_evidence_flag": False,
    }


@pytest.mark.anyio
async def test_synthesize_node_delegates_to_builtin_synthesize_agent() -> None:
    seen_payloads: list[LLMGenerateInput] = []

    def llm_generate(payload: LLMGenerateInput) -> LLMTextOutput:
        seen_payloads.append(payload)
        return LLMTextOutput(
            text="synthesized answer",
            evidence_ids=payload.evidence_ids,
            citation_ids=payload.citation_ids,
        )

    agent_registry = AgentRegistry()
    agent_registry.register(SYNTHESIZE_AGENT)
    service_factory = AgentServiceFactory(
        tool_registry=create_builtin_tool_registry(runners={"llm_generate": llm_generate}),
        model_registry=None,
    )
    synthesis_runner = BuiltinSynthesisRunner(
        agent_registry=agent_registry,
        service_factory=service_factory,
    )
    service_factory.bind_synthesis_runner(synthesis_runner)

    update = await synthesize_node(_state(), synthesis_runner=synthesis_runner)

    assert update["status"] == "done"
    assert update["final_answer"] == "synthesized answer"
    assert update["groundedness_flag"] is True
    assert update["insufficient_evidence_flag"] is False
    assert update["tool_results"][0].tool_name == "llm_generate"
    [payload] = seen_payloads
    assert payload.evidence_ids == ["ev-child"]
    assert payload.citation_ids == ["cit-child"]
    assert any("Child finding" in section for section in payload.context_sections)
    RuntimeRegistry.remove("synthesis-parent")


@pytest.mark.anyio
async def test_synthesize_node_preserves_grounded_rag_search_answer() -> None:
    called = False

    def llm_generate(payload: LLMGenerateInput) -> LLMTextOutput:
        nonlocal called
        called = True
        return LLMTextOutput(text=f"rewritten: {payload.prompt}")

    state = _state()
    state["subtask_results"] = {}
    state["terminal_subtasks"] = set()
    state["successful_subtasks"] = set()
    state["tool_results"] = [
        ToolResult(
            tool_call_id="call-rag",
            tool_name="rag_search_answer",
            status="ok",
            output=RAGSearchAnswerOutput(
                text="日提货总量是131.074462。 [1]",
                evidence=state["evidence"],
                citations=state["citations"],
                groundedness_flag=True,
                insufficient_evidence=False,
            ),
            latency_ms=10.0,
        )
    ]

    agent_registry = AgentRegistry()
    agent_registry.register(SYNTHESIZE_AGENT)
    service_factory = AgentServiceFactory(
        tool_registry=create_builtin_tool_registry(runners={"llm_generate": llm_generate}),
        model_registry=None,
    )
    synthesis_runner = BuiltinSynthesisRunner(
        agent_registry=agent_registry,
        service_factory=service_factory,
    )
    service_factory.bind_synthesis_runner(synthesis_runner)

    update = await synthesize_node(state, synthesis_runner=synthesis_runner)

    assert update["status"] == "done"
    assert update["final_answer"] == "日提货总量是131.074462。 [1]"
    assert update["groundedness_flag"] is True
    assert update["insufficient_evidence_flag"] is False
    assert called is False
    RuntimeRegistry.remove("synthesis-parent")

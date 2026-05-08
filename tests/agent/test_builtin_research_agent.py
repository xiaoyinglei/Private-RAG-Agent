from __future__ import annotations

import pytest

from rag.agent.builtin.research import RESEARCH_AGENT
from rag.agent.builtin.research import create_research_agent_service
from rag.agent.core.compiler import AgentGraphCompiler
from rag.agent.service import AgentRunRequest
from rag.agent.state import ToolCallPlan
from rag.agent.tools.llm_tools import LLMTextOutput
from rag.agent.tools.llm_tools import ALL_LLM_TOOLS
from rag.agent.tools.rag_tools import ALL_RAG_TOOLS
from rag.agent.tools.registry import ToolRegistry


def _registry_with_builtin_tools() -> ToolRegistry:
    registry = ToolRegistry()
    for tool in [*ALL_RAG_TOOLS, *ALL_LLM_TOOLS]:
        registry.register(tool)
    return registry


def test_research_agent_uses_spec_tool_allowlist() -> None:
    assert RESEARCH_AGENT.agent_type == "research"
    assert RESEARCH_AGENT.allowed_tools == [
        "vector_search",
        "keyword_search",
        "grounding",
        "rerank",
        "llm_summarize",
    ]


def test_research_agent_prompt_requires_grounded_citations() -> None:
    prompt = RESEARCH_AGENT.system_prompt

    assert "retrieved evidence" in prompt
    assert "citations" in prompt
    assert "insufficient evidence" in prompt


def test_research_agent_compiles_when_builtin_tools_registered() -> None:
    compiler = AgentGraphCompiler(tool_registry=_registry_with_builtin_tools())

    graph = compiler.compile(RESEARCH_AGENT)

    assert hasattr(graph, "ainvoke")


@pytest.mark.anyio
async def test_research_agent_service_factory_wires_explicit_runners() -> None:
    service = create_research_agent_service(
        runners={
            "llm_summarize": lambda payload: LLMTextOutput(
                text=f"summary:{payload.task}",
                evidence_ids=payload.evidence_ids,
                citation_ids=payload.citation_ids,
            )
        }
    )
    call = ToolCallPlan.create(
        "llm_summarize",
        {"task": "Explain policy", "evidence_ids": ["ev1"], "citation_ids": ["cit1"]},
    )

    result = await service.run(
        AgentRunRequest(
            task="Explain policy",
            run_id="research-factory",
            thread_id="research-factory",
            pending_tool_calls=[call],
        )
    )

    assert result.status == "done"
    assert result.final_answer == "summary:Explain policy"
    assert result.tool_results[0].status == "ok"
    assert result.tool_results[0].output == LLMTextOutput(
        text="summary:Explain policy",
        evidence_ids=["ev1"],
        citation_ids=["cit1"],
    )

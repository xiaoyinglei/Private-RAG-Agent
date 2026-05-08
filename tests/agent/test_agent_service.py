from __future__ import annotations

import pytest

from rag.agent.builtin.research import RESEARCH_AGENT
from rag.agent.core.context import RuntimeRegistry
from rag.agent.service import AgentRunRequest, AgentService
from rag.agent.state import ToolCallPlan
from rag.agent.tools.builtin_registry import create_builtin_tool_registry
from rag.agent.tools.llm_tools import LLMTextOutput
from rag.schema.query import QueryUnderstanding, TaskType


class _ResearchUnderstandingService:
    def analyze(
        self,
        query: str,
        *,
        access_policy: object | None = None,
        execution_location_preference: object | None = None,
    ) -> QueryUnderstanding:
        del query, access_policy, execution_location_preference
        return QueryUnderstanding(task_type=TaskType.RESEARCH, query_type="research")


def _service_with_registry(runners: dict | None = None) -> AgentService:
    return AgentService(
        definition=RESEARCH_AGENT,
        tool_registry=create_builtin_tool_registry(runners=runners),
        query_understanding_service=_ResearchUnderstandingService(),
    )


def test_agent_service_initial_state_creates_runtime_handles() -> None:
    service = _service_with_registry()
    request = AgentRunRequest(task="Explain policy", run_id="svc-state", thread_id="svc-state")

    state = service.initial_state(request)

    assert state["task"] == "Explain policy"
    assert state["run_config"].run_id == "svc-state"
    assert state["run_config"].budget_total == RESEARCH_AGENT.estimated_token_budget
    assert RuntimeRegistry.get("svc-state") is not None
    RuntimeRegistry.remove("svc-state")


@pytest.mark.anyio
async def test_agent_service_run_executes_explicit_tool_call_with_runner() -> None:
    call = ToolCallPlan.create(
        "llm_summarize",
        {"task": "Explain policy", "evidence_ids": ["ev1"], "citation_ids": ["cit1"]},
    )
    service = _service_with_registry(
        runners={
            "llm_summarize": lambda payload: LLMTextOutput(
                text=f"summary:{payload.task}",
                evidence_ids=payload.evidence_ids,
                citation_ids=payload.citation_ids,
            )
        }
    )

    result = await service.run(
        AgentRunRequest(
            task="Explain policy",
            run_id="svc-ok",
            thread_id="svc-ok",
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
    with pytest.raises(KeyError):
        RuntimeRegistry.get("svc-ok")


@pytest.mark.anyio
async def test_agent_service_run_without_runner_fails_closed() -> None:
    call = ToolCallPlan.create("llm_summarize", {"task": "Explain policy"})
    service = _service_with_registry()

    result = await service.run(
        AgentRunRequest(
            task="Explain policy",
            run_id="svc-fail-closed",
            thread_id="svc-fail-closed",
            pending_tool_calls=[call],
        )
    )

    assert result.status == "done"
    assert result.insufficient_evidence_flag is True
    assert result.tool_results[0].status == "error"
    assert result.tool_results[0].error.code == "tool_not_implemented"

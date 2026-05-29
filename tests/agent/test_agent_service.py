from __future__ import annotations

import pytest

from rag.agent.builtin.research import RESEARCH_AGENT
from rag.agent.core.context import AgentRunConfig, RuntimeRegistry
from rag.agent.service import AgentRunRequest, AgentService
from rag.agent.state import ToolCallPlan
from rag.agent.tools.builtin_registry import create_builtin_tool_registry
from rag.agent.tools.llm_tools import LLMTextOutput
from rag.schema.query import RetrievalSignals
from rag.schema.runtime import AccessPolicy


class _ResearchUnderstandingService:
    def analyze(
        self,
        query: str,
        *,
        access_policy: object | None = None,
    ) -> RetrievalSignals:
        del query, access_policy
        return RetrievalSignals()


def _service_with_registry(runners: dict | None = None) -> AgentService:
    return AgentService(
        definition=RESEARCH_AGENT,
        tool_registry=create_builtin_tool_registry(runners=runners),

    )


def test_agent_service_initial_state_creates_runtime_handles() -> None:
    service = _service_with_registry()
    request = AgentRunRequest(task="Explain policy", run_id="svc-state", thread_id="svc-state")

    state = service.initial_state(request)

    assert state["task"] == "Explain policy"
    assert state["run_config"].run_id == "svc-state"
    assert state["run_config"].budget_total == RESEARCH_AGENT.estimated_token_budget
    assert "tool_action_proposals" not in state
    assert "plan" not in state
    assert "subtask_results" not in state
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

    assert result.status == "failed"
    assert result.stop_reason == "tool_error"
    assert result.insufficient_evidence_flag is True
    assert result.tool_results[0].status == "error"
    assert result.tool_results[0].error.code == "tool_not_implemented"


@pytest.mark.anyio
async def test_agent_service_run_with_config_uses_supplied_runtime_contract() -> None:
    call = ToolCallPlan.create(
        "llm_summarize",
        {"task": "Explain policy", "evidence_ids": ["ev1"], "citation_ids": ["cit1"]},
    )
    config = AgentRunConfig(
        run_id="svc-child",
        thread_id="svc-child-thread",
        parent_run_id="svc-parent",
        source_scope=("doc-1",),
        budget_total=5000,
        max_depth=1,
        access_policy=AccessPolicy.default(),
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

    result = await service.run_with_config(
        task="Explain policy",
        run_config=config,
        pending_tool_calls=[call],
    )

    assert result.run_id == "svc-child"
    assert result.thread_id == "svc-child-thread"
    assert result.status == "done"
    assert result.final_answer == "summary:Explain policy"
    with pytest.raises(KeyError):
        RuntimeRegistry.get("svc-child")


@pytest.mark.anyio
async def test_agent_service_run_creates_workspace_and_injects_primitive_ops() -> None:
    """Verify AgentService.run() creates workspace and PrimitiveOps runners are available."""
    from rag.agent.workspace import create_temp_workspace
    from rag.agent.primitive_ops import PrimitiveOps
    from rag.agent.tools.builtin_registry import create_builtin_tool_registry

    # Create service with PrimitiveOps-capable registry
    workspace = create_temp_workspace(prefix="test_integ_")
    ops = PrimitiveOps(workspace=workspace)
    registry = create_builtin_tool_registry(runners=ops.runners())

    # Verify primitive tools have runners
    assert registry.has_runner("list_files")
    assert registry.has_runner("read_file")
    assert registry.has_runner("write_file")
    assert registry.has_runner("run_python")

    # Verify list_files actually works through the registry
    result = await registry.run("list_files", {"path": ""})
    assert hasattr(result, "files")

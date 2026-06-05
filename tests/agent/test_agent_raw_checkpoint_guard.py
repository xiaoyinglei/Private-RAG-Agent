from __future__ import annotations

from pathlib import Path

import pytest
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel

from rag.agent.core.checkpointing import agent_checkpoint_serde
from rag.agent.core.context import AgentRunConfig, RuntimeRegistry
from rag.agent.core.definition import AgentDefinition, ToolPolicy
from rag.agent.graphs.base import build_agent_graph
from rag.agent.memory.models import ExternalizedToolOutput, MemoryPolicy
from rag.agent.memory.store import WorkspaceMemoryStore
from rag.agent.primitive_ops import (
    CandidateHeaderRow,
    RunPythonOutput,
    StructuredProbeInput,
    StructuredProbeOutput,
    StructuredTableProbe,
)
from rag.agent.state import AgentState, ToolCallPlan
from rag.agent.tools.registry import ToolRegistry
from rag.agent.tools.spec import ToolError, ToolPermissions, ToolSpec
from rag.agent.workspace import WorkspaceRuntime
from rag.schema.query import RetrievalSignals
from rag.schema.runtime import AccessPolicy

RAW_SENTINEL = "RAW_SENTINEL_SHOULD_NOT_REACH_CHECKPOINT"


def _workspace(tmp_path: Path) -> WorkspaceRuntime:
    workspace = WorkspaceRuntime(root=tmp_path / "workspace", is_temporary=True)
    workspace.initialize()
    return workspace


def _config(run_id: str) -> AgentRunConfig:
    return AgentRunConfig(
        run_id=run_id,
        thread_id=run_id,
        budget_total=1000,
        max_depth=2,
        access_policy=AccessPolicy.default(),
        tool_policy=ToolPolicy(max_parallel_calls=1),
        memory_policy=MemoryPolicy(max_tool_output_chars=200),
    )


def _state(run_config: AgentRunConfig, call: ToolCallPlan) -> AgentState:
    return {
        "messages": [],
        "evidence": [],
        "citations": [],
        "tool_results": [],
        "task": "Inspect workbook",
        "retrieval_signals": RetrievalSignals(),
        "retrieval_signals_debug": None,
        "run_config": run_config,
        "iteration": 0,
        "status": "running",
        "decision_reason": None,
        "stop_reason": None,
        "needs_user_input": None,
        "pending_tool_calls": [call],
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
        "goal_spec": None,
        "goal_requirements": [],
        "satisfied_requirements": [],
        "open_gaps": [],
        "evidence_refs": [],
        "answer_candidates": [],
        "computation_results": [],
        "structured_observations": [],
        "context_units": [],
        "context_bindings": [],
        "locators": [],
        "asset_refs": [],
        "conflicts": [],
        "no_progress_count": 0,
        "satisfaction_report": None,
        "controller_next": None,
        "agent_plan": None,
        "plan_events": [],
        "memory_refs": [],
        "memory_budget": None,
        "memory_warnings": [],
    }


def _definition(allowed_tools: list[str]) -> AgentDefinition:
    return AgentDefinition(
        agent_type="guard-test",
        description="Guard test",
        system_prompt="Guard test",
        allowed_tools=allowed_tools,
        max_iterations=2,
    )


@pytest.mark.anyio
async def test_graph_execute_checkpoint_externalizes_large_ok_output(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    run_config = _config("guard-ok")
    RuntimeRegistry.remove(run_config.run_id)
    handles = RuntimeRegistry.get_or_create(run_config)
    handles.memory_store = WorkspaceMemoryStore(workspace=workspace)
    registry = ToolRegistry()
    spec = ToolSpec(
        name="structured_probe",
        description="Probe",
        input_model=StructuredProbeInput,
        output_model=StructuredProbeOutput,
        error_model=ToolError,
        permissions=ToolPermissions(read_fs=True),
        timeout_seconds=1.0,
    )

    def runner(_payload: StructuredProbeInput) -> StructuredProbeOutput:
        return StructuredProbeOutput(
            path="input_files/book.xlsx",
            tables=[
                StructuredTableProbe(
                    table_index=0,
                    name="Sheet1",
                    used_range="A1:B2",
                    row_count=2,
                    column_count=2,
                    sample_rows=[["city", "sales"], ["beijing", RAW_SENTINEL * 30]],
                    candidate_header_rows=[
                        CandidateHeaderRow(row_index=1, confidence=0.95, reason="labels")
                    ],
                    data_start_row=2,
                )
            ],
        )

    registry.register(spec, runner=runner)
    checkpointer = MemorySaver(serde=agent_checkpoint_serde())
    graph = build_agent_graph(
        definition=_definition(["structured_probe"]),
        tool_registry=registry,
        checkpointer=checkpointer,
    )
    call = ToolCallPlan.create("structured_probe", {"path": "input_files/book.xlsx"})

    result = await graph.ainvoke(
        _state(run_config, call),
        config={"configurable": {"thread_id": run_config.thread_id}},
    )

    dumped = str(result)
    assert RAW_SENTINEL not in dumped
    [tool_result] = result["tool_results"]
    assert isinstance(tool_result.output, ExternalizedToolOutput)
    assert "Sheet1" in result["structured_observations"][0].locators[0]["table_name"]
    resolved = handles.memory_store.resolve(tool_result.output.ref)
    assert RAW_SENTINEL in resolved.payload.model_dump_json()
    RuntimeRegistry.remove(run_config.run_id)


class _FailInput(BaseModel):
    script: str


@pytest.mark.anyio
async def test_graph_execute_checkpoint_externalizes_large_error_detail(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    run_config = _config("guard-error")
    RuntimeRegistry.remove(run_config.run_id)
    handles = RuntimeRegistry.get_or_create(run_config)
    handles.memory_store = WorkspaceMemoryStore(workspace=workspace)
    registry = ToolRegistry()
    spec = ToolSpec(
        name="run_python",
        description="Run python",
        input_model=_FailInput,
        output_model=RunPythonOutput,
        error_model=ToolError,
        permissions=ToolPermissions(execute_code=True),
        timeout_seconds=1.0,
    )

    def runner(_payload: _FailInput) -> RunPythonOutput:
        return RunPythonOutput(
            ok=False,
            exit_code=1,
            stdout=RAW_SENTINEL * 50,
            stderr="failed",
            stdout_truncated=False,
            stderr_truncated=False,
            duration_ms=1.0,
            generated_files=[],
        )

    registry.register(spec, runner=runner)
    checkpointer = MemorySaver(serde=agent_checkpoint_serde())
    graph = build_agent_graph(
        definition=_definition(["run_python"]),
        tool_registry=registry,
        checkpointer=checkpointer,
    )
    call = ToolCallPlan.create("run_python", {"script": "scratch/fail.py"})
    state = _state(run_config, call)
    state["approved_tool_call_ids"] = [call.tool_call_id]

    result = await graph.ainvoke(
        state,
        config={"configurable": {"thread_id": run_config.thread_id}},
    )

    dumped = str(result)
    assert RAW_SENTINEL not in dumped
    [tool_result] = result["tool_results"]
    assert tool_result.status == "error"
    assert tool_result.error is not None
    assert "externalized_ref" in tool_result.error.detail
    ref = next(ref for ref in result["memory_refs"] if ref.ref_id == tool_result.error.detail["externalized_ref"])
    resolved = handles.memory_store.resolve(ref)
    assert RAW_SENTINEL in resolved.payload.model_dump_json()
    RuntimeRegistry.remove(run_config.run_id)

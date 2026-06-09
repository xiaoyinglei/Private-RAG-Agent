from __future__ import annotations

from pathlib import Path

from langchain_core.messages import HumanMessage

from rag.agent.core.context import AgentRunConfig, RunRegistry
from rag.agent.core.definition import AgentDefinition
from rag.agent.goal_runtime import (
    AnswerCandidate,
    ContextUnit,
    GoalConflict,
    GoalGap,
    StructuredObservation,
)
from rag.agent.graphs.nodes.goal_runtime import extract_obs_legacy
from rag.agent.memory.compactor import MemoryCompactor
from rag.agent.memory.models import (
    ExternalizedToolOutput,
    MemoryPolicy,
    MessageBatchPayload,
    StateChannelReplacement,
)
from rag.agent.memory.store import WorkspaceMemoryStore
from rag.agent.planning import AgentPlan, PlanEvent, PlanStep
from rag.agent.primitive_ops import (
    CandidateHeaderRow,
    FileInfo,
    ListFilesOutput,
    ReadFileOutput,
    RunPythonOutput,
    StructuredProbeOutput,
    StructuredTableProbe,
)
from rag.agent.service import AgentService
from rag.agent.state import AgentState, _merge_keyed_items, _merge_messages
from rag.agent.tools.asset_tools import AssetAnalyzeOutput
from rag.agent.tools.registry import ToolRegistry
from rag.agent.tools.spec import ToolResult
from rag.agent.workspace import WorkspaceRuntime
from rag.schema.query import RetrievalSignals
from rag.schema.runtime import AccessPolicy


class FailingMemoryStore:
    def write_tool_output(self, *_args: object, **_kwargs: object) -> object:
        raise RuntimeError("disk full")


def _workspace(tmp_path: Path) -> WorkspaceRuntime:
    workspace = WorkspaceRuntime(root=tmp_path / "workspace", is_temporary=True)
    workspace.initialize()
    return workspace


def _state(run_config: AgentRunConfig, tool_result: ToolResult) -> AgentState:
    return {
        "messages": [],
        "evidence": [],
        "citations": [],
        "tool_results": [tool_result],
        "task": "Probe the workbook",
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


def _config(run_id: str, policy: MemoryPolicy) -> AgentRunConfig:
    return AgentRunConfig(
        run_id=run_id,
        thread_id=run_id,
        budget_total=1000,
        max_depth=2,
        access_policy=AccessPolicy.default(),
        memory_policy=policy,
    )


def test_messages_reducer_supports_replacement() -> None:
    old_messages = [
        HumanMessage(content="old 1", id="old-1"),
        HumanMessage(content="old 2", id="old-2"),
    ]
    replacement = StateChannelReplacement(
        items=[HumanMessage(content="tail", id="tail")]
    )

    merged = _merge_messages(old_messages, [replacement])

    assert [message.id for message in merged] == ["tail"]


def test_initial_state_compacts_messages_to_summary_and_memory_ref(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    store = WorkspaceMemoryStore(workspace=workspace)
    policy = MemoryPolicy(message_compaction_min_count=4, max_message_tail_count=2)
    service = AgentService(
        definition=AgentDefinition(
            agent_type="research",
            description="Research",
            system_prompt="Research",
            allowed_tools=[],
        ),
        tool_registry=ToolRegistry(),
    )
    messages = [
        HumanMessage(content=f"old message {index}", id=f"msg-{index}")
        for index in range(6)
    ]

    state = service.initial_state_from_config(
        task="summarize chat",
        run_config=_config("message-compact", policy),
        messages=messages,
        memory_store=store,
    )

    assert [message.id for message in state["messages"]] == ["msg-4", "msg-5"]
    assert state["working_summary"] is not None
    assert state["working_summary"].covered_message_ids == ["msg-0", "msg-1", "msg-2", "msg-3"]
    assert len(state["memory_refs"]) == 1
    resolved = store.resolve(state["memory_refs"][0])
    assert isinstance(resolved.payload, MessageBatchPayload)
    assert [message.id for message in resolved.payload.messages] == ["msg-0", "msg-1", "msg-2", "msg-3"]
    assert "old message 0" not in "".join(message.content for message in state["messages"])
    RunRegistry.remove("message-compact")


def test_initial_state_compacts_messages_without_explicit_ids(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    store = WorkspaceMemoryStore(workspace=workspace)
    policy = MemoryPolicy(message_compaction_min_count=4, max_message_tail_count=2)
    service = AgentService(
        definition=AgentDefinition(
            agent_type="research",
            description="Research",
            system_prompt="Research",
            allowed_tools=[],
        ),
        tool_registry=ToolRegistry(),
    )
    messages = [HumanMessage(content=f"history {index}") for index in range(6)]

    state = service.initial_state_from_config(
        task="summarize chat",
        run_config=_config("message-compact-no-id", policy),
        messages=messages,
        memory_store=store,
    )

    assert [message.content for message in state["messages"]] == ["history 4", "history 5"]
    assert state["working_summary"] is not None
    assert len(state["working_summary"].covered_message_ids) == 4
    assert len(state["memory_refs"]) == 1
    resolved = store.resolve(state["memory_refs"][0])
    assert isinstance(resolved.payload, MessageBatchPayload)
    assert [message.content for message in resolved.payload.messages] == [
        "history 0",
        "history 1",
        "history 2",
        "history 3",
    ]
    RunRegistry.remove("message-compact-no-id")


def test_message_compaction_failure_keeps_tail_without_large_history() -> None:
    policy = MemoryPolicy(message_compaction_min_count=4, max_message_tail_count=2)
    service = AgentService(
        definition=AgentDefinition(
            agent_type="research",
            description="Research",
            system_prompt="Research",
            allowed_tools=[],
        ),
        tool_registry=ToolRegistry(),
    )
    messages = [
        HumanMessage(content=f"OLD_RAW_HISTORY_{index} " * 100, id=f"msg-{index}")
        if index < 4
        else HumanMessage(content=f"tail message {index}", id=f"msg-{index}")
        for index in range(6)
    ]

    state = service.initial_state_from_config(
        task="summarize chat",
        run_config=_config("message-compact-failure", policy),
        messages=messages,
        memory_store=FailingMemoryStore(),
    )

    assert [message.id for message in state["messages"]] == ["msg-4", "msg-5"]
    assert "message_compaction_failed" in state["memory_warnings"]
    assert "msg-0" in state["working_summary"].covered_message_ids
    assert "OLD_RAW_HISTORY" not in "".join(message.content for message in state["messages"])
    RunRegistry.remove("message-compact-failure")


def test_compaction_runs_after_structured_observation_extraction(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    policy = MemoryPolicy(max_tool_output_chars=200)
    run_config = _config("memory-post-reducer", policy)
    RunRegistry.remove(run_config.run_id)
    handles = RunRegistry.get_or_create(run_config)
    handles.memory_store = WorkspaceMemoryStore(workspace=workspace)
    output = StructuredProbeOutput(
        path="input_files/book.xlsx",
        file_kind="binary",
        tables=[
            StructuredTableProbe(
                table_index=0,
                name="Sheet1",
                used_range="A1:C10",
                row_count=10,
                column_count=3,
                sample_rows=[["城市", "销量", "备注"], ["北京", 12, "x" * 500]],
                candidate_header_rows=[
                    CandidateHeaderRow(
                        row_index=1,
                        confidence=0.92,
                        reason="label-like row followed by data",
                    )
                ],
                data_start_row=2,
            )
        ],
    )
    result = ToolResult(
        tool_call_id="tc-probe",
        tool_name="structured_probe",
        status="ok",
        output=output,
        latency_ms=0,
    )

    update = extract_obs_legacy(_state(run_config, result))

    [observation] = update["structured_observations"]
    [tool_replacement] = update["tool_results"]
    assert isinstance(tool_replacement, StateChannelReplacement)
    [compacted_result] = tool_replacement.items
    assert isinstance(observation, StructuredObservation)
    assert observation.tool_call_id == "tc-probe"
    assert any(locator.get("table_name") == "Sheet1" for locator in observation.locators)
    assert isinstance(compacted_result.output, ExternalizedToolOutput)
    assert compacted_result.output.original_output_model == "rag.agent.primitive_ops.StructuredProbeOutput"
    assert "Sheet1" in compacted_result.output.summary
    resolved = handles.memory_store.resolve(compacted_result.output.ref)
    assert resolved.payload == output
    RunRegistry.remove(run_config.run_id)


def test_per_tool_summaries_preserve_key_fields() -> None:
    compactor = MemoryCompactor(policy=MemoryPolicy())
    cases = [
        (
            ToolResult(
                tool_call_id="tc-run",
                tool_name="run_python",
                status="ok",
                output=RunPythonOutput(
                    ok=True,
                    exit_code=0,
                    stdout="answer\n" + ("x" * 200),
                    stderr="",
                    stdout_truncated=True,
                    stderr_truncated=False,
                    duration_ms=12.5,
                    generated_files=["reports/out.csv"],
                ),
                latency_ms=0,
            ),
            ("ok=True", "exit_code=0", "reports/out.csv", "stdout_truncated=True"),
        ),
        (
            ToolResult(
                tool_call_id="tc-probe",
                tool_name="structured_probe",
                status="ok",
                output=StructuredProbeOutput(
                    path="input_files/book.xlsx",
                    tables=[
                        StructuredTableProbe(
                            table_index=0,
                            name="Sheet1",
                            used_range="A1:B9",
                            row_count=9,
                            column_count=2,
                            candidate_header_rows=[
                                CandidateHeaderRow(row_index=3, confidence=0.8, reason="labels")
                            ],
                            data_start_row=4,
                        )
                    ],
                ),
                latency_ms=0,
            ),
            ("input_files/book.xlsx", "Sheet1", "header_row=3", "data_start_row=4"),
        ),
        (
            ToolResult(
                tool_call_id="tc-list",
                tool_name="list_files",
                status="ok",
                output=ListFilesOutput(
                    files=[
                        FileInfo(
                            name="sales.csv",
                            path="input_files/sales.csv",
                            size=10,
                            is_dir=False,
                            modified_at=1.0,
                            capabilities=["read_file"],
                        )
                    ]
                ),
                latency_ms=0,
            ),
            ("files=1", "input_files/sales.csv", "read_file"),
        ),
        (
            ToolResult(
                tool_call_id="tc-read",
                tool_name="read_file",
                status="ok",
                output=ReadFileOutput(
                    path="input_files/readme.txt",
                    content="important content" + ("x" * 200),
                    truncated=True,
                    size_bytes=999,
                ),
                latency_ms=0,
            ),
            ("input_files/readme.txt", "size_bytes=999", "truncated=True"),
        ),
        (
            ToolResult(
                tool_call_id="tc-analyze",
                tool_name="asset_analyze",
                status="ok",
                output=AssetAnalyzeOutput(
                    asset_id=7,
                    operation="dataframe_sql",
                    columns=["city", "sales"],
                    rows=[["北京", "12"]],
                    raw_row_count=1,
                    elapsed_ms=5.0,
                    truncated=False,
                    query="select city, sales from table",
                    markdown="| city | sales |\n| 北京 | 12 |",
                ),
                latency_ms=0,
            ),
            ("asset_id=7", "dataframe_sql", "rows=1", "city"),
        ),
    ]

    for result, expected_parts in cases:
        summary = compactor.summarize_tool_result(result)
        for expected in expected_parts:
            assert expected in summary


def test_state_channel_caps_preserve_active_plan_gaps_conflicts_and_traceable_refs() -> None:
    policy = MemoryPolicy(
        max_structured_observations=2,
        max_context_units=2,
        max_answer_candidates=2,
    )
    compactor = MemoryCompactor(policy=policy)
    plan = AgentPlan(
        objective="answer",
        active_step_id="active",
        steps=[PlanStep(step_id="active", title="Current step", status="in_progress")],
    )
    state: dict[str, object] = {
        "structured_observations": [
            StructuredObservation(
                tool_call_id=f"tc-old-{index}",
                tool_name="search",
                status="ok",
                raw_result_ref=f"tc-old-{index}",
            )
            for index in range(3)
        ],
        "context_units": [
            ContextUnit(unit_id=f"unit-old-{index}", unit_type="file")
            for index in range(3)
        ],
        "answer_candidates": [
            AnswerCandidate(text=f"old answer {index}", source_tool_call_id=f"tc-old-{index}")
            for index in range(3)
        ],
    }
    update: dict[str, object] = {
        "agent_plan": plan,
        "open_gaps": [GoalGap(gap_id="evidence", gap_type="evidence", description="needs refs")],
        "conflicts": [
            GoalConflict(conflict_id="conflict-1", description="conflicting evidence")
        ],
        "structured_observations": [
            StructuredObservation(
                tool_call_id="tc-new",
                tool_name="structured_probe",
                status="ok",
                raw_result_ref="tc-new",
            )
        ],
        "context_units": [ContextUnit(unit_id="unit-new", unit_type="table")],
        "answer_candidates": [AnswerCandidate(text="new answer", source_tool_call_id="tc-new")],
    }

    compacted = compactor.compact_update(state, update)

    assert compacted["agent_plan"] == plan
    assert compacted["open_gaps"] == update["open_gaps"]
    assert compacted["conflicts"] == update["conflicts"]
    replacement = compacted["structured_observations"][0]
    assert isinstance(replacement, StateChannelReplacement)
    merged = _merge_keyed_items(
        state["structured_observations"],
        compacted["structured_observations"],
    )
    assert [item.tool_call_id for item in merged] == ["tc-old-2", "tc-new"]


def test_state_retention_pins_active_step_items_and_audits_evictions() -> None:
    policy = MemoryPolicy(
        max_structured_observations=2,
        max_context_units=2,
        max_plan_events=2,
    )
    compactor = MemoryCompactor(policy=policy)
    plan = AgentPlan(
        objective="answer",
        active_step_id="step_active",
        steps=[
            PlanStep(
                step_id="step_active",
                title="Active step",
                status="in_progress",
                tool_call_ids=["tc-pin"],
            )
        ],
    )
    state: dict[str, object] = {
        "agent_plan": plan,
        "structured_observations": [
            StructuredObservation(
                tool_call_id="tc-pin",
                tool_name="search",
                status="ok",
                raw_result_ref="tc-pin",
            ),
            StructuredObservation(
                tool_call_id="tc-old",
                tool_name="search",
                status="ok",
                raw_result_ref="tc-old",
            ),
        ],
        "context_units": [
            ContextUnit(
                unit_id="unit-pin",
                unit_type="file",
                content_ref="tc-pin",
            ),
            ContextUnit(unit_id="unit-old", unit_type="file"),
        ],
        "plan_events": [
            PlanEvent(
                event_id=f"plan_event_old_{index}",
                event_type="llm_update",
                plan_revision=index,
                message=f"old {index}",
            )
            for index in range(3)
        ],
    }
    update: dict[str, object] = {
        "structured_observations": [
            StructuredObservation(
                tool_call_id="tc-new",
                tool_name="search",
                status="ok",
                raw_result_ref="tc-new",
            )
        ],
        "context_units": [ContextUnit(unit_id="unit-new", unit_type="file")],
        "plan_events": [
            PlanEvent(
                event_id="plan_event_new",
                event_type="observation_progress",
                plan_revision=4,
                message="new",
            )
        ],
    }

    compacted = compactor.compact_update(state, update)

    observations = _merge_keyed_items(
        state["structured_observations"],
        compacted["structured_observations"],
    )
    assert [item.tool_call_id for item in observations] == ["tc-pin", "tc-new"]
    units = _merge_keyed_items(state["context_units"], compacted["context_units"])
    assert [item.unit_id for item in units] == ["unit-pin", "unit-new"]
    budget = compacted["memory_budget"]
    assert budget.pinned_item_count >= 2
    assert budget.used_channel_counts["structured_observations"] == 2
    assert any(
        item.channel == "structured_observations" and item.key == "tool_call_id:tc-old"
        for item in budget.evicted_items
    )
    assert any(item.channel == "plan_events" for item in budget.evicted_items)


def test_compaction_failure_externalizes_unavailable_without_raw_state_growth() -> None:
    policy = MemoryPolicy(max_tool_output_chars=100)
    run_config = _config("memory-failure", policy)
    RunRegistry.remove(run_config.run_id)
    handles = RunRegistry.get_or_create(run_config)
    handles.memory_store = FailingMemoryStore()
    output = RunPythonOutput(
        ok=True,
        exit_code=0,
        stdout="raw stdout " * 200,
        stderr="",
        stdout_truncated=False,
        stderr_truncated=False,
        duration_ms=1.0,
        generated_files=[],
    )
    result = ToolResult(
        tool_call_id="tc-run",
        tool_name="run_python",
        status="ok",
        output=output,
        latency_ms=0,
    )

    update = extract_obs_legacy(_state(run_config, result))

    [tool_replacement] = update["tool_results"]
    assert isinstance(tool_replacement, StateChannelReplacement)
    [compacted_result] = tool_replacement.items
    assert isinstance(compacted_result.output, ExternalizedToolOutput)
    assert compacted_result.output.status == "unavailable"
    assert "memory_compaction_failed" in compacted_result.output.warnings
    assert "raw stdout raw stdout" not in compacted_result.model_dump_json()
    assert "memory_compaction_failed" in update["memory_warnings"]
    RunRegistry.remove(run_config.run_id)


def test_checkpoint_guard_sanitizes_large_state_payloads_outside_tool_results() -> None:
    raw = "RAW_STATE_PAYLOAD_SHOULD_NOT_REACH_CHECKPOINT " * 80
    compactor = MemoryCompactor(policy=MemoryPolicy(max_tool_output_chars=120))

    compacted = compactor.compact_update(
        {},
        {
            "context_units": [
                ContextUnit(
                    unit_id="unit-raw",
                    unit_type="file",
                    preview=raw,
                )
            ],
            "answer_candidates": [AnswerCandidate(text=raw, source_tool_call_id="tc-raw")],
        },
    )

    assert "RAW_STATE_PAYLOAD_SHOULD_NOT_REACH_CHECKPOINT" not in str(compacted)
    assert "raw_checkpoint_guard_sanitized" in compacted["memory_warnings"]

from __future__ import annotations

from agent_runtime.planning import (
    AgentPlan,
    GoalCommitment,
    PlanStep,
    PlanStepPatch,
    PlanTracker,
    PlanUpdate,
)
from rag.agent.core.observations import StructuredObservation
from rag.agent.core.turn_contracts import ToolCallPlan


def test_plan_types_have_public_runtime_ownership() -> None:
    assert AgentPlan.__module__ == "agent_runtime.planning"
    assert PlanStep.__module__ == "agent_runtime.planning"
    assert PlanUpdate.__module__ == "agent_runtime.planning"


def test_initialize_plan_is_task_based_without_tool_routing() -> None:
    plan, events = PlanTracker().initialize_task(
        task="Summarize the workspace files.",
    )

    assert plan.objective == "Summarize the workspace files."
    assert plan.active_step_id == "step_task"
    assert plan.steps == [
        PlanStep(
            step_id="step_task",
            title="Work on the current task.",
        )
    ]
    assert events[0].event_type == "initialized"


def test_plan_update_is_bounded_and_filters_unsupported_tools() -> None:
    plan = AgentPlan(
        objective="Answer with evidence.",
        active_step_id="step_task",
        steps=[PlanStep(step_id="step_task", title="Work")],
    )
    update = PlanUpdate(
        mode="replace",
        steps=[
            PlanStep(
                step_id="step_discover",
                title="Discover workspace assets",
                expected_tool_names=["list_files", "unsupported_delete"],
                notes="x" * 1200,
            ),
            PlanStep(step_id="step_probe", title="Probe structured files"),
            PlanStep(step_id="step_extra", title="This step should be trimmed"),
        ],
        active_step_id="step_discover",
    )

    updated, events = PlanTracker(max_steps=2).apply_advisory_update(
        plan,
        update,
        allowed_tool_names=frozenset({"list_files", "structured_probe"}),
    )

    assert [step.step_id for step in updated.steps] == [
        "step_discover",
        "step_probe",
    ]
    assert updated.steps[0].expected_tool_names == ["list_files"]
    assert updated.steps[0].notes is not None
    assert len(updated.steps[0].notes) <= 500
    assert "unsupported_tool_names" in events[0].warnings
    assert "steps_truncated" in events[0].warnings


def test_advisory_update_and_decision_progress_track_tool_call() -> None:
    call = ToolCallPlan(
        tool_call_id="tc_list",
        tool_name="list_files",
        arguments={"path": ""},
    )
    plan = AgentPlan(
        objective="Inspect files before answering.",
        active_step_id="step_discover",
        steps=[PlanStep(step_id="step_discover", title="Discover files")],
    )

    updated, _ = PlanTracker().apply_advisory_update(
        plan,
        PlanUpdate(
            mode="patch",
            step_updates=[
                PlanStepPatch(
                    step_id="step_discover",
                    status="in_progress",
                    expected_tool_names=["list_files"],
                )
            ],
        ),
        allowed_tool_names=frozenset({"list_files"}),
    )
    updated, events = PlanTracker().record_decision_progress(
        updated,
        tool_call_ids=[call.tool_call_id],
        tool_names=[call.tool_name],
    )

    assert updated.steps[0].status == "in_progress"
    assert updated.steps[0].tool_call_ids == ["tc_list"]
    assert events[-1].event_type == "decision_progress"


def test_unexpected_tool_cannot_complete_a_verification_step() -> None:
    plan = AgentPlan(
        objective="Implement and verify.",
        active_step_id="step_verify",
        steps=[
            PlanStep(
                step_id="step_verify",
                title="Run focused tests and verify green",
                status="in_progress",
                expected_tool_names=["run_command"],
            )
        ],
    )
    tracker = PlanTracker()

    decided, decision_events = tracker.record_decision_progress(
        plan,
        tool_call_ids=["tc-search"],
        tool_names=["search_text"],
    )
    observed, observation_events = tracker.record_observation_progress(
        decided,
        observations=[
            StructuredObservation(
                tool_call_id="tc-search",
                tool_name="search_text",
                status="ok",
                raw_result_ref="tc-search",
            )
        ],
    )

    assert decided.steps[0].expected_tool_names == ["run_command"]
    assert decided.steps[0].tool_call_ids == []
    assert decision_events == []
    assert observed is not None
    assert observed.steps[0].status == "in_progress"
    assert observed.status == "needs_replan"
    assert observation_events[-1].warnings == ["observation_unbound"]


def test_llm_update_cannot_claim_plan_or_step_completion() -> None:
    goal_id = "a" * 64
    commitment = GoalCommitment(
        commitment_id="goal_001",
        requirement="Verify evidence before completion.",
    )
    plan = AgentPlan(
        goal_id=goal_id,
        goal_commitments=[commitment],
        objective="Verify evidence before completion.",
        active_step_id="step_verify",
        steps=[
            PlanStep(
                step_id="step_verify",
                title="Verify evidence",
                status="in_progress",
                goal_commitment_ids=[commitment.commitment_id],
            )
        ],
    )

    updated, events = PlanTracker().apply_advisory_update(
        plan,
        PlanUpdate(
            objective="Replace the original goal with unrelated work.",
            status="complete",
            step_updates=[
                PlanStepPatch(
                    step_id="step_verify",
                    status="completed",
                )
            ],
        ),
        allowed_tool_names=frozenset(),
    )

    assert updated.status == "active"
    assert updated.goal_id == goal_id
    assert updated.goal_commitments == [commitment]
    assert updated.objective == "Verify evidence before completion."
    assert updated.steps[0].status == "in_progress"
    assert "llm_completion_ignored" in events[0].warnings
    assert "objective_change_ignored" in events[0].warnings


def test_update_plan_cannot_claim_completion_without_tool_evidence() -> None:
    plan, _events = PlanTracker().initialize_task(
        task="Implement and verify.",
    )

    updated, events = PlanTracker().replace_from_tool(
        plan,
        target_files=["rag/agent/loop/runtime.py"],
        hypothesis="The runtime needs evidence-bound progress tracking.",
        remaining_unknowns=[],
        steps=[
            PlanStep(
                step_id="step_unverified",
                title="Claimed complete without evidence",
                status="completed",
                expected_tool_names=["apply_patch"],
            ),
            PlanStep(
                step_id="step_verified",
                title="Completed with a matching tool result",
                status="completed",
                expected_tool_names=["run_command"],
                tool_call_ids=["call-tests"],
            ),
        ],
        summary="Use objective evidence.",
    )

    assert [step.status for step in updated.steps] == [
        "pending",
        "completed",
    ]
    assert updated.active_step_id == "step_unverified"
    assert events[0].warnings == ["unverified_completion_ignored"]


def test_blocked_run_downgrades_only_unverified_completed_steps() -> None:
    plan = AgentPlan(
        objective="Implement and verify the change.",
        steps=[
            PlanStep(
                step_id="step_001",
                title="Claimed complete without evidence",
                status="completed",
            ),
            PlanStep(
                step_id="step_002",
                title="Completed with tool evidence",
                status="completed",
                tool_call_ids=["call_verified"],
            ),
            PlanStep(
                step_id="step_003",
                title="Still in progress",
                status="in_progress",
            ),
        ],
    )

    updated, events = PlanTracker().record_completion(plan, blocked=True)

    assert updated is not None
    assert updated.status == "blocked"
    assert [step.status for step in updated.steps] == [
        "blocked",
        "completed",
        "blocked",
    ]
    assert events[0].event_type == "blocked"


def test_unbound_observation_requests_replan() -> None:
    plan = AgentPlan(
        objective="Answer carefully.",
        active_step_id="step_unbound",
        steps=[
            PlanStep(
                step_id="step_unbound",
                title="Unbound step",
                status="in_progress",
            )
        ],
    )

    updated, events = PlanTracker().record_observation_progress(
        plan,
        observations=[
            StructuredObservation(
                tool_call_id="tc-other",
                tool_name="search",
                status="ok",
                raw_result_ref="tc-other",
            )
        ],
    )

    assert updated is not None
    assert updated.status == "needs_replan"
    assert events[-1].event_type == "needs_replan"


def test_tool_call_bound_observation_completes_only_matching_step() -> None:
    plan = AgentPlan(
        objective="Use selected tools.",
        active_step_id="step_bound",
        steps=[
            PlanStep(
                step_id="step_bound",
                title="Bound step",
                status="in_progress",
                tool_call_ids=["tc-target"],
            ),
            PlanStep(step_id="step_other", title="Other step"),
        ],
    )

    updated, events = PlanTracker().record_observation_progress(
        plan,
        observations=[
            StructuredObservation(
                tool_call_id="tc-target",
                tool_name="search",
                status="ok",
                raw_result_ref="tc-target",
            )
        ],
    )

    assert updated is not None
    assert [step.status for step in updated.steps] == [
        "completed",
        "pending",
    ]
    assert events[-1].event_type == "observation_progress"

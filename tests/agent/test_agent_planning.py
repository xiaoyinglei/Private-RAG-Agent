from __future__ import annotations

from rag.agent.core.observations import StructuredObservation
from rag.agent.planning import (
    AgentPlan,
    PlanStep,
    PlanStepPatch,
    PlanTracker,
    PlanUpdate,
)
from rag.agent.state import ToolCallPlan


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
        allowed_tool_names=frozenset(
            {"list_files", "structured_probe"}
        ),
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


def test_llm_update_cannot_claim_plan_or_step_completion() -> None:
    plan = AgentPlan(
        objective="Verify evidence before completion.",
        active_step_id="step_verify",
        steps=[
            PlanStep(
                step_id="step_verify",
                title="Verify evidence",
                status="in_progress",
            )
        ],
    )

    updated, events = PlanTracker().apply_advisory_update(
        plan,
        PlanUpdate(
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
    assert updated.steps[0].status == "in_progress"
    assert "llm_completion_ignored" in events[0].warnings


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

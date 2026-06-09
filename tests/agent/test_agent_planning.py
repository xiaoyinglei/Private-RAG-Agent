from __future__ import annotations

from rag.agent.goal_runtime import GoalGap, StructuredObservation
from rag.agent.graphs.nodes.llm_decide import _apply_decision
from rag.agent.planning import (
    AgentPlan,
    PlanStep,
    PlanStepPatch,
    PlanTracker,
    PlanUpdate,
)
from rag.agent.state import ThinkOutput, ToolCallPlan


def test_initialize_plan_tracks_open_gaps_without_tool_routing() -> None:
    gap = GoalGap(gap_id="answer", gap_type="answer", description="Produce an answer.")

    plan, events = PlanTracker().initialize(
        task="Summarize the workspace files.",
        open_gaps=[gap],
    )

    assert plan.objective == "Summarize the workspace files."
    assert plan.status == "active"
    assert plan.active_step_id == "step_answer"
    assert plan.steps == [
        PlanStep(
            step_id="step_answer",
            title="Produce an answer.",
            related_gap_ids=["answer"],
        )
    ]
    assert all(not step.expected_tool_names for step in plan.steps)
    assert events[0].event_type == "initialized"


def test_initialize_plan_preserves_string_gap_ids_from_restored_state() -> None:
    plan, _ = PlanTracker().initialize(
        task="Continue restored run.",
        open_gaps=["evidence"],
    )

    assert plan.steps[0].step_id == "step_evidence"
    assert plan.steps[0].related_gap_ids == ["evidence"]
    assert plan.steps[0].title == "Satisfy evidence."


def test_plan_update_bounds_steps_and_records_unsupported_tool_names() -> None:
    plan = AgentPlan(
        objective="Answer with evidence.",
        active_step_id="step_answer",
        steps=[PlanStep(step_id="step_answer", title="Answer", related_gap_ids=["answer"])],
    )
    update = PlanUpdate(
        mode="replace",
        objective="Answer with evidence.",
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

    updated, events = PlanTracker(max_steps=2).apply_llm_update(
        plan,
        update,
        allowed_tool_names=frozenset({"list_files", "structured_probe"}),
        open_gap_ids=frozenset({"answer"}),
    )

    assert [step.step_id for step in updated.steps] == ["step_discover", "step_probe"]
    assert updated.active_step_id == "step_discover"
    assert updated.steps[0].expected_tool_names == ["list_files"]
    assert updated.steps[0].notes is not None
    assert len(updated.steps[0].notes) <= 500
    assert events[0].event_type == "llm_update"
    assert "unsupported_tool_names" in events[0].warnings
    assert "steps_truncated" in events[0].warnings


def test_llm_decision_applies_plan_update_and_tracks_active_step_tool_call() -> None:
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
    decision = ThinkOutput(
        action="execute",
        tool_calls=[call],
        thought="List files before choosing a parser.",
        plan_update=PlanUpdate(
            mode="patch",
            step_updates=[
                PlanStepPatch(
                    step_id="step_discover",
                    status="in_progress",
                    expected_tool_names=["list_files"],
                    notes="Need file capabilities before parsing.",
                )
            ],
            active_step_id="step_discover",
        ),
    )

    update = _apply_decision(
        decision,
        next_iteration=0,
        current_plan=plan,
        allowed_tool_names=frozenset({"list_files"}),
        open_gap_ids=["answer"],
    )

    updated_plan = update["agent_plan"]
    assert updated_plan.steps[0].status == "in_progress"
    assert updated_plan.steps[0].tool_call_ids == ["tc_list"]
    assert updated_plan.steps[0].expected_tool_names == ["list_files"]
    assert update["plan_events"][-1].event_type == "decision_progress"


def test_llm_patch_cannot_complete_plan_or_existing_step() -> None:
    plan = AgentPlan(
        objective="Verify evidence before completion.",
        status="active",
        active_step_id="step_verify",
        steps=[
            PlanStep(
                step_id="step_verify",
                title="Verify evidence",
                status="in_progress",
            )
        ],
    )

    updated, events = PlanTracker().apply_llm_update(
        plan,
        PlanUpdate(
            mode="patch",
            status="complete",
            step_updates=[
                PlanStepPatch(
                    step_id="step_verify",
                    status="completed",
                    notes="The model considers this finished.",
                )
            ],
        ),
        allowed_tool_names=frozenset(),
        open_gap_ids=frozenset({"answer"}),
    )

    assert updated.status == "active"
    assert updated.steps[0].status == "in_progress"
    assert updated.steps[0].notes == "The model considers this finished."
    assert "llm_completion_ignored" in events[0].warnings


def test_llm_replace_cannot_insert_completed_steps() -> None:
    plan = AgentPlan(
        objective="Continue the task.",
        active_step_id="step_existing",
        steps=[PlanStep(step_id="step_existing", title="Existing step")],
    )

    updated, events = PlanTracker().apply_llm_update(
        plan,
        PlanUpdate(
            mode="replace",
            status="complete",
            steps=[
                PlanStep(
                    step_id="step_claimed_complete",
                    title="Claimed complete by model",
                    status="completed",
                )
            ],
            active_step_id="step_claimed_complete",
        ),
        allowed_tool_names=frozenset(),
        open_gap_ids=frozenset({"answer"}),
    )

    assert updated.status == "active"
    assert updated.steps[0].status == "pending"
    assert updated.active_step_id == "step_claimed_complete"
    assert "llm_completion_ignored" in events[0].warnings


def test_observation_without_explicit_binding_does_not_complete_step() -> None:
    plan = AgentPlan(
        objective="Answer carefully.",
        active_step_id="step_unbound",
        steps=[PlanStep(step_id="step_unbound", title="Unbound step", status="in_progress")],
    )

    updated, events = PlanTracker().record_observation_progress(
        plan,
        observations=[
            StructuredObservation(
                tool_call_id="tc-other",
                tool_name="search",
                status="ok",
                raw_result_ref="tc-other",
                resolved_gaps=["answer"],
            )
        ],
        satisfied_requirement_ids=["answer"],
    )

    assert updated is not None
    assert updated.status == "needs_replan"
    assert updated.steps[0].status == "in_progress"
    assert events[-1].event_type == "needs_replan"
    assert "observation_unbound" in events[-1].warnings


def test_observation_tool_call_binding_completes_only_matching_step() -> None:
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
            PlanStep(step_id="step_other", title="Other step", status="pending"),
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
                resolved_gaps=["answer"],
            )
        ],
        satisfied_requirement_ids=["answer"],
    )

    assert updated is not None
    assert updated.status == "active"
    assert [step.status for step in updated.steps] == ["completed", "pending"]
    assert events[-1].event_type == "observation_progress"

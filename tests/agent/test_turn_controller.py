from __future__ import annotations

from dataclasses import dataclass

from rag.agent.builtin.research import RESEARCH_AGENT
from rag.agent.core.context import AgentRunConfig, RunRegistry
from rag.agent.goal_runtime import GoalGap, GoalSpec, SatisfactionReport
from rag.agent.graphs.nodes import goal_runtime as graph_goal_runtime
from rag.agent.loop.controller import TurnController
from rag.agent.state import AgentState
from rag.schema.query import RetrievalSignals
from rag.schema.runtime import AccessPolicy


@dataclass
class _BindingAssessor:
    assessed: bool = False

    def assess_bindings(self, state: dict[str, object], *, context_units: list[object]) -> list[object]:
        del state, context_units
        self.assessed = True
        return []


class _Checker:
    def check(self, state: dict[str, object]) -> SatisfactionReport:
        del state
        return SatisfactionReport(
            open_gaps=[GoalGap(gap_id="answer", gap_type="answer", description="Produce an answer.")],
            reason="open_gaps",
        )


def _state(run_id: str) -> AgentState:
    config = AgentRunConfig(
        run_id=run_id,
        thread_id=run_id,
        budget_total=1000,
        max_depth=2,
        access_policy=AccessPolicy.default(),
    )
    RunRegistry.remove(run_id)
    RunRegistry.get_or_create(config)
    return {
        "run_config": config,
        "status": "running",
        "task": "answer with evidence",
        "goal_spec": GoalSpec(original_query="answer with evidence"),
        "pending_tool_calls": [],
        "context_units": [],
        "context_bindings": [],
        "tool_results": [],
        "answer_candidates": [],
        "evidence_refs": [],
        "conflicts": [],
        "no_progress_count": 0,
    }  # type: ignore[typeddict-item]


def test_turn_controller_routes_open_gap_to_model_decision_without_auto_proposal() -> None:
    assessor = _BindingAssessor()
    state = _state("turn-controller")

    update = TurnController(
        definition=RESEARCH_AGENT,
        has_tool_decision_provider=True,
        binding_assessor=assessor,
        checker=_Checker(),
    ).advance(state)

    assert update["controller_next"] == "llm_decide"
    assert "tool_action_proposals" not in update
    assert "pending_tool_calls" not in update
    assert assessor.assessed is True
    RunRegistry.remove("turn-controller")


def test_default_binding_assessor_does_not_expose_action_proposal_api() -> None:
    controller = TurnController(
        definition=RESEARCH_AGENT,
        has_tool_decision_provider=True,
    )

    assert not hasattr(controller.binding_assessor, "propose")


def test_turn_controller_does_not_route_legacy_task_dag_state() -> None:
    assessor = _BindingAssessor()
    state = _state("loop-ignores-task-dag")
    state["plan"] = {"legacy": "task_dag"}  # type: ignore[typeddict-unknown-key]

    update = TurnController(
        definition=RESEARCH_AGENT,
        has_tool_decision_provider=True,
        binding_assessor=assessor,
        checker=_Checker(),
    ).advance(state)

    assert update["controller_next"] == "llm_decide"
    assert "pending_tool_calls" not in update
    RunRegistry.remove("loop-ignores-task-dag")


def test_turn_controller_does_not_route_legacy_decomposition_hint() -> None:
    assessor = _BindingAssessor()
    state = _state("loop-ignores-decomposition")
    state["retrieval_signals"] = RetrievalSignals(allow_graph_expansion=True)
    state["route_plan_hint"] = True  # type: ignore[typeddict-unknown-key]

    update = TurnController(
        definition=RESEARCH_AGENT,
        has_tool_decision_provider=True,
        binding_assessor=assessor,
        checker=_Checker(),
    ).advance(state)

    assert update["controller_next"] == "llm_decide"
    assert "pending_tool_calls" not in update
    RunRegistry.remove("loop-ignores-decomposition")


def test_graph_controller_adapter_does_not_route_legacy_task_dag_state() -> None:
    state = _state("graph-preserves-task-dag")
    state["plan"] = {"legacy": "task_dag"}  # type: ignore[typeddict-unknown-key]

    update = graph_goal_runtime.control_turn(
        state,
        definition=RESEARCH_AGENT,
        has_tool_decision_provider=True,
    )

    assert update["controller_next"] == "llm_decide"
    assert "pending_tool_calls" not in update
    RunRegistry.remove("graph-preserves-task-dag")


def test_graph_control_turn_delegates_to_extracted_controller(monkeypatch: object) -> None:
    calls: list[AgentState] = []

    class _StubController:
        def __init__(self, **kwargs: object) -> None:
            del kwargs

        def advance(self, state: AgentState) -> dict[str, object]:
            calls.append(state)
            return {"controller_next": "pause"}

    monkeypatch.setattr(graph_goal_runtime, "TurnController", _StubController)  # type: ignore[attr-defined]
    state = _state("graph-turn-adapter")

    update = graph_goal_runtime.control_turn(
        state,
        definition=RESEARCH_AGENT,
        has_tool_decision_provider=False,
    )

    assert update == {"controller_next": "pause"}
    assert calls == [state]
    RunRegistry.remove("graph-turn-adapter")

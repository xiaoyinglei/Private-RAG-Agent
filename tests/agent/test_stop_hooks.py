from __future__ import annotations

from dataclasses import dataclass

import pytest
from pydantic import BaseModel

from rag.agent.core.goal_contract import GoalDeliverable, GoalSpec
from rag.agent.core.context import AgentRunConfig
from rag.agent.core.definition import AgentRuntimePolicy
from rag.agent.core.finalization import (
    FinishCandidateBuilder,
    FinishCandidateBuildError,
)
from rag.agent.core.output_finalizer import OutputValidationExhaustedError
from rag.agent.loop.state import ModelTurnDraft, create_loop_state
from rag.agent.loop.stop_hooks import (
    GoalContractStopHook,
    StopHookBinding,
    StopHookRunner,
    StopVerdict,
    StructuredOutputStopHook,
    build_stop_hooks,
)
from rag.schema.runtime import AccessPolicy


class _StructuredAnswer(BaseModel):
    answer: str


def _config(run_id: str = "stop-hooks") -> AgentRunConfig:
    return AgentRunConfig(
        run_id=run_id,
        thread_id=run_id,
        budget_total=100,
        max_depth=2,
        access_policy=AccessPolicy.default(),
    )


def _state():
    return create_loop_state(task="Answer carefully", run_config=_config())


@dataclass
class _StaticHook:
    verdict: StopVerdict
    calls: list[str] | None = None

    async def evaluate(self, *, state: object, candidate: str) -> StopVerdict:
        del state
        if self.calls is not None:
            self.calls.append(candidate)
        return self.verdict


class _FailingHook:
    async def evaluate(self, *, state: object, candidate: str) -> StopVerdict:
        del state, candidate
        raise RuntimeError("hook unavailable")


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("action", "accepted", "halted"),
    [
        ("accept", True, False),
        ("warn", True, False),
        ("block", False, False),
        ("halt", False, True),
    ],
)
async def test_runner_supports_all_verdict_actions(
    action: str,
    accepted: bool,
    halted: bool,
) -> None:
    state = _state()
    runner = StopHookRunner(
        hooks=[
            StopHookBinding(
                name="static",
                hook=_StaticHook(
                    StopVerdict(
                        action=action,
                        code=f"static_{action}",
                        message=f"{action} message",
                    )
                ),
                critical=True,
            )
        ],
        max_blocks=3,
    )

    outcome = await runner.evaluate(state=state, candidate="candidate")

    assert outcome.accepted is accepted
    assert outcome.halted is halted
    if action == "warn":
        assert state["finish_state"].warnings[0].code == "static_warn"
    if action == "block":
        assert state["finish_state"].feedback[0].code == "static_block"


@pytest.mark.anyio
async def test_hooks_run_in_stable_order_and_warning_does_not_block() -> None:
    calls: list[str] = []
    state = _state()
    runner = StopHookRunner(
        hooks=[
            StopHookBinding(
                name="warning",
                hook=_StaticHook(
                    StopVerdict(
                        action="warn",
                        code="warning",
                        message="advisory warning",
                    ),
                    calls=calls,
                ),
                critical=False,
            ),
            StopHookBinding(
                name="accept",
                hook=_StaticHook(
                    StopVerdict(action="accept", code="accepted"),
                    calls=calls,
                ),
                critical=True,
            ),
        ],
        max_blocks=3,
    )

    outcome = await runner.evaluate(state=state, candidate="candidate")

    assert outcome.accepted is True
    assert calls == ["candidate", "candidate"]
    assert [item.code for item in state["finish_state"].warnings] == ["warning"]


@pytest.mark.anyio
async def test_repeated_equivalent_block_halts_at_configured_limit() -> None:
    state = _state()
    runner = StopHookRunner(
        hooks=[
            StopHookBinding(
                name="critic",
                hook=_StaticHook(
                    StopVerdict(
                        action="block",
                        code="missing_citation",
                        message="Add a citation.",
                    )
                ),
                critical=True,
            )
        ],
        max_blocks=2,
    )

    first = await runner.evaluate(state=state, candidate="draft one")
    second = await runner.evaluate(state=state, candidate="draft two")

    assert first.blocked is True
    assert second.halted is True
    assert second.code == "stop_hook_block_limit"
    assert state["finish_state"].feedback[0].occurrences == 2


@pytest.mark.anyio
async def test_advisory_failure_warns_but_critical_failure_halts() -> None:
    advisory_state = _state()
    advisory = StopHookRunner(
        hooks=[
            StopHookBinding(
                name="advisory",
                hook=_FailingHook(),
                critical=False,
            )
        ],
        max_blocks=2,
    )
    critical_state = _state()
    critical = StopHookRunner(
        hooks=[
            StopHookBinding(
                name="critical",
                hook=_FailingHook(),
                critical=True,
            )
        ],
        max_blocks=2,
    )

    advisory_outcome = await advisory.evaluate(
        state=advisory_state,
        candidate="candidate",
    )
    critical_outcome = await critical.evaluate(
        state=critical_state,
        candidate="candidate",
    )

    assert advisory_outcome.accepted is True
    assert advisory_state["finish_state"].warnings[0].code == "advisory_failed"
    assert critical_outcome.halted is True
    assert critical_outcome.code == "critical_failed"


class _StructuredFinalizer:
    async def finalize(
        self,
        *,
        definition: AgentRuntimePolicy,
        state: object,
        candidate_text: str,
    ) -> BaseModel:
        del definition, state
        return _StructuredAnswer(answer=candidate_text.upper())


class _ExhaustedFinalizer:
    async def finalize(
        self,
        *,
        definition: AgentRuntimePolicy,
        state: object,
        candidate_text: str,
    ) -> BaseModel:
        del definition, state, candidate_text
        raise OutputValidationExhaustedError(
            attempts=2,
            validation_errors=[
                {
                    "location": ["answer"],
                    "message": "field required",
                    "type": "missing",
                }
            ],
        )


@pytest.mark.anyio
async def test_structured_output_hook_is_critical_and_returns_validated_output() -> None:
    definition = AgentRuntimePolicy.test_factory(
        agent_type="structured",
        description="structured",
        system_prompt="Return structured output.",
        allowed_tools=[],
        output_model=_StructuredAnswer,
    )
    state = _state()
    runner = StopHookRunner(
        hooks=[
            StopHookBinding(
                name="structured_output",
                hook=StructuredOutputStopHook(
                    definition=definition,
                    finalizer=_StructuredFinalizer(),
                ),
                critical=True,
            )
        ],
        max_blocks=definition.max_stop_hook_blocks,
    )

    outcome = await runner.evaluate(state=state, candidate="done")

    assert outcome.accepted is True
    assert outcome.final_output is not None
    assert outcome.final_output.data == {"answer": "DONE"}


@pytest.mark.anyio
async def test_structured_output_exhaustion_halts_with_validation_details() -> None:
    definition = AgentRuntimePolicy.test_factory(
        agent_type="structured",
        description="structured",
        system_prompt="Return structured output.",
        allowed_tools=[],
        output_model=_StructuredAnswer,
    )
    runner = StopHookRunner(
        hooks=[
            StopHookBinding(
                name="structured_output",
                hook=StructuredOutputStopHook(
                    definition=definition,
                    finalizer=_ExhaustedFinalizer(),
                ),
                critical=True,
            )
        ],
        max_blocks=definition.max_stop_hook_blocks,
    )

    outcome = await runner.evaluate(state=_state(), candidate="invalid")

    assert outcome.halted is True
    assert outcome.code == "structured_output_invalid"
    assert outcome.detail["attempts"] == 2


@pytest.mark.anyio
async def test_explicit_goal_contract_blocks_missing_evidence_without_routing_fields() -> None:
    goal = GoalSpec(
        original_query="Answer with evidence",
        deliverables=[
            GoalDeliverable(
                deliverable_id="answer",
                kind="answer",
                acceptance_rule="non_empty_answer",
            ),
            GoalDeliverable(
                deliverable_id="evidence",
                kind="evidence",
                acceptance_rule="traceable_evidence",
            ),
        ],
    )
    state = _state()
    hook = GoalContractStopHook(goal_spec=goal)

    verdict = await hook.evaluate(state=state, candidate="Unsupported answer")

    assert verdict.action == "block"
    assert verdict.code == "goal_contract_unsatisfied"
    assert verdict.detail["unsatisfied_issue_ids"] == ["evidence"]
    assert "open_gap_ids" not in verdict.detail
    assert "open_gaps" not in state
    assert "satisfied_requirements" not in state


@pytest.mark.anyio
async def test_explicit_goal_contract_accepts_traceable_evidence() -> None:
    goal = GoalSpec(
        original_query="Answer with evidence",
        deliverables=[
            GoalDeliverable(
                deliverable_id="answer",
                kind="answer",
                acceptance_rule="non_empty_answer",
            ),
            GoalDeliverable(
                deliverable_id="evidence",
                kind="evidence",
                acceptance_rule="traceable_evidence",
            ),
        ],
    )
    state = _state()
    # PR2: evidence_refs derived from tool_results, not direct state field
    from pydantic import BaseModel

    from rag.agent.tools.spec import ToolResult

    class _EvidenceOutput(BaseModel):
        evidence_refs: list[dict[str, object]]

    state["tool_results"] = [
        ToolResult(
            tool_call_id="tc-1",
            tool_name="test",
            status="ok",
            output=_EvidenceOutput(
                evidence_refs=[{"evidence_id": "evidence-1", "citation_id": "citation-1"}],
            ),
            latency_ms=100,
        )
    ]

    verdict = await GoalContractStopHook(goal_spec=goal).evaluate(
        state=state,
        candidate="Supported answer",
    )

    assert verdict.action == "accept"
    assert verdict.code == "goal_contract_satisfied"


def test_stop_hook_factory_installs_goal_hook_only_when_explicitly_supplied() -> None:
    definition = AgentRuntimePolicy.test_factory(
        agent_type="plain",
        description="plain",
        system_prompt="Answer.",
        allowed_tools=[],
    )
    goal = GoalSpec(original_query="Answer")

    ordinary = build_stop_hooks(definition=definition)
    explicit = build_stop_hooks(definition=definition, goal_spec=goal)

    assert all(binding.name != "goal_contract" for binding in ordinary)
    assert [binding.name for binding in explicit] == ["goal_contract"]


@pytest.mark.anyio
async def test_finish_candidate_builder_rejects_missing_final_answer() -> None:
    builder = FinishCandidateBuilder()

    with pytest.raises(
        FinishCandidateBuildError,
        match="model turn is incomplete",
    ):
        await builder.build(
            ModelTurnDraft(action="finish"),
            state=_state(),
        )

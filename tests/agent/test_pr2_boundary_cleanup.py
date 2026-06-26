"""PR2 Task 10: Boundary cleanup integration tests.

Covers deprecated-field isolation, formatter routing, assembler plumbing,
and golden equivalence preservation.
"""

from __future__ import annotations

import subprocess
from unittest.mock import MagicMock

from pydantic import BaseModel

from rag.agent.core.context import AgentRunConfig
from rag.agent.core.definition import AgentRuntimePolicy
from rag.agent.core.llm_context import AgentLLMContextAssembler
from rag.agent.core.observations import ObservationBatch, StructuredObservation
from rag.agent.loop.runtime import AgentLoop
from rag.agent.loop.state import create_loop_state
from rag.agent.memory.injector import ContextBuilder
from rag.agent.memory.models import ContextSection
from rag.agent.planning import PlanTracker
from rag.agent.tools.observation import ToolExecutionObservation
from rag.agent.tools.registry import ToolRegistry
from rag.agent.tools.spec import ToolResult
from rag.schema.llm import LLMCallStage, LLMStageBudget
from rag.schema.query import EvidenceItem
from rag.schema.runtime import AccessPolicy

# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


class _CharacterTokenAccounting:
    """Char-counting token accounting for test isolation."""

    def count(self, text: str) -> int:
        return len(text)

    def clip(
        self,
        text: str,
        token_budget: int,
        *,
        add_ellipsis: bool = False,
    ) -> str:
        clipped = text[: max(token_budget, 0)]
        if add_ellipsis and len(clipped) < len(text) and token_budget >= 4:
            return clipped[: token_budget - 4].rstrip() + " ..."
        return clipped


def _token_accounting() -> _CharacterTokenAccounting:
    return _CharacterTokenAccounting()


def _minimal_run_config(run_id: str = "pr2-t10") -> AgentRunConfig:
    return AgentRunConfig(
        run_id=run_id,
        thread_id=run_id,
        budget_total=10_000,
        max_depth=3,
        access_policy=AccessPolicy.default(),
    )


def _definition() -> AgentRuntimePolicy:
    return AgentRuntimePolicy.from_legacy(
        agent_type="research",
        description="Research agent",
        system_prompt="You are a research assistant.",
        allowed_tools=["vector_search", "unknown_tool"],
    )


def _stage_budgets() -> dict[LLMCallStage, LLMStageBudget]:
    return {
        LLMCallStage.TOOL_DECISION: LLMStageBudget(
            max_input_tokens=12_000,
            max_output_tokens=2_048,
        ),
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestPR2BoundaryCleanup:
    """Integration tests for PR2 boundary cleanup (Task 10)."""

    # -- 1. _merge_observations is a no-op --------------------------------

    def test_merge_observations_is_noop(self) -> None:
        """_merge_observations must not write to any deprecated LoopState field."""
        state = create_loop_state(
            task="test noop",
            run_config=_minimal_run_config(),
        )
        # Build a batch that the OLD code would have merged into state
        batch = ObservationBatch(
            structured_observations=[
                StructuredObservation(
                    tool_call_id="tc-1",
                    tool_name="vector_search",
                    status="ok",
                    raw_result_ref="tc-1",
                ),
            ],
            evidence=[
                EvidenceItem(
                    evidence_id="ev-1",
                    doc_id=1,
                    citation_anchor="doc#1",
                    text="some evidence",
                    score=0.9,
                ),
            ],
        )
        AgentLoop._merge_observations(state, batch)

        # PR3: deprecated state fields removed from LoopState
        assert "structured_observations" not in state
        assert "evidence" not in state
        assert "citations" not in state
        assert "evidence_refs" not in state
        assert "computation_results" not in state
        assert "context_units" not in state

    # -- 2. Formatter scheduled for registered tool -----------------------

    def test_formatter_scheduled_for_registered_tools(self) -> None:
        """ContextBuilder output contains formatter content for registered tool."""
        registry = ToolRegistry()
        formatter = MagicMock()
        formatter.tool_name = "vector_search"
        formatter.format_result.return_value = ContextSection(
            name="tool_results",
            content="Vector search returned: policy requires prior approval",
            token_count=50,
            required=False,
        )
        registry.register_formatter(formatter)

        state = create_loop_state(
            task="test formatter",
            run_config=_minimal_run_config(),
        )
        state["tool_results"] = [
            ToolResult(
                tool_call_id="tc-vs",
                tool_name="vector_search",
                status="ok",
                output=_EmptyToolOutput(),
                latency_ms=100.0,
            ),
        ]

        cb = ContextBuilder(
            max_context_tokens=8_000,
            token_accounting=_token_accounting(),
            formatter_resolver=lambda name: registry.get_formatter(name),
        )
        ctx = cb.assemble_loop(definition=_definition(), state=state)
        output = ctx.as_text()

        assert "Vector search returned" in output, f"Formatter content missing from ContextBuilder output:\n{output}"

    # -- 3. Fallback for unregistered tools --------------------------------

    def test_fallback_for_unregistered_tools(self) -> None:
        """ContextBuilder uses fallback for tools without a registered formatter."""
        registry = ToolRegistry()
        formatter = MagicMock()
        formatter.tool_name = "vector_search"
        formatter.format_result.return_value = ContextSection(
            name="tool_results",
            content="Vector search result",
            token_count=10,
            required=False,
        )
        registry.register_formatter(formatter)

        state = create_loop_state(
            task="test fallback",
            run_config=_minimal_run_config(),
        )
        state["tool_results"] = [
            # Registered tool
            ToolResult(
                tool_call_id="tc-vs",
                tool_name="vector_search",
                status="ok",
                output=_EmptyToolOutput(),
                latency_ms=100.0,
            ),
            # Unregistered tool -- no formatter, should use fallback
            ToolResult(
                tool_call_id="tc-uk",
                tool_name="unknown_tool",
                status="ok",
                output=_EmptyToolOutput(),
                latency_ms=50.0,
            ),
        ]

        cb = ContextBuilder(
            max_context_tokens=8_000,
            token_accounting=_token_accounting(),
            formatter_resolver=lambda name: registry.get_formatter(name),
        )
        ctx = cb.assemble_loop(definition=_definition(), state=state)
        output = ctx.as_text()

        # Registered formatter content must be present
        assert "Vector search result" in output, f"Registered formatter content missing:\n{output}"
        # Fallback for unregistered tool must be present
        assert "unknown_tool" in output, f"Fallback for unregistered tool missing:\n{output}"
        assert "tc-uk" in output, f"tool_call_id for unregistered tool missing in fallback:\n{output}"

    # -- 4. No state["retrieval_signals"] references (PR3: field removed) ---

    def test_retrieval_signals_no_state_references(self) -> None:
        """No `state["retrieval_signals"]` references remain after PR3 removal."""
        result = subprocess.run(
            [
                "grep",
                "-rn",
                'state\\["retrieval_signals"\\]',
                "rag/agent/",
                "--include=*.py",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        output = result.stdout.strip()
        if output:
            raise AssertionError(f"state['retrieval_signals'] found in production code:\n{output}")

    # -- 5. PlanTracker with ToolExecutionObservation ----------------------

    def test_plan_tracker_with_tool_execution_observation(self) -> None:
        """PlanTracker accepts ToolExecutionObservation via record_observation_progress."""
        tracker = PlanTracker()
        plan, _events = tracker.initialize_task(task="Test observation progress")

        # Record a decision so the plan step gets the tool_call_id
        plan, _events = tracker.record_decision_progress(
            plan,
            tool_call_ids=["tc-step-1"],
            tool_names=["vector_search"],
        )

        obs = ToolExecutionObservation(
            tool_call_id="tc-step-1",
            tool_name="vector_search",
            status="ok",
        )
        plan, events = tracker.record_observation_progress(
            plan,
            observations=[obs],
        )

        assert plan is not None
        assert any(step.status == "completed" for step in plan.steps), (
            "PlanTracker should mark step completed from observation"
        )
        assert any(event.event_type == "observation_progress" for event in events), (
            "Expected observation_progress event"
        )

    # -- 6. Formatter resolver flows through assembler ---------------------

    def test_formatter_resolver_flows_through_assembler(self) -> None:
        """AgentLLMContextAssembler passes formatter_resolver to ContextBuilder."""
        registry = ToolRegistry()
        formatter = MagicMock()
        formatter.tool_name = "vector_search"
        formatter.format_result.return_value = ContextSection(
            name="tool_results",
            content="Assembler formatter: policy requires prior approval",
            token_count=50,
            required=False,
        )
        registry.register_formatter(formatter)

        state = create_loop_state(
            task="test assembler formatter",
            run_config=_minimal_run_config(),
        )
        state["tool_results"] = [
            ToolResult(
                tool_call_id="tc-vs",
                tool_name="vector_search",
                status="ok",
                output=_EmptyToolOutput(),
                latency_ms=100.0,
            ),
        ]

        assembler = AgentLLMContextAssembler(
            token_accounting=_token_accounting(),
            stage_budgets=_stage_budgets(),
            formatter_resolver=lambda name: registry.get_formatter(name),
        )
        result = assembler.assemble_loop_turn(
            definition=_definition(),
            state=state,
            budget_remaining=12_000,
        )

        # The formatter was invoked
        formatter.format_result.assert_called_once()
        # The formatter's content appears in the final prompt
        assert "Assembler formatter" in result.prompt, (
            f"Formatter content not found in assembler prompt:\n{result.prompt}"
        )

    # -- 7. Groundedness flags not written to LoopState --------------------

    def test_groundedness_flags_not_written_to_loopstate(self) -> None:
        """No groundedness flags leak into LoopState after tool execution."""
        state = create_loop_state(
            task="test groundedness",
            run_config=_minimal_run_config(),
        )

        # Simulate a tool result with groundedness info
        state["tool_results"] = [
            ToolResult(
                tool_call_id="tc-ans",
                tool_name="generate_answer",
                status="ok",
                output=_MockGroundedOutput(),
                latency_ms=100.0,
            ),
        ]

        batch = ObservationBatch()
        AgentLoop._merge_observations(state, batch)

        # The state should not have gained groundedness fields
        assert "groundedness_flag" not in state
        assert "insufficient_evidence_flag" not in state

    # -- 8. Golden equivalence still passes --------------------------------

    def test_golden_equivalence_still_passes(self) -> None:
        """Re-run the Task 5 equivalence test to confirm it still passes."""
        from tests.agent.test_pr2_context_equivalence import (  # noqa: PLC0415
            TestPR2ContextEquivalence,
        )

        runner = TestPR2ContextEquivalence()
        runner.test_old_vs_formatter_context_equivalence()
        runner.test_evidence_anchors_in_both_paths()
        runner.test_both_paths_produce_tool_results_with_tool_names()


class _MockGroundedOutput(BaseModel):
    """Minimal output model with groundedness flags for test 7."""

    groundedness_flag: bool = True
    insufficient_evidence_flag: bool = False


class _EmptyToolOutput(BaseModel):
    """Placeholder tool output that Pydantic can instantiate."""

    pass

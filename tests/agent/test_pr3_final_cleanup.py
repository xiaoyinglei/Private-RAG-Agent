from __future__ import annotations

import warnings

import pytest
from pydantic import BaseModel

from rag.agent.core.checkpointing import (
    _migrate_legacy_state,
    agent_checkpoint_serde,
)
from rag.agent.core.context import AgentRunConfig
from rag.agent.core.turn_contracts import ToolCallPlan
from rag.agent.loop.state import (
    PendingToolCall,
    ToolCallLedger,
    create_loop_state,
)
from rag.agent.tools.rag_answer_tools import RAGSearchAnswerOutput
from rag.agent.tools.spec import ToolResult
from rag.schema.query import AnswerCitation, EvidenceItem
from rag.schema.runtime import AccessPolicy

# ── Helpers ──


class _DummyOutput(BaseModel):
    result: str = "ok"


def _config() -> AgentRunConfig:
    return AgentRunConfig(
        run_id="test-pr3-cleanup",
        thread_id="test-pr3-cleanup",
        budget_total=100,
        max_depth=1,
        access_policy=AccessPolicy.default(),
    )


_DEPRECATED_FIELDS = frozenset(
    {
        "retrieval_signals",
        "retrieval_signals_debug",
        "evidence",
        "citations",
        "evidence_refs",
        "answer_candidates",
        "computation_results",
        "structured_observations",
        "context_units",
        "context_bindings",
        "locators",
        "asset_refs",
    }
)


# ── Tests ──


# ── Test 1 ──


def test_pending_single_track_roundtrip() -> None:
    """PendingToolCall v2 must serialize/deserialize through checkpoint serde."""
    serde = agent_checkpoint_serde()

    plan = ToolCallPlan.create("my_tool", {"key": "value", "count": 42})
    ptc = PendingToolCall(plan=plan, status="approved", summary="test summary")

    restored = serde.loads_typed(serde.dumps_typed(ptc))
    assert isinstance(restored, PendingToolCall)
    assert restored.tool_call_id == plan.tool_call_id
    assert restored.tool_name == "my_tool"
    assert restored.status == "approved"
    assert restored.summary == "test summary"
    assert restored.plan.arguments == {"key": "value", "count": 42}


# ── Test 2 ──


def test_tool_call_ledger_bounded_fifo() -> None:
    """ToolCallLedger must keep max_entries (128) when 130 added, preserving active calls."""
    ledger = ToolCallLedger(max_entries=128)
    active_tool_call_ids: set[str] = set()

    for i in range(130):
        plan = ToolCallPlan.create(f"tool_{i}", {"index": i})
        if i >= 125:
            active_tool_call_ids.add(plan.tool_call_id)
        ledger.append_plans([plan], turn=1)

    assert len(ledger.entries) == 130

    ledger.trim(active_tool_call_ids=active_tool_call_ids)
    assert len(ledger.entries) == 128

    # Active calls must be preserved after trimming
    remaining_ids = {e.plan.tool_call_id for e in ledger.entries}
    for active_id in active_tool_call_ids:
        assert active_id in remaining_ids, f"active call {active_id} was removed"


# ── Test 3 ──


def test_transcript_rebuild_from_ledger() -> None:
    """Rebuild native transcript from ledger, verify args match."""
    from rag.agent.core.llm_providers import _rebuild_tool_transcript

    config = _config()
    state = create_loop_state(task="test transcript", run_config=config)

    plans = [
        ToolCallPlan.create("search", {"query": "Paris", "top_k": 5}),
        ToolCallPlan.create("analyze", {"data": [1, 2, 3]}),
    ]
    state["tool_call_ledger"].append_plans(plans, turn=1)

    results = [
        ToolResult(
            tool_call_id=plans[0].tool_call_id,
            tool_name="search",
            status="ok",
            output=_DummyOutput(),
            latency_ms=0,
        ),
        ToolResult(
            tool_call_id=plans[1].tool_call_id,
            tool_name="analyze",
            status="ok",
            output=_DummyOutput(),
            latency_ms=0,
        ),
    ]
    state["tool_results"] = results

    transcript = _rebuild_tool_transcript(state)
    # 2 assistant (tool calls) + 2 tool (results) = 4 messages
    assert len(transcript) == 4

    # Assistant message 1: search tool call
    assert transcript[0].role == "assistant"
    assert transcript[0].tool_calls[0].name == "search"
    assert transcript[0].tool_calls[0].input == {"query": "Paris", "top_k": 5}
    assert transcript[0].tool_calls[0].id == plans[0].tool_call_id

    # Tool result 1
    assert transcript[1].role == "tool"
    assert transcript[1].tool_call_id == plans[0].tool_call_id

    # Assistant message 2: analyze tool call
    assert transcript[2].role == "assistant"
    assert transcript[2].tool_calls[0].name == "analyze"
    assert transcript[2].tool_calls[0].input == {"data": [1, 2, 3]}

    # Tool result 2
    assert transcript[3].role == "tool"
    assert transcript[3].tool_call_id == plans[1].tool_call_id


# ── Test 4 ──


def test_deprecated_fields_not_in_loopstate() -> None:
    """create_loop_state must not include any of the 12 deprecated flat fields."""
    config = _config()
    state = create_loop_state(task="test no deprecated", run_config=config)

    for key in _DEPRECATED_FIELDS:
        assert key not in state, f"deprecated field {key!r} must not appear in create_loop_state"


# ── Test 5 ──


def test_legacy_checkpoint_fields_dropped() -> None:
    """_migrate_legacy_state must drop all deprecated flat fields from old checkpoints."""
    config = _config()
    raw: dict[str, object] = {
        "task": "test migrate",
        "messages": [],
        "run_config": config,
        "iteration": 0,
        "status": "running",
        "pending_tool_calls": [],
        "tool_call_ledger": ToolCallLedger(),
        "tool_execution_records": {},
        "approval_request": None,
        "approval_response": None,
        "approved_tool_call_ids": [],
        "denied_tool_call_ids": [],
        "tool_results": [],
        "working_summary": None,
        "extracted_facts": [],
        "context_budget": None,
        "memory_refs": [],
        "memory_budget": None,
        "memory_warnings": [],
        "reactive_compact_used": False,
        "agent_plan": None,
        "plan_events": [],
        "stop_hook_feedback": [],
        "stop_hook_warnings": [],
        "runtime_diagnostics": [],
        "last_model_turn": None,
        "final_answer": None,
        "final_output": None,
        "output_validation_errors": [],
        "pause": None,
        "terminal": None,
        "latest_transition": None,
        "discovery_active_tools": [],
        "discovery_active_tool_iterations": {},
        "discovery_last_candidates": [],
        "discovery_last_search_query": "",
        "discovery_search_history": [],
        "discovery_pinned_tools": [],
        "active_deferred_tools": [],
        "capability_diagnostics": [],
        "file_manifest": None,
        "persistent_memories": [],
        "memory_index": "",
        # ── PR3 dropped loop_messages and tool_result_store ──
        "loop_messages": [{"type": "human", "content": "old"}],
        "tool_result_store": {"some": "value"},
        # ── PR0 deprecated flat fields to be dropped ──
        "evidence": [],
        "citations": [],
        "retrieval_signals": None,
    }

    migrated = _migrate_legacy_state(raw)
    for key in _DEPRECATED_FIELDS:
        assert key not in migrated, f"deprecated field {key!r} must be dropped by _migrate_legacy_state"
    assert "loop_messages" not in migrated
    assert "tool_result_store" not in migrated


# ── Test 6 ──


def test_compat_module_deprecation_warning() -> None:
    """Calling deprecated compat functions must trigger DeprecationWarning."""
    from rag.agent.core.context import AgentRunConfig

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DeprecationWarning)
        from rag.agent.state import create_agent_state

        create_agent_state(
            task="test",
            run_config=AgentRunConfig(
                run_id="test-warning",
                thread_id="test-warning",
                budget_total=1,
                max_depth=1,
                access_policy=AccessPolicy.default(),
            ),
        )

    deprecation_messages = [
        str(w.message)
        for w in caught
        if issubclass(w.category, DeprecationWarning) and "rag.agent.state is deprecated" in str(w.message)
    ]
    assert len(deprecation_messages) > 0, "Expected DeprecationWarning when calling deprecated create_agent_state"


# ── Test 7 (from Task 5 Step 1) ──


def test_live_loop_state_serde_after_pr3_cleanup() -> None:
    """Live PR3 LoopState payload must serialize without deprecated state fields."""
    serde = agent_checkpoint_serde()
    config = _config()

    plan = ToolCallPlan.create("rag_search_answer", {"query": "What is the capital of France?"})

    result = ToolResult(
        tool_call_id=plan.tool_call_id,
        tool_name="rag_search_answer",
        status="ok",
        output=RAGSearchAnswerOutput(
            text="Paris is the capital of France.",
            evidence=[
                EvidenceItem(
                    evidence_id="ev_1",
                    doc_id=1,
                    citation_anchor="paris-capital",
                    text="Paris is the capital of France.",
                    score=0.95,
                ),
            ],
            citations=[
                AnswerCitation(
                    citation_id="cit_1",
                    evidence_id="ev_1",
                    record_type="section",
                    citation_anchor="paris-capital",
                    file_name="france_overview.pdf",
                ),
            ],
        ),
        latency_ms=0,
    )

    state = create_loop_state(task="What is the capital of France?", run_config=config)
    state["pending_tool_calls"] = [
        PendingToolCall(plan=plan, status="completed", summary="Paris is the capital of France."),
    ]
    state["tool_call_ledger"].append_plans([plan], turn=1)
    state["tool_results"] = [result]

    restored = serde.loads_typed(serde.dumps_typed(state))

    assert "evidence" not in restored
    assert "citations" not in restored
    assert restored["tool_results"][0].output.evidence[0].evidence_id == "ev_1"
    assert restored["tool_results"][0].output.citations[0].citation_id == "cit_1"
    assert len(restored["tool_call_ledger"].entries) == 1
    assert restored["pending_tool_calls"][0].tool_call_id == plan.tool_call_id
    assert restored["pending_tool_calls"][0].status == "completed"


# ── Test 8 ──


def test_tool_execution_boundary_still_uses_tool_call_plan() -> None:
    """ToolBatchRequest.calls must be typed as tuple[ToolCallPlan, ...]."""
    from rag.agent.core.tool_execution import ToolBatchRequest

    plan = ToolCallPlan.create("test_tool", {"arg": "value"})
    config = _config()
    request = ToolBatchRequest(
        calls=(plan,),
        run_config=config,
        allowed_tools=frozenset({"test_tool"}),
    )

    assert isinstance(request.calls[0], ToolCallPlan)
    assert request.calls[0].tool_name == "test_tool"
    assert request.calls[0].arguments == {"arg": "value"}


# ── Test 9 ──


def test_native_provider_transcript_preserves_arguments() -> None:
    """Rebuilt transcript from ToolCallLedger must preserve original tool arguments."""
    from rag.agent.core.llm_providers import _rebuild_tool_transcript

    config = _config()
    state = create_loop_state(task="test arg preservation", run_config=config)

    plans = [
        ToolCallPlan.create("search", {"query": "Paris", "top_k": 5}),
        ToolCallPlan.create("analyze", {"data": [1, 2, 3], "mode": "summary"}),
    ]
    state["tool_call_ledger"].append_plans(plans, turn=1)

    state["tool_results"] = [
        ToolResult(
            tool_call_id=plans[0].tool_call_id,
            tool_name="search",
            status="ok",
            output=_DummyOutput(),
            latency_ms=0,
        ),
        ToolResult(
            tool_call_id=plans[1].tool_call_id,
            tool_name="analyze",
            status="ok",
            output=_DummyOutput(),
            latency_ms=0,
        ),
    ]

    transcript = _rebuild_tool_transcript(state)

    # Verify original tool arguments preserved in rebuilt transcript
    assert transcript[0].tool_calls[0].input == {"query": "Paris", "top_k": 5}
    assert transcript[2].tool_calls[0].input == {"data": [1, 2, 3], "mode": "summary"}


# ── Cleanup ──


@pytest.fixture(autouse=True)
def _cleanup_run_registry() -> None:
    from rag.agent.core.context import RunRegistry

    yield
    RunRegistry.remove("test-pr3-cleanup")

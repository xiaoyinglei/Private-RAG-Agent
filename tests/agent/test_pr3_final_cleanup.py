from __future__ import annotations

import importlib

import pytest

from rag.agent.core.checkpointing import (
    _migrate_legacy_state,
    agent_checkpoint_serde,
)
from rag.agent.core.context import AgentRunConfig, RunRegistry
from rag.agent.core.messages import (
    ModelMessage,
    tool_result_message,
)
from rag.agent.core.messages import (
    ToolCall as ModelToolCall,
)
from rag.agent.core.turn_contracts import ToolCallPlan
from rag.agent.loop.state import (
    PendingToolCall,
    ToolCallLedger,
    create_loop_state,
)
from rag.agent.tools.tool import ToolContentBlock, ToolResult
from rag.schema.runtime import AccessPolicy


def _config() -> AgentRunConfig:
    return AgentRunConfig(
        run_id="test-pr3-cleanup",
        thread_id="test-pr3-cleanup",
        llm_budget_total=100,
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


def _result(plan: ToolCallPlan) -> ToolResult:
    return ToolResult(
        tool_call_id=plan.tool_call_id,
        tool_name=plan.tool_name,
        content=(
            ToolContentBlock(type="text", data={"text": "result ok"}),
        ),
        structured_content={"result": "ok"},
    )


def test_pending_single_track_roundtrip() -> None:
    plan = ToolCallPlan.create("search_knowledge", {"query": "policy"})
    pending = PendingToolCall(
        plan=plan,
        status="approved",
        summary="test summary",
    )

    serde = agent_checkpoint_serde()
    restored = serde.loads_typed(serde.dumps_typed(pending))

    assert isinstance(restored, PendingToolCall)
    assert restored.tool_call_id == plan.tool_call_id
    assert restored.tool_name == "search_knowledge"
    assert restored.status == "approved"
    assert restored.plan.arguments == {"query": "policy"}


def test_tool_call_ledger_is_bounded_fifo_and_preserves_active_calls() -> None:
    ledger = ToolCallLedger(max_entries=128)
    active_ids: set[str] = set()
    for index in range(130):
        plan = ToolCallPlan.create(f"tool_{index}", {"index": index})
        if index >= 125:
            active_ids.add(plan.tool_call_id)
        ledger.append_plans([plan], turn=1)

    ledger.trim(active_tool_call_ids=active_ids)

    assert len(ledger.entries) == 128
    assert active_ids.issubset(
        entry.plan.tool_call_id for entry in ledger.entries
    )


def test_canonical_transcript_preserves_calls_arguments_and_results() -> None:
    state = create_loop_state(task="transcript", run_config=_config())
    plans = [
        ToolCallPlan.create("search", {"query": "Paris", "top_k": 5}),
        ToolCallPlan.create("analyze", {"data": [1, 2, 3]}),
    ]
    state["tool_call_ledger"].append_plans(plans, turn=1)
    results = [_result(plan) for plan in plans]
    state["tool_results"] = results
    transcript: list[ModelMessage] = []
    for plan, result in zip(plans, results, strict=True):
        transcript.extend(
            [
                ModelMessage(
                    role="assistant",
                    content="",
                    tool_calls=(
                        ModelToolCall(
                            id=plan.tool_call_id,
                            name=plan.tool_name,
                            input=dict(plan.arguments),
                        ),
                    ),
                ),
                tool_result_message(result),
            ]
        )
    state["canonical_transcript"] = transcript

    assert len(transcript) == 4
    assert transcript[0].tool_calls[0].name == "search"
    assert transcript[0].tool_calls[0].input == {
        "query": "Paris",
        "top_k": 5,
    }
    assert transcript[1].role == "tool"
    assert transcript[1].tool_call_id == plans[0].tool_call_id
    assert transcript[2].tool_calls[0].input == {"data": [1, 2, 3]}
    assert transcript[3].tool_call_id == plans[1].tool_call_id


def test_deprecated_fields_are_absent_from_new_and_migrated_state() -> None:
    state = create_loop_state(task="new state", run_config=_config())
    assert _DEPRECATED_FIELDS.isdisjoint(state)

    raw: dict[str, object] = dict(state)
    raw.update(
        {
            "loop_messages": [{"type": "human", "content": "old"}],
            "tool_result_store": {"old": "value"},
            "evidence": [],
            "citations": [],
            "retrieval_signals": None,
        }
    )
    migrated = _migrate_legacy_state(raw)

    assert _DEPRECATED_FIELDS.isdisjoint(migrated)
    assert "loop_messages" not in migrated
    assert "tool_result_store" not in migrated


def test_removed_agent_state_module_stays_gone() -> None:
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("rag.agent.state")


def test_live_state_serde_preserves_final_result_and_ledger() -> None:
    plan = ToolCallPlan.create(
        "search_knowledge",
        {"query": "capital of France"},
    )
    state = create_loop_state(task="capital", run_config=_config())
    state["pending_tool_calls"] = [
        PendingToolCall(plan=plan, status="completed", summary="Paris")
    ]
    state["tool_call_ledger"].append_plans([plan], turn=1)
    state["tool_results"] = [
        ToolResult(
            tool_call_id=plan.tool_call_id,
            tool_name=plan.tool_name,
            structured_content={
                "answer_text": "Paris",
                "results": [{"evidence_id": "ev-1"}],
            },
        )
    ]

    serde = agent_checkpoint_serde()
    restored = serde.loads_typed(serde.dumps_typed(state))

    assert _DEPRECATED_FIELDS.isdisjoint(restored)
    result = restored["tool_results"][0]
    assert isinstance(result, ToolResult)
    assert result.structured_content["results"][0]["evidence_id"] == "ev-1"
    assert len(restored["tool_call_ledger"].entries) == 1
    assert restored["pending_tool_calls"][0].status == "completed"


@pytest.fixture(autouse=True)
def _cleanup_run_registry() -> None:
    yield
    RunRegistry.remove("test-pr3-cleanup")

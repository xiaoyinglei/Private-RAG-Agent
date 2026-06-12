from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest

from tests.agent.parity.loop_scenarios import run_loop_scenarios

BASELINE_PATH = (
    Path(__file__).parent
    / "parity"
    / "baselines"
    / "legacy_graph_v1.json"
)

OBSOLETE_CONTROLLER_FIELDS = {
    "produced_gaps",
    "related_gap_ids",
    "related_step_ids",
    "resolved_gaps",
}


def _load_baseline() -> dict[str, Any]:
    loaded = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    return cast(dict[str, Any], loaded)


def _assert_legacy_subset(legacy: object, loop: object) -> None:
    if isinstance(legacy, dict):
        assert isinstance(loop, dict)
        for key, value in legacy.items():
            if key in OBSOLETE_CONTROLLER_FIELDS:
                continue
            assert key in loop
            _assert_legacy_subset(value, loop[key])
        return
    if isinstance(legacy, list):
        assert isinstance(loop, list)
        assert len(loop) == len(legacy)
        for legacy_item, loop_item in zip(legacy, loop, strict=True):
            _assert_legacy_subset(legacy_item, loop_item)
        return
    assert loop == legacy


def _assert_common_tool_parity(
    legacy: dict[str, Any],
    loop: dict[str, Any],
) -> None:
    for field in (
        "final_answer",
        "final_output",
        "output_validation_errors",
        "tool_results",
        "answer_candidates",
        "evidence",
        "citations",
        "retrieval_signals",
        "retrieval_signals_debug",
        "evidence_refs",
        "computation_results",
        "locators",
        "insufficient_evidence_flag",
    ):
        assert loop[field] == legacy[field], field


@pytest.mark.anyio
async def test_agent_loop_matches_frozen_legacy_capabilities() -> None:
    baseline = _load_baseline()
    assert baseline["source_commit"] == "dbd746b9"
    assert baseline["schema_version"] == 1

    legacy = baseline["scenarios"]
    loop = cast(
        dict[str, dict[str, Any]],
        await run_loop_scenarios(),
    )
    assert set(loop) == set(legacy)

    for scenario in ("single_tool", "multiple_tools", "tool_retry"):
        _assert_common_tool_parity(legacy[scenario], loop[scenario])
        assert loop[scenario]["status"] == "completed"
        assert loop[scenario]["groundedness_flag"] is False

    rag_legacy = legacy["rag_grounding"]
    rag_loop = loop["rag_grounding"]
    _assert_common_tool_parity(rag_legacy, rag_loop)
    assert rag_loop["groundedness_flag"] is True
    _assert_legacy_subset(
        rag_legacy["context_units"],
        rag_loop["context_units"],
    )
    _assert_legacy_subset(
        rag_legacy["structured_observations"],
        rag_loop["structured_observations"],
    )
    locator = rag_loop["context_units"][0]["locator"]
    assert locator["rerank_score"] == 0.99
    assert locator["retrieval_channels"] == ["vector", "rerank"]
    assert locator["retrieval_family"] == "hybrid"

    approval_legacy = legacy["approval_resume"]
    approval_loop = loop["approval_resume"]
    assert approval_loop["paused"]["status"] == "paused"
    for field in (
        "pending_tool_calls",
        "tool_results",
        "human_input_request",
    ):
        assert approval_loop["paused"][field] == approval_legacy["paused"][field]
    _assert_common_tool_parity(
        approval_legacy["resumed"],
        approval_loop["resumed"],
    )
    assert approval_loop["resumed"]["observed"] == {
        "runner_calls": ["approved"]
    }
    # The loop clears the fulfilled request instead of exposing stale approval data.
    assert approval_legacy["resumed"]["human_input_request"] is not None
    assert approval_loop["resumed"]["human_input_request"] is None

    structured_legacy = legacy["structured_output"]
    structured_loop = loop["structured_output"]
    _assert_common_tool_parity(structured_legacy, structured_loop)
    # Schema validation proves shape, not evidence grounding.
    assert structured_legacy["groundedness_flag"] is True
    assert structured_loop["groundedness_flag"] is False

    child_legacy = legacy["child_agent"]
    child_loop = loop["child_agent"]
    _assert_common_tool_parity(child_legacy, child_loop)
    assert child_loop["observed"] == child_legacy["observed"]
    assert child_loop["groundedness_flag"] is True


@pytest.mark.anyio
async def test_agent_loop_documents_intentional_controller_changes() -> None:
    baseline = _load_baseline()
    legacy = baseline["scenarios"]
    loop = cast(
        dict[str, dict[str, Any]],
        await run_loop_scenarios(),
    )

    plain = loop["plain_without_tools"]
    assert legacy["plain_without_tools"]["status"] == "paused"
    assert plain["status"] == "completed"
    assert plain["final_answer"] == "Direct answer."

    goal_legacy = legacy["explicit_goal_spec"]
    goal_loop = loop["explicit_goal_spec"]
    assert goal_loop["status"] == goal_legacy["status"] == "paused"
    for field in ("tool_results", "answer_candidates", "evidence_refs"):
        assert goal_loop[field] == goal_legacy[field]
    assert goal_loop["open_gap_ids"] == []
    assert goal_loop["satisfied_requirements"] == []
    assert goal_loop["stop_hook_feedback"][0]["code"] == (
        "goal_contract_unsatisfied"
    )

    compaction_legacy = legacy["message_compaction"]
    compaction_loop = loop["message_compaction"]
    for field in ("messages", "working_summary", "memory_refs"):
        assert compaction_loop[field] == compaction_legacy[field]
    assert compaction_loop["iteration"] == 1
    assert compaction_legacy["iteration"] == 0

    fallback_legacy = legacy["model_fallback"]
    fallback_loop = loop["model_fallback"]
    assert fallback_loop["status"] == fallback_legacy["status"] == "paused"
    assert fallback_loop["observed"]["model_resolutions"] == (
        fallback_legacy["observed"]["model_resolutions"]
    )
    assert "fallback" in fallback_loop["observed"]["transition_reasons"]

    for scenario in loop.values():
        states = (
            scenario.values()
            if "paused" in scenario and "resumed" in scenario
            else (scenario,)
        )
        for state in states:
            assert state["open_gap_ids"] == []
            assert state["satisfied_requirements"] == []
            observations = state["structured_observations"]
            for observation in observations:
                for field in OBSOLETE_CONTROLLER_FIELDS:
                    assert observation.get(field, []) == []

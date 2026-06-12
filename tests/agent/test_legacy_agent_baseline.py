from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.agent.parity.scenarios import LEGACY_SCENARIO_NAMES, run_legacy_scenarios

BASELINE_PATH = (
    Path(__file__).parent
    / "parity"
    / "baselines"
    / "legacy_graph_v1.json"
)

EXPECTED_SOURCE_COMMIT = "dbd746b9"
EXPECTED_SCHEMA_VERSION = 1


@pytest.mark.anyio
async def test_legacy_graph_matches_frozen_migration_baseline() -> None:
    baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))

    assert baseline["schema_version"] == EXPECTED_SCHEMA_VERSION
    assert baseline["source_commit"] == EXPECTED_SOURCE_COMMIT
    assert set(baseline["scenarios"]) == set(LEGACY_SCENARIO_NAMES)
    assert await run_legacy_scenarios() == baseline["scenarios"]


def test_legacy_baseline_covers_required_migration_capabilities() -> None:
    assert set(LEGACY_SCENARIO_NAMES) == {
        "approval_resume",
        "child_agent",
        "explicit_goal_spec",
        "message_compaction",
        "model_fallback",
        "multiple_tools",
        "plain_without_tools",
        "rag_grounding",
        "single_tool",
        "structured_output",
        "tool_retry",
    }

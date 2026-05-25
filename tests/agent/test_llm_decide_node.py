from __future__ import annotations

from rag.agent.graphs.nodes.llm_decide import _apply_decision
from rag.agent.state import ThinkOutput, ToolCallPlan


def test_llm_decision_rewrites_reused_tool_call_ids() -> None:
    decision = ThinkOutput(
        action="execute",
        tool_calls=[
            ToolCallPlan(
                tool_call_id="tc_001",
                tool_name="asset_inspect",
                arguments={"asset_id": 14},
            )
        ],
        thought="inspect the matching asset",
    )

    update = _apply_decision(
        decision,
        next_iteration=3,
        used_tool_call_ids={"tc_001"},
    )

    [call] = update["pending_tool_calls"]
    assert call.tool_name == "asset_inspect"
    assert call.arguments == {"asset_id": 14}
    assert call.tool_call_id.startswith("tc_")
    assert call.tool_call_id != "tc_001"
    assert update["iteration"] == 3


def test_llm_decision_keeps_unique_tool_call_ids() -> None:
    call = ToolCallPlan(
        tool_call_id="tc_unique",
        tool_name="asset_analyze",
        arguments={"asset_id": 14, "operation": "dataframe_sql", "query": "SELECT 1"},
    )
    decision = ThinkOutput(
        action="execute",
        tool_calls=[call],
        thought="compute with SQL",
    )

    update = _apply_decision(
        decision,
        next_iteration=4,
        used_tool_call_ids={"tc_other"},
    )

    assert update["pending_tool_calls"] == [call]

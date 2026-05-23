from __future__ import annotations

from rag.agent.core.llm_prompts import build_route_prompt


def test_route_prompt_routes_table_analysis_to_direct_asset_flow() -> None:
    prompt = build_route_prompt(
        {
            "task": "北方和东北日提货合计是多少？",
            "pending_tool_calls": [],
        }
    )

    assert "单次 RAG 检索即可回答" in prompt
    assert "读取表格、执行计算" in prompt
    assert "asset_inspect/asset_analyze" in prompt

from __future__ import annotations

from rag.agent.core.llm_prompts import build_retrieval_hint_prompt, build_tool_decision_prompt


def test_route_prompt_generates_hints_for_model_driven_asset_flow() -> None:
    prompt = build_retrieval_hint_prompt(
        {
            "task": "北方和东北日提货合计是多少？",
            "pending_tool_calls": [],
        }
    )

    assert "检索提示分析器" in prompt
    assert "不要决定执行路径" in prompt
    assert "asset_list/asset_inspect" in prompt
    assert "asset_read_slice/asset_analyze" in prompt
    assert "agent_*" in prompt


def test_evaluate_prompt_scopes_asset_analysis_to_open_gaps() -> None:
    prompt = build_tool_decision_prompt(
        {
            "task": "北方和东北日提货合计是多少？",
            "iteration": 2,
            "tool_results": [],
        },
        budget_remaining=5000,
        context_text=(
            "open_gaps: answer, evidence\n"
            "asset_id=14 analysis_capabilities=[dataframe_sql] "
            "columns=[区域公司, 日_日提货]"
        ),
        allowed_tools=["asset_inspect", "asset_analyze"],
    )

    assert "每一次 LLM 决策必须对应当前 open_gaps" in prompt
    assert "应立即调用 asset_analyze" in prompt
    assert '"operation": "dataframe_sql"' in prompt
    assert '"query": "SELECT ... FROM sheet ..."' in prompt

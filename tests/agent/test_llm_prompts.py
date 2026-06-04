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


def test_tool_decision_prompt_includes_primitive_tool_contracts() -> None:
    prompt = build_tool_decision_prompt(
        {
            "task": "读取表格并用 Python 计算合计",
            "iteration": 1,
            "tool_results": [],
        },
        budget_remaining=5000,
        context_text="open_gaps: computation",
        allowed_tools=[
            "list_files",
            "read_file",
            "structured_probe",
            "write_file",
            "run_python",
        ],
    )

    assert 'list_files: {"path": str' in prompt
    assert 'read_file: {"path": str' in prompt
    assert 'structured_probe: {"path": str' in prompt
    assert 'write_file: {"path": str, "content": str' in prompt
    assert 'run_python: {"script_path": "scratch/...py"' in prompt
    assert "run_python 只能执行 scratch/ 下的 .py 文件" in prompt
    assert "read_file 只读取有界文本" in prompt
    assert "is_binary=True" in prompt
    assert "capabilities" in prompt
    assert "候选表头行" in prompt
    assert "不要假设第一行就是表头" in prompt
    assert "xlsx" not in prompt
    assert "openpyxl" not in prompt
    assert "data_only" not in prompt
    assert "重复表头" not in prompt


def test_tool_decision_prompt_tells_model_how_to_finish_workspace_results() -> None:
    prompt = build_tool_decision_prompt(
        {
            "task": "用 Python 计算 CSV 合计并给出最终答案",
            "iteration": 3,
            "tool_results": [],
        },
        budget_remaining=5000,
        context_text=(
            "open_gaps: answer\n"
            "tool_name=run_python status=ok preview: stdout: Total amount: 40.0\n"
            "path=reports/summary.txt"
        ),
        allowed_tools=["read_file", "write_file", "run_python", "llm_summarize"],
    )

    assert "open_gaps 仍包含 answer" in prompt
    assert "调用 llm_summarize" in prompt
    assert "不要直接 action=\"synthesize\"" in prompt
    assert "run_python 成功后不要重复运行同一脚本" in prompt
    assert "run_python 失败" in prompt
    assert "stderr" in prompt
    assert "overwrite=True" in prompt

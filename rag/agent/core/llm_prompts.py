from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rag.agent.loop.state import LoopState


# ── Retrieval hint prompt ──


def build_retrieval_hint_prompt(state: LoopState) -> str:
    task = state.get("task", "")
    pending_count = len(state.get("pending_tool_calls", []))

    return f"""你是检索提示分析器。根据用户任务生成结构化检索信号，供 AgentLoop 的模型工具决策使用。

不要决定执行路径，不要生成工具调用，不要建立并行规划工作流。模型会在主循环中选择
search/grounding/rerank/asset_list/asset_inspect/asset_read_slice/asset_analyze
或 agent_* 工具。

当前任务: {task}
待执行工具数: {pending_count}

请生成检索信号。返回 JSON:
{{
    "reason": "检索提示依据",
    "retrieval_signals": {{
        "special_targets": ["table"] 或 [],
        "quoted_terms": ["精确词1", "精确词2"] 或 [],
        "allow_graph_expansion": true 或 false
    }}
}}

检索信号字段说明：
- special_targets：任务是否专门针对 table/figure/caption/formula 等特殊元素。如果不是，填空数组。
- quoted_terms：任务中需要精确匹配的关键词（专有名词、编号、代码等）。从任务原文中提取，不要编造。
- allow_graph_expansion：是否需要展开知识图谱进行多跳推理。涉及关联关系、实体间查询时设为 true。

重要约束：
- 不要编造 doc_id、file_name、page_number、document_title。
- 不要填写 metadata_filters。
- quoted_terms 只从任务原文提取，不要凭空生成。"""


def build_loop_turn_prompt(
    state: LoopState,
    *,
    budget_remaining: int,
    allowed_tools: Sequence[str] = (),
) -> str:
    """Build the model contract for the ordinary Python loop kernel."""

    task = state.get("task", "")
    iteration = state.get("iteration", 0)
    tool_results = state.get("tool_results", [])
    ok_count = sum(
        1
        for result in tool_results
        if getattr(result, "status", None) == "ok"
    )
    error_count = sum(
        1
        for result in tool_results
        if getattr(result, "status", None) == "error"
    )
    return f"""You are the model decision boundary for one bounded agent loop turn.

Current task: {task}
Iteration: {iteration}
Remaining token budget: {budget_remaining}
Tool results: {ok_count} successful, {error_count} failed
Available tools: {", ".join(allowed_tools) if allowed_tools else "none"}

Return one structured outcome:
{{
    "action": "execute" | "finish" | "pause",
    "tool_calls": [
        {{"tool_call_id": "tc_xxxxxxxxxxxx", "tool_name": "vector_search", "arguments": {{"query": "..."}}}}
    ],
    "final_answer": "A complete candidate answer when action is finish",
    "pause_reason": "The external input required when action is pause"
}}

Rules:
- When a tool can materially advance the task, return action="execute" with one or more concrete calls.
- actual tool calls take precedence over an inconsistent finish label.
- When the task can be answered from current trusted context, return action="finish" with a non-empty final_answer.
- Use action="pause" only when external input or authorization is genuinely required.
- The current plan is advisory task context. It does not authorize tools or decide whether the run may finish.
- Do not repeat a completed tool call. Read prior structured results before choosing another call.
- Preserve citation identifiers, evidence links, retrieval scores, rerank scores,
  computations, and artifact paths in the answer.
- Keep tool arguments bounded. Do not place full documents, tables, logs, or scratchpad reasoning in the response.

{_format_tool_contracts(allowed_tools)}""".strip()


def _format_tool_contracts(allowed_tools: Sequence[str]) -> str:
    contracts = {
        "vector_search": 'vector_search: {"query": str, "top_k": int}',
        "keyword_search": 'keyword_search: {"query": str, "top_k": int}',
        "grounding": 'grounding: {"query": str, "evidence_ids": list[str]}',
        "rerank": 'rerank: {"query": str, "items": list[object]}',
        "asset_list": (
            'asset_list: {"doc_id": int?, "source_id": int?, "section_id": int?, '
            '"asset_type": str?, "limit": int}'
        ),
        "asset_inspect": (
            'asset_inspect: {"asset_id": int, "head_rows": int, "tail_rows": int}'
        ),
        "asset_read_slice": (
            'asset_read_slice: {"asset_id": int, "row_start": int, "row_count": int, '
            '"columns": list[str]?}'
        ),
        "asset_analyze": (
            'asset_analyze: {"asset_id": int, "operation": "dataframe_sql", '
            '"query": "SELECT ... FROM sheet ..."}'
        ),
        "llm_summarize": 'llm_summarize: {"task": str, "context_sections": list[str]}',
        "rag_search_answer": 'rag_search_answer: {"query": str, "top_k": int}',
        "list_files": 'list_files: {"path": str, "pattern": str?, "limit": int?}',
        "read_file": (
            'read_file: {"path": str, "max_bytes": int?}; '
            "只读取有界文本，二进制/非文本内容返回 is_binary=True 且不返回正文"
        ),
        "structured_probe": (
            'structured_probe: {"path": str, "max_rows": int?, "max_columns": int?, '
            '"max_tables": int?}; 返回有界样本、候选表头行和数据起始行'
        ),
        "write_file": (
            'write_file: {"path": str, "content": str, "overwrite": bool?}; '
            "只能写 scratch/artifacts/reports/logs"
        ),
        "run_python": (
            'run_python: {"script_path": "scratch/...py", "args": list[str]?, '
            '"timeout_seconds": float?}; run_python 只能执行 scratch/ 下的 .py 文件'
        ),
    }
    names = list(allowed_tools)
    lines = [
        contracts[name]
        for name in names
        if name in contracts
    ]
    if not lines:
        return ""
    return "可用工具输入契约:\n" + "\n".join(f"- {line}" for line in lines)

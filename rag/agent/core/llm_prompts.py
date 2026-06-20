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
    """Build the model contract for the ordinary Python loop kernel.

    ``allowed_tools`` should already be the resolved visible tool list
    (caller uses VisibleToolResolver / resolve_visible_tools before
    calling this function).
    """

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
    visible_names = list(allowed_tools)

    return f"""Task: {task}
Iteration: {iteration}
Budget remaining: {budget_remaining} tokens
Tools completed: {ok_count} ok, {error_count} failed
Available tools: {", ".join(visible_names) if visible_names else "none"}

Analyze the task and current context, then decide your next action.

If a tool can advance the task → return action="execute" with concrete tool_calls.
If you have enough context to answer → return action="finish" with a complete, well-cited final_answer.
If you need external input → return action="pause" with a clear pause_reason.

Do not repeat completed tool calls. Preserve citations, scores, and artifact paths.
Keep tool arguments bounded — no full documents or logs in arguments.

Return JSON:
{{
    "action": "execute" | "finish" | "pause",
    "tool_calls": [{{"tool_call_id": "tc_xxx", "tool_name": "...", "arguments": {{...}}}}],
    "final_answer": "...",
    "pause_reason": "..."
}}""".strip()

from __future__ import annotations

from rag.agent.state import AgentState

# ── Router prompt ──


def build_route_prompt(state: AgentState) -> str:
    task = state.get("task", "")
    retrieval_signals = state.get("retrieval_signals")
    pending_count = len(state.get("pending_tool_calls", []))

    signals_desc = "无特殊检索信号"
    if retrieval_signals is not None:
        parts = []
        if retrieval_signals.special_targets:
            parts.append(f"目标类型: {', '.join(retrieval_signals.special_targets)}")
        if retrieval_signals.quoted_terms:
            parts.append(f"精确词: {', '.join(retrieval_signals.quoted_terms)}")
        if retrieval_signals.allow_graph_expansion:
            parts.append("知识图谱扩展已启用")
        if parts:
            signals_desc = "; ".join(parts)

    return f"""你是任务路由器。根据用户任务判断应走哪条执行路径。

路由标准（按执行需求，不按固定分类）：

- fast_path：单次 RAG 检索即可回答。不需要拆分子任务、并行 Agent、用户确认或外部副作用。
- decompose：需要多个独立检索、多维度证据、多对象对比，或需要并行子 Agent / 任务 DAG。
- direct：需要 Agent 循环调用工具（search/grounding/rerank 等），或需要多轮 evaluate、用户确认。

当前任务: {task}
检索信号: {signals_desc}
待执行工具数: {pending_count}

请判断路由。返回 JSON:
{{"route": "fast_path" | "decompose" | "direct", "reason": "判断依据"}}"""


# ── Evaluator prompt ──


def build_evaluate_prompt(
    state: AgentState,
    *,
    budget_remaining: int,
    context_text: str,
) -> str:
    task = state.get("task", "")
    iteration = state.get("iteration", 0)
    tool_results = state.get("tool_results", [])
    ok_count = sum(1 for r in tool_results if hasattr(r, "status") and r.status == "ok")
    error_count = sum(1 for r in tool_results if hasattr(r, "status") and r.status == "error")

    return f"""你是证据评估器。根据当前收集的证据和工具结果，决定下一步行动。

当前任务: {task}
迭代次数: {iteration}
预算剩余 tokens: {budget_remaining}
工具结果: {ok_count} 成功, {error_count} 失败

{context_text}

请判断下一步。返回 JSON:
{{
    "action": "execute" | "synthesize" | "pause",
    "tool_calls": [
        {{"tool_call_id": "tc_xxxxxxxxxxxx", "tool_name": "vector_search", "arguments": {{"query": "...", "top_k": 8}}}}
    ],
    "thought": "推理过程",
    "confidence": 0.0~1.0,
    "stop_reason": "证据充分时说明原因",
    "needs_user_input": "需要用户决策时说明问题"
}}

规则：
- action="execute" 时 tool_calls 必须非空，每个 tool_call_id 前缀为 tc_
- 证据充分 → action="synthesize"，stop_reason 说明原因
- 还需要检索 → action="execute"，给出具体 tool_calls
- 需要用户决策 → action="pause"，needs_user_input 说明问题
- 预算耗尽 → action="synthesize"，stop_reason="budget_exhausted"
- 如果证据已包含 retrieval_channels 冲突标记，考虑是否需要用户选择"""


# ── Planner prompt ──


def build_plan_prompt(
    state: AgentState,
    *,
    allowed_tools: list[str],
    max_depth: int,
) -> str:
    task = state.get("task", "")
    tools_list = ", ".join(allowed_tools) if allowed_tools else "无"

    return f"""你是任务规划器。将复杂任务拆解为可并行或串行的子任务 DAG。

当前任务: {task}
可用工具: {tools_list}
允许的 Agent 类型: research, compare, factcheck
最大嵌套深度: {max_depth}

请生成一个 TaskDAG。返回 JSON:
{{
    "subtasks": [
        {{
            "subtask_id": "唯一 ID（如 s1, s2）",
            "agent_type": "research | compare | factcheck",
            "prompt": "该子任务的完整指令",
            "priority": 0~10（数字越大越优先）,
            "estimated_tokens": 8000
        }}
    ],
    "edges": [
        {{"from_id": "s1", "to_id": "s3"}}
    ]
}}

规则：
- subtask_id 必须唯一
- edges 表示依赖：to_id 依赖 from_id 完成
- 无依赖的子任务可并行执行（留空 edges 即可）
- 不要创建循环依赖
- priority 用于同批次内的排序
- 每个子任务只做一件事，不要合并多个独立问题"""
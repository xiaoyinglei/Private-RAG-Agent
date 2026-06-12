from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from rag.agent.state import AgentState

if TYPE_CHECKING:
    from rag.agent.loop.state import LoopState

# ── Goal contract prompt ──


def build_goal_contract_prompt(state: AgentState) -> str:
    task = state.get("task", "").strip()
    return f"""You define a bounded completion contract for an agent task.

Return only a structured GoalContractHint. Select required deliverable kinds:
- answer: always required
- evidence: only when the user requires traceable sources or citations
- computation: only when the task requires a calculated or reproducible result

Add constraints only when the task explicitly fixes a source, table, sheet,
document, scope, or other acceptance boundary. Do not invent identifiers,
titles, files, or constraints. The contract describes acceptance conditions;
it does not choose tools or execution steps.

User task:
{task}"""


# ── Retrieval hint prompt ──


def build_retrieval_hint_prompt(state: AgentState) -> str:
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


# ── Tool decision prompt ──


def build_tool_decision_prompt(
    state: AgentState,
    *,
    budget_remaining: int,
    context_text: str,
    allowed_tools: Sequence[str] = (),
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
可用工具: {", ".join(allowed_tools) if allowed_tools else "未显式限制"}

{context_text}

{_format_tool_contracts(allowed_tools)}

请判断下一步。返回 JSON:
{{
    "action": "execute" | "synthesize" | "pause",
    "tool_calls": [
        {{"tool_call_id": "tc_xxxxxxxxxxxx", "tool_name": "vector_search", "arguments": {{"query": "...", "top_k": 8}}}}
    ],
    "thought": "推理过程",
    "confidence": 0.0~1.0,
    "stop_reason": "证据充分时说明原因",
    "needs_user_input": "需要用户决策时说明问题",
    "plan_update": {{
        "mode": "replace" | "patch",
        "objective": "当前目标摘要，可省略",
        "status": "active" | "blocked" | "needs_replan",
        "active_step_id": "step_xxx",
        "steps": [
            {{
                "step_id": "step_xxx",
                "title": "有界步骤说明",
                "status": "pending" | "in_progress" | "blocked" | "skipped",
                "related_gap_ids": ["answer"],
                "expected_tool_names": ["list_files"],
                "notes": "短说明"
            }}
        ],
        "step_updates": [
            {{"step_id": "step_xxx", "status": "in_progress", "notes": "短说明"}}
        ],
        "summary": "计划状态摘要"
    }}
}}

规则：
- action="execute" 时 tool_calls 必须非空，每个 tool_call_id 前缀为 tc_
- 只有目标检查已经确认不存在 open_gaps 或 conflicts 时，才可用 action="synthesize"
- 还需要检索 → action="execute"，给出具体 tool_calls
- 需要用户决策 → action="pause"，needs_user_input 说明问题
- 预算耗尽且目标尚未满足 → action="pause"，needs_user_input 说明无法继续补证据
- 每一次 LLM 决策必须对应当前 open_gaps；如果没有 open_gaps，不要继续调用工具
- 维护 plan_update，让主循环拥有可恢复 todo/replan 状态；计划不是权限，不能替代 tool_calls，
  也不能绕过工具白名单、审批、workspace 或 Python 沙箱。
- 不得把 plan status 设为 complete，也不得把 step status 设为 completed。LLM 只能描述计划和推进状态；
  step 完成必须来自绑定成功的 structured observation，plan 完成必须来自 goal checker。
- plan_update 最多保留 12 个步骤；每个步骤只写短标题、相关 open_gaps、预期工具名和短 notes。
  不要把完整表格、完整日志、完整工具输出、长 scratchpad 或大段推理写进计划。
- 如果当前计划方向不再匹配 open_gaps，用 mode="replace" 给出新的有界步骤；否则用 mode="patch"
  更新 active_step_id、步骤状态和短 notes。
- 如果 open_gaps 仍包含 answer，但已有 read_file、write_file、run_python 或其他工具结果
  足以回答，应调用 llm_summarize 生成 answer_candidate；不要直接 action="synthesize"。
- 如果证据已包含 retrieval_channels 冲突标记，考虑是否需要用户选择
- 对 workspace 文件任务，先用 list_files 获取精确 path 和文件能力字段，再根据
  readable_as_text、is_binary、file_kind、mime_type、capabilities 选择工具。
  read_file 只读取有界文本；如果文件不可作为文本读取，或 read_file 返回 is_binary=True，
  不要重试 read_file。需要解析非文本、结构化或二进制内容时，先用 write_file 写入
  scratch/*.py，再用 run_python 执行合适的解析代码。
  如果可用工具包含 structured_probe，优先用它获取有界样本、候选表、候选表头行和数据起始行，
  再决定是否需要写 Python 做进一步计算。
  脚本中的文件路径必须逐字使用 list_files 返回的 path。
  解析表格或结构化数据时，先检查标题、说明、空行和前几行样本，不要假设第一行就是表头；
  输出结构时说明采用的表头行和数据起始行。
  列出 workspace 根目录时用 path=""，不要用 path="/"。
  run_python 成功后不要重复运行同一脚本；如 stdout、generated_files 或 reports/* 已足以回答，应调用
  llm_summarize 汇总成最终答案候选。
  run_python 失败时先阅读 stderr/stdout，定位脚本错误；不要重复运行同一个失败脚本。需要修复时调用
  write_file 且 overwrite=True 覆盖 scratch/*.py，再重新 run_python。
- 对文件/结构化资产问题，先用检索工具找到 asset_id；拿到 asset_id 后优先调用
  asset_inspect 理解资产结构和可用 analysis_capabilities。需要局部行列内容时用
  asset_read_slice 读取有边界的切片；需要读数、筛选、排序、聚合或校验时用
  asset_analyze。不要只根据摘要文本回答可执行资产里的计算。
- 如果已经 inspect 到一个资产，并且它有 dataframe_sql 能力，且列名/预览行足以回答
  当前 open_gaps，应立即调用 asset_analyze 做计算或校验；不要因为还存在其他候选资产
  就继续逐个 asset_inspect。只有当你能说明具体未解决歧义时，才继续 inspect 其他资产。
- 不要把完整表格放进状态或上下文；只保留候选 asset_id、结构摘要、切片、分析规格、计算结果和 evidence/citation 定位。
- 如果任务没有指定资产、sheet、产品、场景、口径等范围，但已有多个候选资产都可能回答同一指标，
  不要任选一个 asset_id；应调用 asset_list/asset_inspect 收集候选，分别计算并标注候选答案，
    或在必须给唯一答案时 pause 请求用户澄清。"""


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

# Agent Architecture: Tool-Centric + LangGraph Engine

## 1. Overview

将 Claude Code 的 "Agent as Tool" 组合模式和 LangGraph 的状态图执行引擎结合，构建可组合、可中断、可并行的工业级 Agent 编排层。

**核心原则**：不手搓 while 循环。LangGraph 是"操作系统"，Agent 编译为图，封装为 Tool。

## 2. Why This Combination

| 问题 | Tool-Centric 提供 | LangGraph 提供 |
|------|------------------|---------------|
| 如何定义 Agent | `AgentDefinition` 注册表 | —— |
| 如何运行 Agent | 封装为 Tool 被父 Agent 调用 | `StateGraph` 执行引擎 |
| 如何做决策 | Agent 的 system_prompt | `conditional_edge` + LLM 结构化 routing |
| 如何中断/恢复 | —— | `interrupt()` + `checkpointer` + `Command(resume=...)` |
| 如何并行 | 无依赖子任务可并行调度 | `Send()` API 原生并行，reducer 合并 |
| 如何重试/降级 | Tool 级 retry/fallback | conditional edge 天然支持分支 |
| 子 Agent 组合 | AgentAsTool | `Send()` 到子 Graph 或嵌套 invoke |

## 3. Architecture Layers

```
┌──────────────────────────────────────────────────────────┐
│                      CLI / API 层                         │
│  rag agent chat  │  runtime.analyze_task()                │
├──────────────────────────────────────────────────────────┤
│                   Orchestrator Graph                      │
│  route → plan → execute(sub-agents) → synthesize          │
├───────────────────────┬──────────────────────────────────┤
│    Agent 定义层        │       Tool 注册表                  │
│  AgentDefinition      │  ToolSpec(name, input, output,   │
│  AgentRegistry        │    permissions, timeout, retry,  │
│  AgentGraphCompiler   │    budget, audit, idempotent)    │
├───────────────────────┴──────────────────────────────────┤
│                LangGraph 执行引擎                          │
│  StateGraph  │  Annotated Reducers  │  Checkpointer       │
│  interrupt() │  Command(resume)     │  Send()并行          │
├──────────────────────────────────────────────────────────┤
│              Memory / Context 管理                         │
│  working_summary │ extracted_facts │ context injector     │
├──────────────────────────────────────────────────────────┤
│              现有 RAG 管线 (L3-L6)                         │
│  planning_graph → l3_l4_engine → grounding → synthesis   │
└──────────────────────────────────────────────────────────┘
```

---

## 4. Contract Layer 1: ToolSpec（工具契约）

每个工具必须定义完整契约，否则不能注册进 `ToolRegistry`。

### 4.1 ToolPermissions

```python
@dataclass(frozen=True)
class ToolPermissions:
    read_db: bool = False           # 是否读 PostgreSQL
    write_db: bool = False          # 是否写 PostgreSQL
    read_object_store: bool = False # 是否读 S3/Object Store
    embed: bool = False             # 是否调用 embedding 模型
    generate: bool = False          # 是否调用 LLM 生成
    external_network: bool = False  # 是否访问外网（Web Search / Web Fetch）
    kg_mutation: bool = False       # 是否修改知识图谱
    user_data: bool = False         # 是否接触用户私有数据
```

### 4.2 ToolSpec

```python
@dataclass(frozen=True)
class ToolSpec:
    name: str                              # 唯一工具名，如 "vector_search"
    description: str                       # LLM 可读的功能描述
    input_model: type[BaseModel]           # 输入参数的 Pydantic 模型
    output_model: type[BaseModel]          # 成功返回值的 Pydantic 模型
    error_model: type[BaseModel]           # 错误返回值的 Pydantic 模型
    permissions: ToolPermissions           # 权限标记
    timeout_seconds: float                 # 超时时间
    max_retries: int = 0                   # 失败后最大重试次数
    idempotent: bool = False               # 是否为幂等操作（决定重试安全性）
    token_budget_cost: int = 0             # 单次调用的预估 token 消耗
    requires_confirmation: bool = False    # 执行前是否需要用户确认
    audit_log: bool = False               # 是否写入审计日志
```

### 4.3 ToolResult / ToolError（互斥）

```python
class ToolResult(BaseModel):
    """成功/失败互斥：status='ok' 时 output 必填，status='error' 时 error 必填"""
    tool_call_id: str                    # 对应 ToolCallPlan.tool_call_id
    tool_name: str
    status: Literal["ok", "error"]
    output: BaseModel | None = None      # status='ok' 时非空
    error: ToolError | None = None       # status='error' 时非空
    latency_ms: float
    token_used: int = 0
    retry_count: int = 0

    @model_validator(mode="after")
    def _check_exclusivity(self) -> ToolResult:
        """output 和 error 严格互斥"""
        if self.status == "ok":
            if self.output is None:
                raise ValueError("output is required when status='ok'")
            if self.error is not None:
                raise ValueError("error must be None when status='ok'")
        if self.status == "error":
            if self.error is None:
                raise ValueError("error is required when status='error'")
            if self.output is not None:
                raise ValueError("output must be None when status='error'")
        return self

class ToolError(BaseModel):
    code: str                             # "timeout" | "permission_denied" | "rate_limited" | "internal"
    message: str
    retryable: bool
    detail: dict[str, object] = Field(default_factory=dict)
```

### 4.4 Agent Tool = ToolSpec + AgentDefinition 组合

```python
@dataclass(frozen=True)
class AgentToolSpec:
    """子 Agent 封装的完整契约，组合 ToolSpec 和 AgentDefinition"""
    tool_spec: ToolSpec                    # 基础工具契约
    agent_definition: AgentDefinition      # 对应的 Agent 定义
    inherits_context: bool = True          # 是否继承父 AgentRunConfig / source_scope / access_policy
```

---

## 5. Contract Layer 2: AgentRunConfig + RuntimeRegistry（序列化 / 运行时分离）

AgentState 会被 LangGraph checkpointer 序列化。asyncio.Lock、asyncio.Event、BudgetLedger 等运行时对象不能进入 checkpoint state。因此拆为两层：

- **AgentRunConfig**（可序列化）→ 放在 AgentState 中
- **AgentRuntimeHandles**（不可序列化）→ 存在外部 RuntimeRegistry 中，按 run_id 索引

```python
@dataclass(frozen=True)
class AgentRunConfig:
    """可序列化的运行配置，进入 AgentState，可被 checkpointer 持久化"""
    # ── 无默认值字段 ──
    run_id: str
    thread_id: str
    budget_total: int                     # 原始总预算（快照）
    max_depth: int                        # 剩余可嵌套深度（每层 -1）
    access_policy: AccessPolicy
    execution_location_preference: ExecutionLocationPreference
    # ── 有默认值字段 ──
    parent_run_id: str | None = None
    source_scope: tuple[str, ...] = ()
    deadline_iso: str | None = None       # ISO 8601，可序列化
    trace_parent_id: str | None = None
    budget_committed: int = 0             # 序列化快照（仅用于 checkpoint 恢复）
    budget_reserved: dict[str, int] = field(default_factory=dict)
    tool_policy: ToolPolicy = field(default_factory=ToolPolicy)  # 可序列化的工具策略（从 AgentDefinition 复制）


class RuntimeRegistry:
    """外部注册表，按 run_id 存储运行时活对象。创建时从 AgentRunConfig 初始化 BudgetLedger"""
    _handles: dict[str, AgentRuntimeHandles] = {}

    @classmethod
    def get_or_create(cls, run_config: AgentRunConfig) -> AgentRuntimeHandles:
        if run_config.run_id not in cls._handles:
            cls._handles[run_config.run_id] = AgentRuntimeHandles(
                budget_ledger=BudgetLedger(total=run_config.budget_total),
                cancellation=asyncio.Event(),
            )
        return cls._handles[run_config.run_id]

    @classmethod
    def get(cls, run_id: str) -> AgentRuntimeHandles:
        """获取已存在的 handles。调用方保证 run_id 已被 get_or_create 注册过"""
        return cls._handles[run_id]

    @classmethod
    def remove(cls, run_id: str) -> None:
        cls._handles.pop(run_id, None)


@dataclass
class AgentRuntimeHandles:
    """不可序列化的运行时对象。由 RuntimeRegistry.get_or_create() 保证字段非空"""
    budget_ledger: BudgetLedger
    cancellation: asyncio.Event
```

### 5.2 BudgetLedger（并发安全预算管理）

```python
class BudgetLedger:
    """线程安全的 token 预算账本。父 Agent 原子 reserve，子任务结束 commit/refund。
    使用 asyncio.Lock 以避免与 async 事件循环死锁。"""
    def __init__(self, total: int):
        self._total = total
        self._lock = asyncio.Lock()
        self._reserved: dict[str, int] = {}
        self._committed: int = 0

    async def remaining(self) -> int:
        async with self._lock:
            return max(0, self._total - self._committed - sum(self._reserved.values()))

    async def reserve(self, lease_id: str, amount: int) -> bool:
        """原子预留。持有锁期间直接计算 remaining，不调用 self.remaining()。"""
        async with self._lock:
            current = max(0, self._total - self._committed - sum(self._reserved.values()))
            if amount > current:
                return False
            self._reserved[lease_id] = amount
            return True

    async def commit(self, lease_id: str, actual: int) -> int:
        """确认消费。返回 overrun（超出预留的 token 数，0 表示正常）"""
        async with self._lock:
            reserved = self._reserved.pop(lease_id, 0)
            overrun = max(0, actual - reserved)
            self._committed += actual  # 记录真实消耗，即使超支
            return overrun

    async def refund(self, lease_id: str) -> int:
        """退回预留。返回退回的 token 数"""
        async with self._lock:
            return self._reserved.pop(lease_id, 0)
```

**上下文继承规则**：子 Agent 创建时 `derive_child_config(parent_config, child_def)`:
- `run_id` → 新生成
- `thread_id` → 新生成（独立 checkpoint 空间，避免与父冲突）
- `parent_run_id` → 父的 `run_id`
- `access_policy` → `parent_config.access_policy.narrow(child_def.access_policy)`
- `source_scope` → 子集（子 Agent 只能缩小范围）
- `budget_total` → 从 `ToolSpec.token_budget_cost` 估算，或父显式分配
- `budget_committed/reserved` → 初始化为 0
- `max_depth` → `parent_config.max_depth - 1`（≤0 时拒绝嵌套）
- 运行时 handles（BudgetLedger、cancellation event）→ 从父 `RuntimeRegistry` 继承或新建

```python
def derive_child_config(parent: AgentRunConfig, child_def: AgentDefinition) -> AgentRunConfig:
    if parent.max_depth <= 0:
        raise RuntimeError(f"Agent nesting depth exceeded for {child_def.agent_type}")
    child_id = str(uuid4())
    return AgentRunConfig(
        run_id=child_id,
        thread_id=child_id,
        parent_run_id=parent.run_id,
        access_policy=(
            parent.access_policy.narrow(child_def.access_policy)
            if child_def.access_policy is not None
            else parent.access_policy
        ),
        source_scope=parent.source_scope,
        execution_location_preference=parent.execution_location_preference,
        max_depth=parent.max_depth - 1,
        budget_total=child_def.estimated_token_budget,
        tool_policy=child_def.tool_policy,
    )
```

### 5.3 AgentDefinition

```python
@dataclass(frozen=True)
class AgentDefinition:
    """Agent 的完整定义契约，驱动 compiler 生成图"""
    agent_type: str                          # "research" | "compare" | "factcheck" | "orchestrator"
    description: str                         # 注册表中给 LLM 看的用途描述
    system_prompt: str                       # Agent 专用 system prompt
    allowed_tools: list[str]                 # 允许的 tool 名称列表
    access_policy: AccessPolicy | None = None  # Agent 级别的权限收窄（None 表示继承父）
    estimated_token_budget: int = 8000        # 单次运行预估 token 消耗（用于 BudgetLedger 预留）
    model_policy: ModelPolicy = field(default_factory=ModelPolicy)
    output_model: type[BaseModel] | None = None  # Agent 最终输出的 Pydantic 模型
    max_iterations: int = 10
    max_depth: int = 2
    tool_policy: ToolPolicy = field(default_factory=ToolPolicy)

@dataclass(frozen=True)
class ModelPolicy:
    model_alias: str = "opus"                # "opus" | "sonnet" | "haiku"
    fallback_model: str | None = "sonnet"    # 主模型不可用时的降级
    thinking: bool = True                    # 是否启用 extended thinking
    temperature: float = 0.0

@dataclass(frozen=True)
class ToolPolicy:
    max_parallel_calls: int = 4              # 单次 execute 最大并行 tool 数
    require_confirmation_for: frozenset[str] = field(default_factory=frozenset)  # 必须用户确认的 tool 名
    deny_tools: frozenset[str] = field(default_factory=frozenset)  # 显式禁止的 tool 名
```

---

## 6. Contract Layer 3: AgentState（状态契约，带 reducer）

### 6.1 State 定义

```python
from langgraph.graph import add_messages
from typing import Annotated

class ThinkOutput(BaseModel):
    """evaluate 节点输出的结构化决策"""
    action: Literal["execute", "synthesize", "pause"]
    tool_calls: list[ToolCallPlan] = Field(default_factory=list)
    thought: str                          # LLM 推理过程（可解释性）
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    stop_reason: str | None = None        # action=synthesize 时
    needs_user_input: str | None = None   # action=pause 时

class ToolCallPlan(BaseModel):
    tool_call_id: str                     # 稳定唯一 ID，贯穿 execute/retry/audit/reducer
    tool_name: str
    arguments: dict[str, object]

    @classmethod
    def create(cls, tool_name: str, arguments: dict) -> ToolCallPlan:
        return cls(
            tool_call_id=f"tc_{uuid4().hex[:12]}",
            tool_name=tool_name,
            arguments=arguments,
        )

class WorkingSummary(BaseModel):
    """旧 messages 脱水后的短期工作摘要，只保留当前 run 仍需使用的信息"""
    summary: str
    covered_message_ids: list[str]
    updated_at: str
    token_count: int

class ExtractedFact(BaseModel):
    """从旧 messages / tool results 中抽取的当前任务事实。Phase A 只进 working memory，不写长期存储"""
    fact_id: str
    text: str
    source_message_ids: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    stale: bool = False

class ContextBudgetSnapshot(BaseModel):
    """每次 LLM 调用前的上下文预算分配，便于诊断 prompt 膨胀"""
    max_context_tokens: int
    system_tokens: int = 0
    evidence_tokens: int = 0
    working_memory_tokens: int = 0
    recalled_memory_tokens: int = 0
    message_tail_tokens: int = 0
    tool_result_tokens: int = 0

class AgentState(TypedDict):
    # ── LangGraph 内置 reducer ──
    messages: Annotated[list[BaseMessage], add_messages]

    # ── 自定义 reducer（并行安全）──
    evidence: Annotated[
        list[EvidenceItem],
        _merge_evidence       # 按 evidence_id 去重，scalar 取 max，冲突保留两者
    ]
    citations: Annotated[
        list[Citation],
        _merge_citations      # 按 citation_id 去重
    ]
    tool_results: Annotated[
        list[ToolResult],
        _merge_tool_results   # 按 tool_call_id 去重
    ]

    # ── 非并行字段（单写者）──
    task: str
    run_config: AgentRunConfig
    plan: TaskDAG | None
    iteration: int
    status: str                           # "running" | "paused" | "done" | "failed"
    pending_tool_calls: list[ToolCallPlan]
    confirmed_tool_call_ids: set[str]     # 用户已确认的 tool_call_id；SQLiteSaver(pickle)可直接序列化，JSON checkpointer 需转为 list
    user_decision: str | None
    next_subtasks: list[SubTaskNode] | None   # evaluate 写入的路由信息
    working_summary: WorkingSummary | None    # Phase A：旧 messages 的脱水摘要
    extracted_facts: list[ExtractedFact]       # Phase A：脱水后的任务事实，不等同长期记忆
    context_budget: ContextBudgetSnapshot | None

    # ── 并行子任务追踪（reducer 合并）──
    subtask_results: Annotated[
        dict[str, SubTaskResult],          # {subtask_id: SubTaskResult}
        _merge_subtask_results
    ]
    terminal_subtasks: Annotated[
        set[str],                          # 已终止的 subtask_id（成功或失败）
        _merge_sets
    ]
    successful_subtasks: Annotated[
        set[str],                          # 成功的 subtask_id（用于满足依赖）
        _merge_sets
    ]

    # ── 仅在 status="done" 时写入 ──
    final_answer: str | None
    groundedness_flag: bool
    insufficient_evidence_flag: bool
```

### 6.2 Reducer 实现

```python
def _merge_evidence(left: list[EvidenceItem], right: list[EvidenceItem]) -> list[EvidenceItem]:
    """并行 Send 合并：evidence_id 去重，冲突时保留两者并标记"""
    merged: dict[str, EvidenceItem] = {}
    for item in left + right:
        existing = merged.get(item.evidence_id)
        if existing is None:
            merged[item.evidence_id] = item
        elif _texts_contradict(existing.text, item.text):
            # 冲突：保留两者，各自标记 retrieval_channels += ["conflict"]
            merged[item.evidence_id] = existing.model_copy(
                update={"retrieval_channels": [*existing.retrieval_channels, "conflict"]}
            )
            merged[f"{item.evidence_id}__conflict"] = item.model_copy(
                update={"evidence_id": f"{item.evidence_id}__conflict",
                         "retrieval_channels": [*item.retrieval_channels, "conflict"]}
            )
        elif item.score > existing.score:
            merged[item.evidence_id] = item
    return sorted(merged.values(), key=lambda e: e.score, reverse=True)

def _merge_citations(left, right):
    return list({c.citation_id: c for c in left + right}.values())

def _merge_tool_results(left, right):
    return list({r.tool_call_id: r for r in left + right}.values())

def _merge_subtask_results(left: dict, right: dict) -> dict:
    return {**left, **right}  # {subtask_id: SubTaskResult}

def _merge_sets(left: set, right: set) -> set:
    return left | right        # set union for terminal_subtasks / successful_subtasks
```

---

## 7. Graph: Nodes and Edges

用工程化命名替代通用 ReAct 命名：

```
Nodes:
  route              → 复杂度分检：fast_path / decompose / direct
  plan               → 拆解任务，生成 TaskDAG（Orchestrator 专用）
  execute            → 执行普通 tool_call（search / grounding / generate）
  execute_subagent   → 执行一个子 Agent 任务（被 Send 并行扇出）
  observe            → 处理 tool result，注入消息历史
  evaluate           → 评估证据充分性，决定下一步
  pause              → 暂停等待用户输入（调用 interrupt()）
  synthesize         → 格式化最终输出

Edges:
  route ──conditional──▶ fast_path        (简单查询 → 直接 RAG 管线)
  route ──conditional──▶ plan             (复杂查询 → 拆解任务)
  route ──conditional──▶ execute           (中等查询 → 单 Agent)
  plan  ──Send()──────▶ execute_subagent   (并行扇出到子 Agent)
  execute ──normal────▶ observe        (Command(goto='pause') 时优先)
  execute_subagent ──normal──▶ evaluate   (子 Agent 完成后评估)
  observe ──normal────▶ evaluate
  evaluate ──conditional─▶ execute         (继续检索)
  evaluate ──conditional─▶ execute_subagent(调度下一批依赖子任务)
  evaluate ──conditional─▶ pause           (需要用户决策)
  evaluate ──conditional─▶ synthesize      (证据充分/预算耗尽/所有子任务完成)
  pause ──normal───────▶ evaluate          (用户响应后→重新评估)
```

### 7.1 route 节点（复杂度分检）

简单查询走快路径，避免无效编排开销：

```python
def route_node(state: AgentState) -> dict:
    task = state["task"]
    understanding = _classify_complexity(task)

    if understanding["complexity"] == "simple":
        # 不进入 Agent Loop，直接调 L3-L4-L5-L6 管线
        return {"status": "fast_path", "route_reason": "simple_lookup"}
    elif understanding["complexity"] == "decompose":
        return {"status": "decompose", "route_reason": "multi_hop_or_compare"}
    else:
        return {"status": "direct", "route_reason": "single_agent_research"}

def _classify_complexity(task: str) -> dict:
    """不用 LLM，用 query_understanding_service 做语义分流"""
    # 复用现有的 QueryUnderstandingService.analyze()
    # task_type=lookup → simple
    # task_type=comparison/synthesis/timeline → decompose
    # 其他 → direct
```

**fast_path 实现**：不创建 Agent Loop，直接调用 `rag_search_answer` Tool（一次性 L3→L4→L5→L6），延迟 <2s。

### 7.2 execute 节点（异步）

```python
async def execute_node(state: AgentState) -> dict:
    """异步执行 tool_calls。校验 deny_tools、confirmation、max_parallel_calls"""
    pending = state.get("pending_tool_calls", [])
    if not pending:
        return {}

    tool_policy = state["run_config"].tool_policy
    results: list[ToolResult] = []

    # 1. deny_tools → 生成审计错误（不静默丢弃）
    denied, rest = [], []
    for tc in pending:
        if tc.tool_name in tool_policy.deny_tools:
            denied.append(ToolResult(
                tool_call_id=tc.tool_call_id, tool_name=tc.tool_name,
                status="error",
                error=ToolError(code="tool_denied", message=f"{tc.tool_name} is denied by ToolPolicy", retryable=False),
                latency_ms=0,
            ))
        else:
            rest.append(tc)
    results.extend(denied)

    # 2. 需要确认的 tool → 检查是否已确认；未确认的触发 pause
    confirmed = state.get("confirmed_tool_call_ids", set())
    needs_confirmation = [
        tc for tc in rest
        if tc.tool_name in tool_policy.require_confirmation_for
        and tc.tool_call_id not in confirmed
    ]
    if needs_confirmation:
        # 返回 Command(goto='pause') —— 因为 execute→observe 是 normal edge，
        # 必须显式跳转到 pause 节点，否则 paused 状态会被 observe 吞掉
        return Command(
            goto="pause",
            update={
                "status": "paused",
                "needs_user_input": f"需要确认执行工具: {[tc.tool_name for tc in needs_confirmation]}",
                "pending_tool_calls": needs_confirmation,
            },
        )

    # 3. 按 max_parallel_calls 分批
    executables = rest[:tool_policy.max_parallel_calls]
    excess = rest[tool_policy.max_parallel_calls:]

    gathered = await asyncio.gather(
        *[_execute_one_tool(tc, state["run_config"]) for tc in executables],
        return_exceptions=True,
    )
    for i, result_or_exc in enumerate(gathered):
        if isinstance(result_or_exc, Exception):
            results.append(ToolResult(
                tool_call_id=executables[i].tool_call_id,
                tool_name=executables[i].tool_name,
                status="error",
                error=ToolError(code="internal", message=str(result_or_exc), retryable=True),
                latency_ms=0,
            ))
        else:
            results.append(result_or_exc)

    return {"tool_results": results, "pending_tool_calls": excess}
```

### 7.3 think → LLM 结构化输出

`plan` 和 `evaluate` 节点内部调用 LLM，要求结构化输出：

```python
async def evaluate_node(state: AgentState) -> dict:
    """评估证据，输出结构化决策"""
    handles = RuntimeRegistry.get(state["run_config"].run_id)
    budget_remaining = await handles.budget_ledger.remaining()

    prompt = _build_evaluate_prompt(
        task=state["task"],
        evidence=state["evidence"],
        iteration=state["iteration"],
        budget_remaining=budget_remaining,
    )
    try:
        decision: ThinkOutput = llm.with_structured_output(ThinkOutput).invoke(prompt)
    except ValidationError:
        decision = _fallback_evaluate(state)

    if decision.action == "execute":
        return {"pending_tool_calls": decision.tool_calls, "iteration": state["iteration"] + 1}
    elif decision.action == "synthesize":
        return {"status": "done", "stop_reason": decision.stop_reason}
    elif decision.action == "pause":
        return {"status": "paused", "needs_user_input": decision.needs_user_input}
```

注意：所有内部调用 `budget_ledger` 的节点都通过 `RuntimeRegistry.get(run_id)` 获取运行时 handles。

`plan` 节点同理，用 `ThinkOutput` 输出带依赖标注的 `TaskPlan`。

> **实现注意**：Section 7.3 的 evaluate_node（无 TaskDAG 路径）和 Section 9.3 的 evaluate_node（有 TaskDAG 路径）在实现时合并为一个异步函数。入口处先检查 `state.get("plan")`：如果为 None 走单 Agent 路径（调用 LLM 输出 ThinkOutput），否则走递归调度路径（查 terminal/successful/ready）。LLM 调用在单 Agent 路径中完成。

---

## 8. 中断/恢复协议

### 8.1 pause 节点

```python
def pause_node(state: AgentState) -> AgentState:
    """使用 LangGraph interrupt() 真正挂起图执行"""
    from langgraph.types import interrupt

    # interrupt() 保存 checkpoint，阻塞等待外部 Command(resume=...)
    user_decision = interrupt({
        "question": state.get("needs_user_input", "需要你的决策"),
        "context": {
            "task": state["task"],
            "evidence_count": len(state["evidence"]),
            "iteration": state["iteration"],
        },
        "options": [
            "continue_retrieval",
            "accept_and_complete",
            "switch_data_source",
        ],
    })
    # ── 从这里恢复执行（interrupt 返回用户注入的值）──
    return {
        "user_decision": user_decision,
        "status": "running",  # 恢复 running
    }
```

### 8.2 恢复协议

```python
# ── 场景 A：pause 由 LLM 决策触发（evaluate→pause）──
config = {"configurable": {"thread_id": run_id}}
graph.astream(
    Command(resume="continue_retrieval"),
    config=config,
)

# ── 场景 B：pause 由 execute_node 触发（工具需确认）──
# 用户在 CLI 确认后，写入 confirmed_tool_call_ids 再恢复
graph.astream(
    Command(
        resume="confirmed",
        update={"confirmed_tool_call_ids": {"tc_a1b2c3d4e5f6"}}
    ),
    config=config,
)
# 恢复后：pause → evaluate → execute（此时 check 发现已确认，正常执行）
```

**幂等性保证**：`interrupt()` 之前的所有逻辑必须是幂等的，因为恢复时会重跑 pause 节点。具体做法是 pause 节点不做任何副作用写操作（只读 state → 调 interrupt → 返回），确保重跑安全。

---

## 9. 并行执行与 Send API

### 9.1 TaskDAG

Plan 节点输出的 `TaskPlan` 包含依赖关系：

```python
class TaskPlan(BaseModel):
    subtasks: list[SubTaskNode]
    edges: list[TaskEdge]       # from_subtask_id → to_subtask_id

class SubTaskNode(BaseModel):
    subtask_id: str
    agent_type: str             # "research" | "compare" | "factcheck"
    prompt: str
    priority: int
    estimated_tokens: int | None = None  # 预估 token 消耗，用于 BudgetLedger.reserve()；None 时使用 AgentDefinition 的默认值

class TaskEdge(BaseModel):
    from_id: str
    to_id: str                  # to 依赖 from 的结果
```

### 9.2 execute_subagent 节点

`subtask` 字段由 Send 注入（非 AgentState 定义），节点完成后它会被下一次 Send 覆写。

```python
async def execute_subagent_node(state: AgentState) -> dict:
    """执行单个子 Agent。成功 commit 预算，失败 refund 预算。
    失败时只写 terminal_subtasks，不写 successful_subtasks → 依赖它的下游不会被调度"""
    subtask = state["subtask"]
    run_config = state["run_config"]

    agent_def = AgentRegistry.get(subtask.agent_type)
    compiler = AgentGraphCompiler(ToolRegistry, model_provider)
    graph = compiler.compile(agent_def)

    try:
        result = await graph.ainvoke(
            {"task": subtask.prompt},
            config={"configurable": {"thread_id": run_config.thread_id + "/" + subtask.subtask_id}},
        )
    except Exception as exc:
        handles = RuntimeRegistry.get(run_config.run_id)
        await handles.budget_ledger.refund(subtask.subtask_id)
        return {
            "subtask_results": {
                subtask.subtask_id: SubTaskResult(
                    subtask=subtask,
                    status=SubTaskStatus.FAILED,
                    error_message=str(exc),
                )
            },
            "terminal_subtasks": {subtask.subtask_id},
            # 不写 successful_subtasks → 依赖此 subtask 的下游不会被拓扑排序选中
        }

    # 成功：commit 预算 + 写入成功状态
    handles = RuntimeRegistry.get(run_config.run_id)
    actual_tokens = result.get("token_used", 0)
    await handles.budget_ledger.commit(subtask.subtask_id, actual_tokens)

    return {
        "evidence": result.get("evidence", []),
        "citations": result.get("citations", []),
        "subtask_results": {subtask.subtask_id: _to_subtask_result(subtask, result)},
        "terminal_subtasks": {subtask.subtask_id},
        "successful_subtasks": {subtask.subtask_id},
    }
```

`SubTaskResult` 新增失败字段：

```python
class SubTaskResult(BaseModel):
    subtask: SubTask
    status: SubTaskStatus = SubTaskStatus.PENDING
    findings: list[str] = Field(default_factory=list)
    evidence: list[EvidenceItem] = Field(default_factory=list)
    traces: list[ExecutionStepTrace] = Field(default_factory=list)
    error_message: str | None = None       # 仅在 status=FAILED 时写入
```
```

### 9.3 递归推进：预算预留放在 async 节点，conditional edge 只做纯路由

预算 `reserve()` 是 async 操作，必须在 `evaluate_node`（async）中完成，不能放在同步的 conditional edge 中。

```python
# ── 节点：写状态 + 预留预算（async）──
async def evaluate_node(state: AgentState) -> dict:
    dag = state.get("plan")
    if dag is None:
        return _single_agent_evaluate(state)

    successful = state.get("successful_subtasks", set())
    terminal = state.get("terminal_subtasks", set())
    all_terminal = all(st.subtask_id in terminal for st in dag.subtasks)

    if all_terminal:
        return {"status": "done", "stop_reason": "all_subtasks_terminal"}

    # _topological_ready 只检查 successful_subtasks → 失败子任务不满足下游依赖
    ready = _topological_ready(dag, successful)
    if not ready:
        # 还有子任务未完成但依赖未满足 → 异常或等待
        return {"status": "failed", "stop_reason": "deadlock_in_task_dag"}

    # 预算预留（async，在节点内完成，不在 conditional edge 中）
    handles = RuntimeRegistry.get(state["run_config"].run_id)
    schedulable: list[SubTaskNode] = []
    for st in ready:
        estimated = st.estimated_tokens or 8000
        if await handles.budget_ledger.reserve(st.subtask_id, estimated):
            schedulable.append(st)
        else:
            # 预算不足 → 该子任务直接失败，仍标记 terminal
            pass  # handled below

    return {
        "status": "running",
        "next_subtasks": schedulable,
        # 预算不足的子任务：标记为 terminal 但不成功，不解锁下游
        "terminal_subtasks": {st.subtask_id for st in ready if st not in set(schedulable)},
    }


# ── conditional edge：纯同步函数，只做路由，不调 async ──
from langgraph.types import Send

def route_after_evaluate(state: AgentState) -> str | list[Send]:
    if state["status"] == "done" or state["status"] == "failed":
        return "synthesize"
    if subtasks := state.get("next_subtasks"):
        return [
            Send("execute_subagent", {"subtask": st, "run_config": state["run_config"]})
            for st in subtasks
        ]
    return "execute"
```

### 9.4 并行结果合并

并行 `Send` 各自返回 `AgentState` 的 delta，LangGraph 自动用 `Annotated` reducer 合并。关键字段的合并行为：

| 字段 | 并行行为 |
|------|---------|
| `messages` | `add_messages` 按 msg ID 去重合并 |
| `evidence` | `_merge_evidence` 按 evidence_id 去重，冲突保留两者 |
| `citations` | `_merge_citations` 按 citation_id 去重 |
| `tool_results` | `_merge_tool_results` 按 tool_call_id 去重 |
| 其余字段 | 不并行写（只有 execute_subagent 节点写它们负责的 sub-state） |

---

## 10. KG 写工具独立治理

KG 写操作（`kg_upsert`、`create_artifact`）是系统最高风险工具，不能仅靠 Agent 级别的 `requires_user_confirmation`。

### 10.1 多层防护

```
Layer 1: ToolPermissions.kg_mutation = True → runtime 强制审计
Layer 2: ToolSpec.requires_confirmation = True → 每次执行前用户确认
Layer 3: ToolSpec.idempotent = True → 基于 content_hash 去重，防止重复写入
Layer 4: AgentRunConfig.access_policy → 检查是否允许写 KG
Layer 5: 写入时记录 provenance(agent_type, run_id, timestamp)
```

### 10.2 KG 写工具的 ToolSpec 示例

```python
kg_upsert_tool = ToolSpec(
    name="kg_upsert",
    description="Upsert a node or edge in the knowledge graph",
    input_model=KgUpsertInput,
    output_model=KgUpsertOutput,
    error_model=ToolError,
    permissions=ToolPermissions(kg_mutation=True, write_db=True),
    timeout_seconds=10.0,
    max_retries=2,
    idempotent=True,              # content_hash 去重
    requires_confirmation=True,   # 用户确认
    audit_log=True,               # 审计日志
)
```

### 10.3 Provenance 记录

每次 KG 写操作在 `metadata_json` 中强制写入：

```json
{
  "provenance": {
    "run_id": "abc123",
    "agent_type": "research",
    "parent_run_id": "xyz789",
    "timestamp": "2026-05-07T10:30:00Z",
    "content_hash": "sha256:def456"
  }
}
```

---

## 11. Data Flow

### 11.1 简单查询（快路径）

```
User Query
  │
  ▼
Orchestrator.route()            ← 复杂度分检：simple_lookup
  │
  ▼
fast_path_node                  ← 调用 rag_search_answer Tool
  │                               (一次调用 L3→L4→L5→L6)
  ▼
synthesize                      ← 返回带引用的答案
  │
  延迟 < 2s，不开 Agent Loop
```

### 11.2 复杂查询（多 Agent + 递归并行调度）

```
User Query: "对比 A 制度和 B 制度的公积金比例"
  │
  ▼
route_node                      ← decompose（复杂度分检）
  ▼
plan_node                       ← LLM 拆解为 TaskDAG:
  │                               s1 → s3
  │                               s2 → s3
  │
  ├─ evaluate: "还有未完成的子任务吗？"
  │            → 有！ready = [s1, s2]
  │
  │  Send(execute_subagent, s1) ──┬── 第1波并行
  │  Send(execute_subagent, s2) ──┘
  │       │                        │
  │       ▼                        ▼
  │  execute_subagent_node      execute_subagent_node
  │  各自写入:                         各自写入:
  │    subtask_results[s1]             subtask_results[s2]
  │    terminal_subtasks={s1}          terminal_subtasks={s2}
  │    successful_subtasks={s1}        successful_subtasks={s2}
  │
  ├─ evaluate: "s1,s2 成功，还有未完成的？(_topological_ready 只检查 successful)"
  │            → 有！ready = [s3]（s1,s2 都成功了，依赖满足）
  │
  │  Send(execute_subagent, s3)  ← 第2波（依赖 s1,s2）
  │       │
  │       ▼
  │  execute_subagent_node
  │  写入: subtask_results[s3], terminal_subtasks={s3}, successful_subtasks={s3}
  │
  ├─ evaluate: "所有子任务 terminal" → synthesize
  ▼
synthesize_node                  ← 合成最终报告
```

---

## 12. Built-in Agent Types

| Agent | 工具集 | 职责 |
|-------|--------|------|
| **ResearchAgent** | vector_search, keyword_search, grounding, rerank, llm_summarize | 深度单主题研究 |
| **CompareAgent** | vector_search, grounding, llm_compare | 多文档对比分析 |
| **FactCheckAgent** | grounding, keyword_search | 事实核查 + 引用验证 |
| **SynthesizeAgent** | llm_generate, llm_summarize | 不检索，合成已收集证据 |
| **Orchestrator** | research_agent, compare_agent, factcheck_agent, synthesize_agent, rag_search_answer | 拆解任务，调度子 Agent |

---

## 13. Memory + Context Management

### 13.1 四层模型

| Layer | 定位 | 存储/实现 | 阶段 |
|-------|------|-----------|------|
| **Instruction Memory** | Agent 行为规则、项目约束、工具策略提示 | 合并进 `AgentDefinition.system_prompt` 或项目文档；不新建运行时存储 | Phase A 只读取 |
| **Working Memory** | 当前 Agent run 的短期上下文压缩 | `AgentState.working_summary` + `extracted_facts` + `context_budget` | Phase A 实现 |
| **Episodic Memory** | 可复用的协作经验：用户偏好、反馈、项目背景、外部引用入口 | Phase B 新增长期记忆策略；只存过滤后的决策经验，不存执行 trace | Phase B 设计 |
| **Semantic Memory** | 领域事实候选，不作为新事实源 | 自动填充现有 `ArtifactRepo` / `GraphRepo`：低置信度写 `SUGGESTED` 或 candidate | Phase B 设计 |

`Semantic Memory` 不新建独立 store。项目已有 `ArtifactRepo` 存储可审阅的 `KnowledgeArtifact`，也已有 `GraphRepo` 存储结构化 `GraphNode` / `GraphEdge`。Memory extractor 只能作为自动写入路径：从 Agent 运行中抽取 candidate facts，再写入 `KnowledgeArtifact(status=SUGGESTED)` 或 `GraphRepo.save_candidate_edge()`。最终事实权威仍是 evidence、artifact 审批状态和 graph provenance。

`Episodic Memory` 不替代 `TelemetryService`。Telemetry 继续记录运行事件、工具耗时、分支命中、失败类型等全量或半全量事件；Episodic Memory 只保存未来会影响决策的稀疏经验，并通过 `source_event_id` / `source_trace_id` 关联来源事件。

### 13.2 Working Memory Compaction（Phase A）

Phase A 只解决长 Agent 上下文膨胀，不实现长期记忆写入。

触发条件：

- `messages` 估算 token 超过 `context_compaction_threshold`。
- 即将执行 `evaluate` / `plan` / `synthesize` 等 LLM 节点，且剩余上下文不足。
- 用户手动请求压缩或 `rag agent resume` 恢复长会话。

脱水产物：

- `working_summary`：旧消息的任务状态、关键决策、未完成事项。
- `extracted_facts`：只服务当前 run 的事实列表，可带 `evidence_ids`，但不会写入长期存储。
- `context_budget`：记录本次 prompt 中 system、evidence、working memory、tail messages、tool results 的 token 分配。

保留策略：

- 保留最近 N 条原始消息作为 tail window。
- 不切断 `tool_use` / `tool_result` 对；若 tail 起点落在 tool result，必须向前扩展到对应 tool call。
- 不丢弃未完成的 pending tool call、confirmation request、open decision。
- 压缩结果必须可序列化，不能包含 repo 连接、lock、event、provider client 等 runtime handle。

Phase A 目录只增加：

```text
rag/agent/memory/
├── models.py       # WorkingSummary, ExtractedFact, ContextBudgetSnapshot
├── compactor.py    # WorkingMemoryDehydrator：旧 messages → summary + facts + tail
└── injector.py     # ContextInjector：每次 LLM 调用前装配 bounded context
```

### 13.3 上下文装配顺序

每次 LLM 调用前由 `ContextInjector` 装配 bounded context。顺序必须体现：**RAG evidence 是事实权威，memory 只是历史线索**。

1. `AgentDefinition.system_prompt`
2. instruction memories / policy hints
3. 当前 task + user input
4. `RAG evidence + citations`
5. `working_summary + extracted_facts`
6. recalled episodic/reference memories（Phase B，标记为 historical hints）
7. 最近 N 条原始 `messages` tail
8. pending `tool_results` / open decisions

注入 recalled memories 时必须带边界提示：

```text
These memories are historical hints, not authoritative evidence.
If they conflict with retrieved evidence or current tool results, trust retrieved evidence.
```

### 13.4 写入策略（Phase B 设计，Phase A 不实现）

长期记忆写入只在明确价值场景触发：

- 用户明确要求“记住 / 以后都这样 / 忘掉这个”。
- 用户纠正系统行为，且该纠正未来可复用。
- 长任务结束后产生稳定项目背景、外部引用入口或可复用协作经验。
- Semantic candidate 有明确 evidence/provenance，可以进入 `ArtifactRepo` / `GraphRepo` 的候选流程。

写入前必须执行三步：

1. **分类**：`user` / `feedback` / `project` / `reference` / `domain_fact_candidate`
2. **去重**：先查相似 memory、artifact、graph candidate；能更新就不新建
3. **审计**：高风险、用户隐私、KG 写入默认 `SUGGESTED` / candidate，等待批准

冲突规则：

- `memory` vs `RAG evidence`：以 evidence 为准，memory 标记 `STALE` 或 `CONFLICTED`。
- `episodic memory` vs `TelemetryService`：Telemetry 保留原始事件，memory 保留决策总结；两者只通过 source id 关联。
- `semantic memory` vs `ArtifactRepo/GraphRepo`：不允许第二事实源；只走 artifact/graph candidate 写入。

---

## 14. Directory Layout

```
rag/agent/
├── __init__.py
├── schema.py                    # 公开数据模型（保留）
├── state.py                     # AgentState TypedDict + reducer + working memory fields
├── service.py                   # 入口 API（重构）
│
├── core/
│   ├── context.py               # AgentRunConfig, RuntimeRegistry, BudgetLedger
│   ├── definition.py            # AgentDefinition
│   ├── registry.py              # AgentRegistry
│   ├── compiler.py              # AgentGraphCompiler
│   └── agent_as_tool.py         # AgentAsTool, AgentToolSpec
│
├── memory/
│   ├── models.py                # WorkingSummary, ExtractedFact, ContextBudgetSnapshot
│   ├── compactor.py             # WorkingMemoryDehydrator
│   └── injector.py              # ContextInjector（bounded context 装配）
│
├── tools/
│   ├── spec.py                  # ToolSpec, ToolPermissions, ToolResult, ToolError
│   ├── registry.py              # ToolRegistry
│   ├── rag_tools.py             # vector_search, keyword_search, grounding, rerank, graph_expand
│   ├── llm_tools.py             # llm_generate, llm_summarize, llm_compare
│   ├── kg_tools.py              # kg_query, kg_upsert, create_artifact
│   └── agent_tools.py           # AgentAsTool 的注册入口
│
├── builtin/
│   ├── orchestrator.py          # Orchestrator AgentDefinition
│   ├── research.py              # ResearchAgent AgentDefinition
│   ├── compare.py               # CompareAgent AgentDefinition
│   ├── factcheck.py             # FactCheckAgent AgentDefinition
│   └── synthesize.py            # SynthesizeAgent AgentDefinition
│
├── graphs/
│   ├── base.py                  # 构建通用 route→execute→observe→evaluate→synthesize 图
│   ├── orchestrator.py          # Orchestrator 专用图（含 plan + Send 子图）
│   └── nodes/
│       ├── plan.py              # plan 节点实现
│       ├── route.py             # route 节点实现
│       ├── execute.py           # execute 节点实现（异步并行 tool calls）
│       ├── execute_subagent.py  # execute_subagent 节点实现（子 Agent 执行器）
│       ├── observe.py           # observe 节点实现
│       ├── evaluate.py          # evaluate 节点实现（含 ThinkOutput + 递归调度）
│       ├── pause.py             # pause 节点实现（含 interrupt）
│       └── synthesize.py        # synthesize 节点实现
│
├── planner.py                   # 废弃，逻辑迁移到 graphs/nodes/plan.py
├── executor.py                  # 废弃，逻辑迁移到 graphs/nodes/execute.py
├── critic.py                    # 废弃，逻辑迁移到 graphs/nodes/evaluate.py
├── synthesizer.py               # 废弃，逻辑迁移到 graphs/nodes/synthesize.py
├── understanding.py             # 保留，被 route 节点复用
└── report.py                    # 保留，被 synthesize 节点复用
```

---

## 15. CLI Interface

```bash
# 单次查询（快路径，不开 Agent Loop）
rag query "公积金政策是什么"

# Agent 模式
rag agent chat                          # 交互式对话（支持中断/恢复）
rag agent run "对比 A 和 B 政策" --json  # 单次运行，JSON 输出
rag agent resume <run_id>               # 恢复暂停的 Agent 运行
```

---

## 16. Risks & Mitigations

| 风险 | 缓解 |
|------|------|
| 并行 Send 结果冲突 | 所有可并行写字段使用 `Annotated` reducer（去重+排序+冲突标记） |
| pause 重跑非幂等操作 | pause 节点不写副作用，interrupt 前逻辑全部幂等 |
| 子 Agent 递归过深 | `max_depth` 硬限制（默认 2），每层 -1，≤0 拒绝嵌套 |
| Agent 循环失控 | `max_iterations` + `BudgetLedger.remaining()` 双重保护 |
| LLM 输出不遵循 ThinkOutput schema | `ValidationError` → fallback 评估逻辑 |
| KG 写工具滥用 | 五层防护：permissions + confirmation + idempotent + access_policy + provenance |
| 快路径误判复杂查询 | route 节点使用确定性规则 + `QueryUnderstandingService`，不用 LLM |
| messages 膨胀导致上下文溢出 | Phase A 引入 WorkingMemoryDehydrator，保留 bounded tail window，不切断 tool_use/tool_result 对 |
| memory 被当成事实权威 | ContextInjector 明确 `RAG evidence > memory`；冲突时 evidence 胜出，memory 标记 STALE/CONFLICTED |
| Semantic Memory 形成第二事实源 | 不新建 semantic store；只写入 `ArtifactRepo` / `GraphRepo` candidate 路径 |

---

## 17. Implementation Phases

| Phase | Scope | 关键交付物 |
|-------|-------|-----------|
| **Phase 1** | 三大契约层 | `ToolSpec`, `ToolPermissions`, `ToolResult`, `ToolError`, `AgentRunConfig`, `RuntimeRegistry`, `BudgetLedger`, `AgentDefinition`, `AgentState` + reducer |
| **Phase 2** | Tool 注册表 | `ToolRegistry` + 5 个 RAG Tool 的 `ToolSpec` + 基础实现 |
| **Phase 3** | 基础 Graph | `graphs/base.py`：route → execute → observe → evaluate → synthesize |
| **Phase 4** | Working Memory（Phase A） | `memory/models.py`, `compactor.py`, `injector.py`；bounded context + tail window |
| **Phase 5** | ResearchAgent | 第一个完整的端到端可运行 Agent |
| **Phase 6** | Orchestrator + 并行 | `plan` 节点 + `Send` 调度 + reducer 验证 |
| **Phase 7** | 中断/恢复 | `pause` 节点 + `interrupt()` + `Command(resume)` |
| **Phase 8** | 其余 Agent | CompareAgent, FactCheckAgent, SynthesizeAgent |
| **Phase 9** | CLI + 向后兼容 | `rag agent` 命令 + `AnalysisAgentService` 适配 |
| **Phase 10** | Long-term Memory（Phase B） | Episodic write policy；Semantic candidate 写入 `ArtifactRepo` / `GraphRepo` |

---

## 18. Change Log

| Version | Key Changes |
|---------|-------------|
| **v1** | 最初概念架构：AgentDefinition、AgentGraph、AgentAsTool |
| **v2** | 三大契约层：ToolSpec、AgentRunContext、AgentState(reducer)；interrupt()中断；结构化ThinkOutput；fast_path快路径；KG五层治理 |
| **v3** | `ToolCallPlan.tool_call_id` 全局唯一；`execute_node` async def；`BudgetLedger` 并发安全预算；`AgentDefinition` 完整字段（含 ModelPolicy/ToolPolicy）；`ToolResult` status 互斥（ok\|error）；`execute_subagent` 节点 + `subtask_results`/`completed_subtasks` reducer + evaluate 递归调度 |
| **v4** | BudgetLedger 死锁修复（asyncio.Lock + 锁内直接计算）；evaluate 接 conditional edge 返回 Send 列表（非 dict）；ToolResult 双重互斥校验；execute_subagent 失败 refund 预算；execute 节点 enforce ToolPolicy；统一 budget_ledger 引用 |
| **v5** | AgentRunConfig + RuntimeRegistry 分离；evaluate_node → async def；预算 reserve 移至 async 节点内；confirmation→pause/interrupt；deny_tools→ToolError；terminal/successful_subtasks 分离 |
| **v6** | AgentDefinition 加 access_policy/estimated_token_budget；confirmed_tool_call_ids 防无限 pause；Command(goto='pause')；next_subtasks/subtask 入 state；commit() 不截断超支 |
| **v7** | AgentRunConfig 字段顺序修正；RuntimeRegistry.get_or_create() 保证 handles 初始化；tool_policy 复制到 AgentRunConfig；evaluate_node 预算不足写 terminal_subtasks(非 completed_subtasks)；数据流图统一 terminal/successful_subtasks |
| **v8** | 增加 Memory + Context Management；Phase A 收敛为 Working Memory compaction/injection；Semantic Memory 复用 ArtifactRepo/GraphRepo candidate 路径；Episodic Memory 与 TelemetryService 边界明确 |
| **v9** (current) | AgentToolSpec 改为组合模式(非 frozen 继承)；SubTaskNode 增加 estimated_tokens；两个 evaluate_node 合并说明；confirmed_tool_call_ids 序列化注释 |

<div align="center">

# 面向私有业务文档的本地知识 Agent

面向私有知识库的 evidence-first RAG 与 LangGraph Agent 编排项目。

<p>
  <img src="https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.12">
  <img src="https://img.shields.io/badge/Package-uv-2b3137?style=for-the-badge" alt="uv">
  <img src="https://img.shields.io/badge/Agent-LangGraph-1f6feb?style=for-the-badge" alt="LangGraph">
  <img src="https://img.shields.io/badge/Vector-Milvus-00a1ea?style=for-the-badge" alt="Milvus">
  <img src="https://img.shields.io/badge/Metadata-PostgreSQL-336791?style=for-the-badge&logo=postgresql&logoColor=white" alt="PostgreSQL">
</p>

<p>
  <img src="https://img.shields.io/badge/Chat-deepseek--v4--flash-111827?style=for-the-badge" alt="deepseek-v4-flash">
  <img src="https://img.shields.io/badge/Embedding-Qwen3--4B--4bit-10b981?style=for-the-badge" alt="Qwen3 Embedding">
  <img src="https://img.shields.io/badge/Rerank-bge--reranker--v2--m3-f97316?style=for-the-badge" alt="bge reranker">
</p>

<p>
  <a href="#快速开始">快速开始</a> ·
  <a href="#当前默认运行配置">默认配置</a> ·
  <a href="#已验证端到端结果">端到端结果</a> ·
  <a href="#agent-编排层">Agent 设计</a> ·
  <a href="#历史基线结果与实验配置">历史基线</a> ·
  <a href="#目录地图">目录地图</a>
</p>

</div>

这个项目覆盖文档解析、结构化入库、摘要索引、混合检索、原文精读、表格计算、引用回答、离线评测，以及基于 LangGraph 的多 Agent 任务编排。目标是让企业内部制度、流程、销售资料、Word/PDF/Excel/PPT/图片等异构资料进入同一套可检索、可引用、可评测的知识系统，并为复杂问题提供可拆解、可追踪、失败显式化的 Agent 执行框架。

## News

- 2026-05-25：Agent 主图收口为模型驱动工具循环：`initialize_goal -> controller -> llm_decide/execute -> reduce -> controller -> finalize`。移除 capability 自动选路、父级 TaskDAG 编排和 fast-path 特殊节点；子 Agent 只经 `agent_*` 工具委派。
- 2026-05-23：新增通用资产分析工具链：`asset_list`、`asset_inspect`、`asset_analyze`。Agent 通过统一资产接口查看结构、读取样例、执行受限分析，而不是给 Excel/PDF/PPT 分别堆业务工具。
- 2026-05-23：撤掉表格 fast path 强推和“总计/合计/小计”这类硬编码语义规则。表格问题应走通用资产分析链路：先定位资产，再 inspect，再执行 SQL/Pandas 类计算，最后绑定证据。
- 2026-05-23：优化 Excel 入库性能：读取公式、有效列和 parquet 转换时跳过空尾列，避免十几 MB 报表被虚高 `max_column` 拖慢。
- 2026-05-23：本地 openai-compatible 生成不支持 structured generation 时，text fallback 会解析 JSON 字符串和 markdown code fence，避免把整段 JSON 当自然语言答案。
- 2026-05-23：`--reranker-model none` 时诊断明确关闭 rerank；当时的 Agent 控制节点增加独立 max token 控制，减少本地小模型调用延迟。
- 2026-05-20：README 补齐本地 Qwen / embedding / rerank 服务管理、私有文档入库、RAG 查询、Agent 查询、JSON diagnostics 和省内存运行手册。
- 2026-05-17：历史默认模型切到 `deepseek-v4-flash`、`mlx-community/Qwen3-Embedding-4B-4bit-DWQ`、`BAAI/bge-reranker-v2-m3`。
- 2026-05-16：完成真实 `PostgreSQL + parquet object store + Milvus` 端到端验证。


## 能力一览

| 能力 | 当前状态 | 关键实现 |
| --- | --- | --- |
| 多格式入库 | 已支持 PDF、Word、Markdown、Excel、PPT、图片、纯文本 | `rag/ingest/pipeline.py`、`rag/ingest/parsers/*` |
| 多粒度索引 | doc / section / asset 三类 summary index | Milvus collections + summary records |
| 混合检索 | 支持 `fast / auto / deep / asset` profile | L3 planning + L4 retrieval + rerank |
| Grounding | 原文回读、anchor replacement、neighbor expansion | `rag/retrieval/grounding_service.py` |
| 表格计算 | Excel asset 转 parquet，DuckDB 受限 `SELECT` | `table_sampler.py`、`table_executor.py` |
| 通用资产分析 | asset list / inspect / analyze，统一接触真实文件资产 | `rag/agent/tools/asset_tools.py` |
| Agent 编排 | AgentLoop + model-selected tools + approval pause | `rag/agent/*` |
| 评测 | 公开 MedicalRetrieval mini + 私有 329 条 golden queries | `scripts/evaluate_private_retrieval.py` |

## 系统流程

```mermaid
%%{init: {"theme": "base", "themeVariables": {"fontSize": "11px", "primaryColor": "#f8fafc", "primaryTextColor": "#0f172a", "primaryBorderColor": "#cbd5e1", "lineColor": "#94a3b8", "clusterBkg": "#ffffff", "clusterBorder": "#e2e8f0"}, "flowchart": {"nodeSpacing": 18, "rankSpacing": 22, "curve": "basis", "padding": 4}} }%%
flowchart TB
    subgraph RAG["RAG pipeline"]
        direction LR
        r1["文件"] --> r2["解析"] --> r3["切片"] --> r4["摘要"] --> r5["向量"] --> r6["检索"] --> r7["证据"] --> r8["回答"]
    end

    subgraph AG["Agent orchestration"]
        direction LR
        a1["定义"] --> a2["目标检查"] --> a3["模型选工具"] --> a4["执行"] --> a5["观察归并"] --> a6["答案"]
    end

    classDef rag fill:#eef6ff,stroke:#7aa7d9,color:#0f172a;
    classDef agent fill:#f4f0ff,stroke:#a78bfa,color:#0f172a;
    class r1,r2,r3,r4,r5,r6,r7,r8 rag;
    class a1,a2,a3,a4,a5,a6 agent;
```

RAG 负责把原始资料变成可引用证据，Agent 运行在 RAG 能力之上，用工具契约、预算和 LangGraph 状态流转处理复杂任务。

## 架构总览

这个项目的主要业务场景是私有资料问答和文件分析：制度/流程文档回答审批规则、费用报销、销售政策；销售日报 Excel 回答提货量、区域汇总、产品口径；PPT/Word/PDF 回答跨文档事实并给出处。系统分成两层：RAG 负责把资料定位成可信证据，Agent 在证据和资产之上做多步分析、工具调用和结果校验。

```text
L1 Storage：事实层
  原始文件、Document、SectionRecord、AssetRecord、locator、权限、版本、处理状态

L2 Indexing：索引层
  制度文档摘要、正文 section 摘要、Excel/PPT 表格资产摘要 -> Embedding -> Milvus

L3 Planning：查询规划
  判断普通制度问答、复杂多跳问题、表格/资产问题，选择 fast / auto / deep / asset

L4 Retrieval：候选召回
  多粒度 summary 检索、候选清洗、RRF/融合、可选 rerank、召回诊断

L5 Grounding：证据回读
  回读制度原文、邻近 section、Excel/PPT/图片资产、asset anchor、受 token 预算控制

L6 Synthesis：回答合成
  基于 EvidenceItem 生成最终回答、引用、权限/合规复核

Agent Layer：任务执行
  AgentLoop、ResearchAgent、通用资产工具、agent-as-tool、BudgetLedger、working memory
```

### L1：事实层

L1 保存事实数据和可追溯定位信息。制度条款、报销审批规则、销售政策正文都落在 `Document / SectionRecord`；Excel sheet、PPT 表格、图片 OCR 区域等非正文内容落在 `AssetRecord`：

- `Document`：文档版本、权限、状态、来源。
- `SectionRecord`：正文窗口，带 `raw_locator`、byte range、token 窗口元数据。
- `AssetRecord`：表格、图片、OCR 区域、PPT 表格等非正文资产。
- Object Store：保存原始文件、visible text、表格对象、schema/sample 和 DuckDB 可读存储指针。

### L2：索引层

L2 保存检索入口。Milvus 中按粒度拆成三类 summary index，分别解决“先找哪份制度”“定位哪一节原文”“定位哪张表/哪页 PPT/哪个图片区域”的问题：

- `doc_summary`：文档级主题召回。
- `section_summary`：正文 section 召回。
- `asset_summary`：表格、图片、OCR、PPT 资产召回。

索引层保存 summary、向量、标量过滤字段和主键映射。原文、表格和权限信息仍由事实层提供。

### L3/L4：规划与检索

L3 判断查询应该如何检索，L4 负责候选召回和排序。比如审批规则问题通常走 `auto`，销售日报/Excel 数字问题优先走 `asset`，跨多个制度对比才需要更深的检索或 Agent 拆解。系统支持这些 `retrieval_profile`：

- `fast`
- `auto`
- `deep`
- `asset`

规划层处理复杂度、语义路由、版本过滤和谓词下推；检索层对 doc / section / asset summary 做多路召回、候选清洗、融合和 rerank。

### L5：精读与证据层

L5 将 summary 命中的候选重新映射回原始正文或资产对象，确保最终答案不是只基于摘要猜测：

- 命中正文 section 后，通过 `visible_text_key + byte_range` 回读原文。
- 命中含表格锚点的 section 后，通过 `[ASSET_ANCHOR:...]` 找到绑定资产。
- 表格资产通过 DuckDB Text-to-SQL Sandbox 执行受限查询。
- grounding 阶段受 token、目标数、并发和超时预算控制。

### L6：回答合成层

L6 只基于 `EvidenceItem` 合成回答。回答保留 `doc_id / section_id / asset_id`、citation anchor、检索分数、rerank 分数和 evidence metadata，便于复查“这个审批结论来自哪份制度哪一节”或“这个汇总数字来自哪个 Excel sheet”。

## 核心设计

### Summary-First, Grounding-Later

先用高密度 summary 做轻量召回，再回原文和资产对象精读。summary 负责定位，最终事实来自 grounding 后的 evidence。

### Facts in Storage, Search in Index

PostgreSQL / Object Store 保存事实；Milvus 保存向量索引和检索入口。原文、表格、定位、权限、版本归事实层，向量、BM25、标量过滤归索引层。

### Token-First

切分、窗口、摘要输入输出、grounding budget 和 Agent context budget 都按 token 控制：

- SectionRefiner 按 token 滑动窗口。
- eval 出题按 token 二次窗口。
- 摘要输入输出按 token 裁剪。
- L5 grounding 和 Agent BudgetLedger 都按 token 记账。

### Asset-Aware Retrieval

表格、图片、OCR、PPT 表格都作为 `AssetRecord` 独立保存。正文中保留 `[ASSET_ANCHOR:...]`，精读阶段再解析锚点并回填对应资产 evidence。

### DuckDB Table Sandbox

表格资产以 `schema / sample_rows / row_count / column_count / storage_key` 进入上下文。涉及过滤、排序、聚合、排名、交叉对比的问题，由 LLM 生成受限 `SELECT`，交给 DuckDB Sandbox 执行，再将计算结果交给合成层。

### Evidence Over Memory

Agent memory 用于 working memory compaction / injection，帮助控制上下文窗口。回答事实优先级为 RAG evidence 高于 memory；当两者冲突时，以 evidence 为准。

## Agent 编排层

Agent 层采用 Tool-Centric + Python while-loop kernel 设计。LangGraph 保留为外层复杂编排器，不再承担单 Agent 的逐轮控制。

### 设计说明

Agent 层不是单独绕开 RAG 的聊天入口，而是把 L3-L6 能力封装为可验证工具，并由 `AgentLoop` 管理单 Agent 的模型、工具、审批、checkpoint 和停止状态。设计目标是让每一次 Agent run 都能回答这些问题：用了哪个 `AgentDefinition`、允许哪些工具、消耗多少预算、哪些工具需要审批、召回了哪些 evidence、哪些 citation 被带入最终回答、为什么停止。

核心边界如下：

- `AgentDefinition` 只描述 Agent 类型、系统提示、工具 allowlist、模型选择、最大迭代、最大嵌套深度和工具策略。
- `ToolSpec` 只描述工具契约，包括 Pydantic 输入输出、权限、timeout、retry、预算成本和是否需要确认。
- `AgentRunConfig` 保存可序列化运行配置；`RuntimeRegistry` 保存不可序列化 runtime handles，例如 `BudgetLedger` 和 cancellation event。
- `LoopState` 使用 TypedDict 保存有界且可检查的 messages、evidence、citations、tool results、structured observations、审批和 checkpoint 元数据。
- `ModelTurnProvider` 每轮只能返回 `execute(tool_calls)`、`finish(final_answer)` 或 `pause(reason)`；`finish` 必须携带非空答案。
- `GoalSpec` 只在调用方显式提供时安装 stop hook，不参与默认循环路由。
- `AgentDelegationRequest` 只表达由普通 `agent_*` 工具发起的一次有界子 Agent 调用；父循环不维护子任务图。
- `AgentService` 是外部调用边界，负责构造初始状态、装配 `AgentLoop`、注入 request-scoped tool registry，并返回结构化 `AgentRunResult`。

运行链路如下：

```text
AgentRunRequest
  -> AgentRunConfig
  -> AgentService.initial_state()
  -> AgentLoop
     -> pending tool calls? execute with approval and execution records
     -> extract bounded StructuredObservation
     -> compact context when required
     -> ModelTurnProvider.next_turn()
        -> execute: checkpoint and continue
        -> finish: run explicit stop hooks, then complete
        -> pause: persist pause state for resume
  -> AgentRunResult
```

旧 `route/plan/evaluate`、`ToolDecisionProvider` 和独立 synthesis finalizer 不再属于主控制流。`retrieval_hint` 只生成检索元数据；模型在普通循环中直接选择工具或返回完整最终答案。需要专门综合时，模型可以显式调用 `agent_synthesize` 子 Agent 工具，内核不会在 `finish` 缺少答案时隐式启动另一个 Agent。

工具执行遵循 fail-closed 策略。工具未注册、runner 缺失、输入非法、输出非法、权限被拒、审批未通过、预算不足、timeout 或 runner 异常都会变成结构化 `ToolResult(status="error")` 或显式失败状态，不做静默 fallback。写库、知识图谱变更、外部副作用等高风险工具通过 `ToolPermissions`、`requires_confirmation` 和 `approval_policy` 进入暂停/恢复链路。

子 Agent 通过 agent-as-tool 接入父图。`build_agent_tool_spec()` 会把子 Agent 包成普通工具契约，`AgentAsToolAdapter` 在每个 request 的 runtime registry 里注入，避免并发 run 互相污染预算、深度和访问策略。父 Agent 派发子任务时会通过 `derive_child_config()` 限制嵌套深度，并继承必要的 access policy 和 source scope。

RAG evidence 是事实优先级最高的上下文。Working memory 只用于压缩当前 run 的历史消息和抽取线索，不能覆盖 grounding 后的 evidence。最终回答必须保留 evidence、citation、retrieval score、rerank score 和 grounding metadata，便于复查和评测。

文件和表格分析的长期目标是 Codex-like 的通用资产分析能力：少量稳定通用工具 + 多个底层资产适配器 + 统一中间表示 + 可执行分析环境。Agent 应能直接接触文件，能 inspect，能读取局部内容，能执行 SQL/Pandas 类分析，能把结果绑定回证据。它不应该靠给每种文件、每个问题、每个字段不断加规则。

当前通用工具边界：

- `rag_search_answer`：单次 RAG 定位和回答，适合普通制度问答或快速 smoke。
- `asset_list`：按 doc/source/section 列出已入库资产。
- `asset_inspect`：查看资产结构、列、样例行、可分析能力。
- `asset_read_slice`：按行列范围读取受限局部内容，并返回可回绑引用的定位信息。
- `asset_analyze`：对支持的资产执行受限只读分析，目前支持 `dataframe_sql`。

底层文件差异应由 adapter 处理，例如 Excel adapter 负责 sheet/cell/formula/parquet，PDF adapter 负责 page/text block/table/image，PPT adapter 负责 slide/shape/table/notes。Agent 层保持统一的 `Asset / Block / TableBlock / TextBlock / Locator / EvidenceRef / AnalysisResult` 这类中间表示，不增长成一堆资产专用业务工具。

### 已实现

- 工具契约：`ToolSpec / ToolPermissions / ToolResult / ToolError` 描述工具输入、输出、权限、错误、预算和重试策略。
- 工具注册与执行：`ToolRegistry` 支持注册工具 spec 和 callable runner，并执行 Pydantic 输入/输出校验。
- 运行配置：`AgentRunConfig / RuntimeRegistry / BudgetLedger` 区分可序列化 run config 和不可序列化 runtime handles。
- Agent 定义：`AgentDefinition / ModelPolicy / ToolPolicy` 描述 Agent 类型、系统提示、工具 allowlist、模型偏好和工具策略。
- 状态契约：`AgentState` 使用 TypedDict 和 reducer 合并 messages、evidence、citations、tool results 与结构化 observations。
- 工具委派契约：`AgentDelegationRequest / DelegatedAgentRunner` 为 `agent_*` 工具执行一次受预算和深度约束的 child run。
- Agent 组合：`AgentRegistry / AgentToolSpec / AgentAsToolRunner` 支持注册 Agent，并把子 Agent 封装成可调用工具。
- 图执行骨架：LangGraph base graph 已迁移为 `GoalInitializer -> Controller Loop -> execute/reduce/check/llm_decide/finalize`。
- 子 Agent 工具化：父循环通过 `ToolExecutor` 执行 `agent_*` 工具，child 结果以普通 tool observation 回到父状态。
- Goal Satisfaction Runtime：已实现 `GoalSpec`、`StructuredObservation`、`StateReducer`、`SatisfactionChecker` 和 gap-aware context injection。
- Working memory Phase A：已实现 working summary、extracted facts 和 bounded context injection。
- 内置 ResearchAgent：已提供 research AgentDefinition、RAG/LLM tool specs、builtin tool registry 和 service factory。

### 已具备的能力

- 可以用统一契约定义 Agent、工具、运行预算、访问策略、工具权限和执行位置偏好。
- 可以运行基础 LangGraph Agent 流程，由目标满足度驱动工具执行、状态归约、必要的 LLM 决策、暂停和最终结果表达。
- 可以执行已注册工具，并在工具未注册、runner 缺失、参数非法、输出非法、预算不足、timeout、runner 异常时返回结构化失败结果。
- 可以把子 Agent 作为工具调用，支持父 run 向子 run 派发任务、继承必要上下文并回收子任务结果。
- 可以让模型在同一工具循环中选择检索、资产分析或 `agent_*` 委派，并由父循环统一归并 observations 与证据。
- 可以对长上下文做 working memory compaction / injection，并优先向 LLM 注入 structured observations、open gaps、satisfied requirements 和必要 evidence，而不是默认塞完整工具输出。
- 可以保证回答事实以 RAG evidence 为最高优先级，memory 只作为当前 run 的上下文线索。
- 可以通过 `AgentService` 返回结构化 `AgentRunResult`，包含 status、final answer、stop reason、tool results、evidence、citations 和 groundedness flags。
- 可以使用内置 ResearchAgent 作为当前可运行 Agent 模板，后续 Agent 可以复用同一套契约和图执行框架。

### 下一步规划实现

- 完成通用文件分析 Agent 主链路：RAG 定位资产 -> `asset_inspect` 读取结构 -> `asset_read_slice` 按需读取局部 -> `asset_analyze` 执行只读分析 -> evidence/citation 回绑。
- 补齐更多底层资产 adapter：PDF 表格、PPT 表格/备注、Word 表格、图片 OCR 区域都转成统一 `Asset / Block / Locator / EvidenceRef / AnalysisResult`。
- 完善 LLM tool-decision provider 和 golden case 评测，重点覆盖制度问答、报销审批、销售日报 Excel 读数/汇总、歧义口径澄清。
- 完整接入 LangGraph `interrupt()`、checkpointer、`Command(resume=...)`，支持需要人工确认的工具审批和恢复。
- 扩展 Orchestrator、CompareAgent、FactCheckAgent、SynthesizeAgent，但复用同一套工具契约和 evidence 优先级。
- 完善 Agent CLI/API，让外部调用方能稳定启动 run、查看状态、恢复任务、读取结构化结果。
- 推进长期 memory Phase B：写入策略、去重、冲突标记，以及 memory 与 RAG evidence 的优先级治理。

## 当前能力

### 文档入库

支持这些文件类型：

- `.pdf`
- `.docx`
- `.md / .markdown`
- `.xlsx / .xls`
- `.pptx`
- `.png / .jpg / .jpeg`
- `.txt`

解析路径：

- Word / PDF / Markdown：Docling 结构树和标题分段。
- Excel：Pandas / OpenPyXL 读取 sheet，表格作为 asset。
- PPTX：`python-pptx` 解析 slide 文本、表格、备注。
- 图片：OCR 模块抽取 visible text 和 OCR region。

### 检索与回答

- 三类 summary index：doc / section / asset。
- 支持 retrieval profile：`fast / auto / deep / asset`。
- 支持 rerank、candidate cleanup、neighbor expansion。
- grounding 回读原文 byte range 和 asset anchor。
- 表格查询走 DuckDB Text-to-SQL Sandbox。
- 最终回答基于 `EvidenceItem`，保留 citation 和 metadata。

### 评测

- 公开 benchmark：MedicalRetrieval mini。
- 私有制度数据：329 条 golden queries。
- 支持按题型拆分指标，观察 doc hit、section hit、MRR、rerank 消融和 top-k 扩展效果。

## 历史基线结果与实验配置

下面是之前已经跑过的基线结果，保留用于和当前 `configs/models.yaml` 默认链路做横向对比。这里的模型和存储配置是历史实验快照，不代表当前默认运行配置；切换 embedding 模型后必须重建对应索引。

### 公开数据：MedicalRetrieval mini

| 基线 | 向量后端 | embedding | rerank | Recall@10 | MRR@10 | NDCG@10 | avg_latency_ms |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| `BAAI/bge-m3 + sqlite` | SQLite | `BAAI/bge-m3` | on | 0.776667 | 0.690972 | 0.712199 | 2472.225 |
| `BAAI/bge-m3 + milvus` | Milvus | `BAAI/bge-m3` | on | 0.670000 | 0.588259 | 0.608173 | 563.793 |
| `qwen3-embedding:8b + milvus` | Milvus | `qwen3-embedding:8b` | on | 0.820000 | 0.705854 | 0.733644 | 695.559 |

公开 qwen8b baseline 当时的复现实验配置：

| 项 | 值 |
| --- | --- |
| storage root | `data/benchmarks/medical_retrieval/index/mini-milvus-qwen8b-v1` |
| Milvus collection prefix | `medical_retrieval_mini_qwen8b_v1` |
| vector backend | `milvus` |
| vector dsn | `http://127.0.0.1:19530` |
| embedding provider / model | `ollama` / `qwen3-embedding:8b` |
| summary provider / model | `local-hf` / `Qwen/Qwen3-8B-MLX-4bit` |
| summary backend | `mlx` |
| chunk window | `480` tokens, overlap `64` tokens |
| graph extraction | skipped |
| retrieval profile | `auto` |
| evaluate top_k / evidence_top_k | `10` / `20` |
| rerank provider / model | `local-bge` / `BAAI/bge-reranker-v2-m3` |

### 私有数据：公司制度/销售中心资料

黄金测试集：

- query_count：`329`
- 题型分布：
  - `single_section_fact`：163
  - `rule_condition`：105
  - `process_step`：61

整体结果：

| 配置 | top_k | rerank | doc_mrr | section_mrr | doc_hit@1 | doc_hit@3 | doc_hit@5 | doc_hit@10 | section_hit@1 | section_hit@3 | section_hit@5 | section_hit@10 |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| sqlite rerank | 10 | on | 0.9300 | 0.7588 | 0.8967 | 0.9574 | 0.9757 | 0.9818 | 0.6535 | 0.8602 | 0.8875 | 0.9027 |
| sqlite no rerank | 10 | off | 0.9127 | 0.7200 | 0.8663 | 0.9574 | 0.9757 | 0.9818 | 0.6049 | 0.8359 | 0.8723 | 0.9027 |
| sqlite no rerank | 20 | off | 0.9127 | 0.7200 | 0.8663 | 0.9574 | 0.9757 | 0.9818 | 0.6049 | 0.8359 | 0.8723 | 0.9027 |

按题型结果，开启 rerank：

| 题型 | query_count | doc_mrr | section_mrr | doc_hit@1 | doc_hit@3 | doc_hit@10 | section_hit@1 | section_hit@3 | section_hit@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `process_step` | 61 | 0.9727 | 0.7865 | 0.9508 | 1.0000 | 1.0000 | 0.7049 | 0.8689 | 0.9344 |
| `rule_condition` | 105 | 0.9221 | 0.7929 | 0.8857 | 0.9524 | 0.9810 | 0.6762 | 0.8952 | 0.9429 |
| `single_section_fact` | 163 | 0.9192 | 0.7265 | 0.8834 | 0.9448 | 0.9755 | 0.6196 | 0.8344 | 0.8650 |

结论：

- 文档级命中稳定，`doc_hit@10` 达到 0.9818。
- section 级 top10 命中达到 0.9027。
- rerank 对 section 级排序有明显帮助，`section_mrr` 从 0.7200 提升到 0.7588。
- `top_k=20` 对当前私有数据没有带来额外 top20 命中，后续优化重点在候选生成和 section top1 排序。

旧私有文档实验配置快照：

| 项 | 值 |
| --- | --- |
| storage root | `data/eval_private/company_policy_milvus_v4` |
| Milvus collection prefix | `company_policy_v4` |
| vector backend / dsn | `milvus` / `http://127.0.0.1:19530` |
| embedding provider / model | `ollama` / `qwen3-embedding:8b` |
| summary provider / model | `local-hf` / `Qwen/Qwen3-8B-MLX-4bit` |
| summary backend | `mlx` |
| chunk window | `800` tokens, overlap `120` tokens |
| ingest batch / embedding batch | `8` / `8` |
| golden dataset | `data/eval_private/golden_eval_dataset_v4.jsonl` |
| failed golden output | `data/eval_private/golden_eval_failed_v4.jsonl` |
| section export | `data/eval_private/company_policy_sections_v4.jsonl` |
| retrieval profile | `auto` |
| evaluate top_k / retrieval_pool_k | `20` / `20` |
| neighbor radius | `1` |
| rerank provider / model | `local-bge` / `BAAI/bge-reranker-v2-m3` |
| eval output | `data/eval_private/private_retrieval_eval_v4_rerank.json` |
| misses output | `data/eval_private/private_retrieval_misses_v4_rerank.jsonl` |

旧测试题生成配置：

| 项 | 值 |
| --- | --- |
| generator model | `Qwen/Qwen3-8B-MLX-4bit` |
| server | OpenAI-compatible `mlx_lm.server` |
| base URL | `http://127.0.0.1:8080/v1` |
| max tokens | `1024` |
| temperature | `0.1` |
| thinking | disabled via `{"enable_thinking":false}` |
| max window / overlap / min window | `700` / `80` / `120` tokens |

## 当前默认运行配置

模型目录统一在 `configs/models.yaml` 中维护，业务代码不直接写 provider、模型名、base URL 或 API key。

当前默认：

- `defaults.primary_model`：`mimo_cloud`，作为未指定任务模型时的 fallback。
- 摘要 / 回答 / planner / synthesize / factcheck：`qwen3_8b_mlx_4bit`
- Embedding：`mlx-community/Qwen3-Embedding-4B-4bit-DWQ`
- Rerank：`BAAI/bge-reranker-v2-m3`

真实端到端推荐链路：

```text
PostgreSQL metadata
  + local object store / parquet table assets
  + Milvus vector indexes
  + Redis cache
  + DeepSeek chat
  + MLX embedding
  + BGE rerank
```

表格 / 资产分析规则：

- Excel 入库后表格资产会记录 `row_count / column_count / schema / sample_rows / storage_key`。
- 表格资产会转换为 DuckDB 可读的 `.parquet` 对象。
- 涉及真实数据值、筛选、求和、计数、排序、排名、对比或聚合的问题，必须走 `<compute_request>`。
- DuckDB 执行 `SELECT` 后会把 `TABLE_COMPUTE_RESULT` 和 `Executed SQL` 注入证据，再生成最终回答。
- `sample_rows` 只用于识别 schema，不允许被当成完整表格直接回答。
- 不允许通过“总计/合计/小计”等业务关键词硬编码来修某一张表；如果问题缺少产品、sheet、日期或统计口径，应暴露歧义或要求澄清。

## 已验证端到端结果

最近一次真实链路验证：

- Postgres schema：`rag_e2e_20260516_150131`
- Milvus collection prefix：`rag_e2e_20260516_150131`
- Milvus collections：
  - `rag_e2e_20260516_150131__doc_summary__default`
  - `rag_e2e_20260516_150131__section_summary__default`
  - `rag_e2e_20260516_150131__asset_summary__default`
- 表格对象：`data/e2e_agent_pq_milvus/20260516_150131/objects/*.parquet`
- `sqlite_vector_index_used: false`

验证问题：

| 类型 | 问题 | 结果 |
| --- | --- | --- |
| RAG 制度问答 | 单笔国内差旅报销金额超过 12000 元需要谁审批？ | 命中制度原文，回答为业务线 VP |
| RAG SLA 问答 | P0 客户生产故障的首次响应目标和恢复目标分别是多少？ | 命中 `15 分钟` 和 `2 小时` |
| 表格计算 | 请计算华东区域 Q1 的开票量合计是多少？ | 触发 DuckDB SQL，返回 `375` |

表格计算执行证据：

```sql
SELECT SUM("开票量") FROM sheet WHERE "区域"='华东' AND "季度"='Q1'
```

返回：

```text
TABLE_COMPUTE_RESULT
sum("开票量") = 375
```

## 安装

安装依赖：

```bash
uv sync
```

准备 `.env`：

```bash
cat > .env <<'EOF'
DEEPSEEK_API_KEY=your_deepseek_key
MIMO_API_KEY=your_mimo_key
EOF
```

确认基础设施：

```bash
lsof -nP -iTCP:19530 -sTCP:LISTEN
lsof -nP -iTCP:5432 -sTCP:LISTEN
lsof -nP -iTCP:6379 -sTCP:LISTEN
```

默认本地端口：

| 服务 | 端口 | 说明 |
| --- | ---: | --- |
| Milvus | `19530` | 向量索引 |
| Milvus Web/metrics | `9091` | 已被 Milvus docker 占用，不要给 rerank 用 |
| Postgres | `5432` | metadata |
| Redis | `6379` | cache |
| Qwen generation service | `8080` | `Qwen/Qwen3-8B-MLX-4bit` |
| Embedding service | `9090` | `mlx-community/Qwen3-Embedding-4B-4bit-DWQ` |
| Rerank service | `9092` | `BAAI/bge-reranker-v2-m3` |

## 模型服务管理

当前默认模型配置在 `configs/models.yaml`：

| 能力 | 默认别名 | 实际模型 / 服务 |
| --- | --- | --- |
| 生成 / 摘要 / Agent 路由 | `qwen3_8b_mlx_4bit` | `Qwen/Qwen3-8B-MLX-4bit`，OpenAI-compatible，`127.0.0.1:8080` |
| Embedding | `qwen3_embedding_4b_4bit_dwq` | `mlx-community/Qwen3-Embedding-4B-4bit-DWQ`，HTTP service，`127.0.0.1:9090` |
| Rerank | `bge_reranker_v2_m3` | `BAAI/bge-reranker-v2-m3`，HTTP service，`127.0.0.1:9092` |

内存策略：

- 不要一次把生成模型、embedding、rerank 都常驻，Mac 内存容易爆。
- 入库阶段需要生成摘要时：启动 Qwen 生成服务 + embedding 服务；不要启动 rerank。
- 普通 RAG / Agent 查询：启动 Qwen 生成服务 + embedding 服务；rerank 默认先关。
- 只有做排序质量评测或需要更强重排时，才单独启动 rerank。
- 切换 embedding 模型后必须换新的 Milvus collection prefix，旧向量不能混用。

先检查是否已经有同模型服务，避免重复常驻占内存：

```bash
ps aux | rg -i 'embedding-service|rerank-service|Qwen3-Embedding|bge-reranker|mlx_lm|vllm|ollama|uvicorn' \
  | rg -v 'rg -i|exec_command'

lsof -nP -iTCP -sTCP:LISTEN \
  | rg ':(8080|8081|8000|8001|9090|9091|9092|11434|19530|5432|6379)\b' || true
```

如果发现重复的 Qwen / embedding / rerank 服务，先杀旧进程。不要杀 Milvus、Postgres、Redis：

```bash
kill <old_qwen_pid> <old_embedding_pid> <old_rerank_pid>
```

启动 Qwen 生成服务。`enable_thinking=false` 很重要，否则 Qwen3 可能把内容放到 reasoning 字段，导致摘要生成空输出：

```bash
screen -S rag_qwen_8080 -X quit >/dev/null 2>&1 || true
screen -dmS rag_qwen_8080 zsh -lc '
cd "/Users/leixiaoying/LLM/RAG学习"
uv run python -m mlx_lm.server \
  --model Qwen/Qwen3-8B-MLX-4bit \
  --host 127.0.0.1 \
  --port 8080 \
  --chat-template-args '"'"'{"enable_thinking": false}'"'"'
'
```

启动 embedding 服务。内存紧张时用 `--batch-size 1`，更稳；内存充足时可以调到 `2/4/8`：

```bash
screen -S rag_embedding_9090 -X quit >/dev/null 2>&1 || true
screen -dmS rag_embedding_9090 zsh -lc '
cd "/Users/leixiaoying/LLM/RAG学习"
uv run rag embedding-service \
  --model mlx-community/Qwen3-Embedding-4B-4bit-DWQ \
  --port 9090 \
  --batch-size 1
'
```

rerank 是可选服务。先不要开；需要重排时再启动。注意 `9091` 被 Milvus 占用，rerank 用 `9092`：

```bash
screen -S rag_rerank_9092 -X quit >/dev/null 2>&1 || true
screen -dmS rag_rerank_9092 zsh -lc '
cd "/Users/leixiaoying/LLM/RAG学习"
uv run rag rerank-service \
  --model BAAI/bge-reranker-v2-m3 \
  --port 9092 \
  --batch-size 4 \
  --max-length 1024
'
```

健康检查：

```bash
curl -sS http://127.0.0.1:8080/v1/models
curl -sS http://127.0.0.1:9090/health
curl -sS http://127.0.0.1:9092/health
screen -ls
```

预期返回：

```text
{"object":"list","data":[...]}
{"model":"mlx-community/Qwen3-Embedding-4B-4bit-DWQ","embedding_space":"mlx/Qwen3-Embedding-4B-4bit-DWQ","dimension":2560}
{"model":"BAAI/bge-reranker-v2-m3"}
```

关闭服务：

```bash
screen -S rag_qwen_8080 -X quit >/dev/null 2>&1 || true
screen -S rag_embedding_9090 -X quit >/dev/null 2>&1 || true
screen -S rag_rerank_9092 -X quit >/dev/null 2>&1 || true
```

如果只想临时释放某个模型的内存，关对应 screen 即可。例如入库完成后可以先关 Qwen，之后要生成答案时再开；做纯检索评测时可以不开 Qwen。

## 私有文档端到端运行手册

先启动 Qwen 和 embedding 服务；rerank 默认不开，需要时再按“常用开关”打开。

### 统一变量

入库和查询必须使用同一套 `STORAGE_ROOT / VECTOR_DSN / VECTOR_PREFIX`。切换 embedding 模型或想重建干净索引时，换新的 `STORAGE_ROOT` 和 `VECTOR_PREFIX`。

```bash
cd "/Users/leixiaoying/LLM/RAG学习"

# 数据位置：按实际数据改这两个变量。
export INPUT_PATH="/absolute/path/to/one-file.docx"
export INPUT_DIR="/absolute/path/to/private-docs"

# 索引位置：同一批入库和查询必须保持一致。
export STORAGE_ROOT="data/indexes/private_docs_v1"
export VECTOR_DSN="http://127.0.0.1:19530"
export VECTOR_PREFIX="private_docs_v1"

# 复用常驻 embedding 服务，避免每条命令重新加载 embedding 模型。
export RAG_EMBEDDING_SERVICE_URL="http://127.0.0.1:9090"

# 默认省内存：不开 rerank。
unset RAG_RERANK_SERVICE_URL
```

### 入库

入库会做：解析文档 -> 切分 section / asset -> 生成摘要 -> embedding -> 写 Milvus。入库阶段不需要 rerank。

单个文档：

```bash
unset RAG_RERANK_SERVICE_URL

uv run python scripts/ingest_private_documents.py \
  --input "$INPUT_PATH" \
  --storage-root "$STORAGE_ROOT" \
  --batch-size 1 \
  --embedding-batch-size 1 \
  --strict-summary-generation \
  --vector-backend milvus \
  --vector-dsn "$VECTOR_DSN" \
  --vector-collection-prefix "$VECTOR_PREFIX" \
  --output "$STORAGE_ROOT/ingest_result.json"
```

批量目录：

```bash
unset RAG_RERANK_SERVICE_URL

uv run python scripts/ingest_private_documents.py \
  --input "$INPUT_DIR" \
  --storage-root "$STORAGE_ROOT" \
  --batch-size 1 \
  --embedding-batch-size 1 \
  --strict-summary-generation \
  --vector-backend milvus \
  --vector-dsn "$VECTOR_DSN" \
  --vector-collection-prefix "$VECTOR_PREFIX" \
  --output "$STORAGE_ROOT/ingest_result.json"
```

入库结果检查：

```bash
cat "$STORAGE_ROOT/ingest_result.json"

uv run python - <<'PY'
import os
from pymilvus import connections, utility

prefix = os.environ["VECTOR_PREFIX"]
connections.connect(alias="check", uri=os.environ["VECTOR_DSN"])
try:
    print([name for name in utility.list_collections(using="check") if name.startswith(prefix)])
finally:
    connections.disconnect("check")
PY
```

### RAG 查询

普通制度/流程问答用 `auto`：

```bash
unset RAG_RERANK_SERVICE_URL

uv run rag query \
  --query "单笔国内差旅报销金额超过 12000 元需要谁审批？请给出处" \
  --storage-root "$STORAGE_ROOT" \
  --vector-backend milvus \
  --vector-dsn "$VECTOR_DSN" \
  --vector-collection-prefix "$VECTOR_PREFIX" \
  --reranker-model none \
  --retrieval-profile auto
```

Excel / 表格 / PPT 表格 / 图片 OCR 这类资产问题优先用 `asset`，并建议加 `--json` 看证据和计算结果：

```bash
unset RAG_RERANK_SERVICE_URL

uv run rag query \
  --query "日提货总量是多少？请给出处" \
  --storage-root "$STORAGE_ROOT" \
  --vector-backend milvus \
  --vector-dsn "$VECTOR_DSN" \
  --vector-collection-prefix "$VECTOR_PREFIX" \
  --reranker-model none \
  --retrieval-profile asset \
  --json
```

JSON 重点字段：

- `answer.answer_text`
- `answer.answer_sections[].evidence_ids`
- `answer.citations`
- `context.evidence`
- `retrieval_diagnostics.operator_plan`
- `retrieval_diagnostics.rerank_skipped`
- `generation_attempts`

### 批量检索评测

适合已有 golden JSONL 时看 doc / section 命中、MRR 和 misses。不开 rerank：

```bash
unset RAG_RERANK_SERVICE_URL

uv run python scripts/evaluate_private_retrieval.py \
  --golden-path data/eval_private/golden_eval_dataset.jsonl \
  --storage-root "$STORAGE_ROOT" \
  --retrieval-profile auto \
  --top-k 10 \
  --retrieval-pool-k 20 \
  --neighbor-radius 1 \
  --no-rerank \
  --vector-backend milvus \
  --vector-dsn "$VECTOR_DSN" \
  --vector-collection-prefix "$VECTOR_PREFIX" \
  --output "$STORAGE_ROOT/retrieval_eval_no_rerank.json" \
  --misses-output "$STORAGE_ROOT/retrieval_misses_no_rerank.jsonl"
```

开启 rerank 时，先启动 rerank 服务，再改两处：设置 `RAG_RERANK_SERVICE_URL`，把 `--no-rerank` 换成 `--rerank`。

### Agent 测试

普通制度问答：

```bash
unset RAG_RERANK_SERVICE_URL

uv run rag agent run \
  "单笔国内差旅报销金额超过 12000 元需要谁审批？请给出处" \
  --agent research \
  --storage-root "$STORAGE_ROOT" \
  --vector-backend milvus \
  --vector-dsn "$VECTOR_DSN" \
  --vector-collection-prefix "$VECTOR_PREFIX" \
  --reranker-model none \
  --verbose
```

表格资产问题：

```bash
uv run rag agent run \
  "日提货总量是多少？请检查相关表格并给出处" \
  --agent research \
  --storage-root "$STORAGE_ROOT" \
  --vector-backend milvus \
  --vector-dsn "$VECTOR_DSN" \
  --vector-collection-prefix "$VECTOR_PREFIX" \
  --reranker-model none \
  --verbose
```

期望工具链：

```text
rag_search_answer 定位候选资产
  -> asset_list / asset_inspect 查看 sheet、列、样例、能力
  -> asset_analyze 执行 dataframe_sql
  -> 最终回答带 asset_id / citation
```

交互式：

```bash
uv run rag agent chat \
  --agent research \
  --storage-root "$STORAGE_ROOT" \
  --vector-backend milvus \
  --vector-dsn "$VECTOR_DSN" \
  --vector-collection-prefix "$VECTOR_PREFIX" \
  --reranker-model none
```

### 常用开关

| 需求 | 做法 |
| --- | --- |
| 关闭 rerank 省内存 | `unset RAG_RERANK_SERVICE_URL`，查询命令加 `--reranker-model none` |
| 开启 HTTP rerank | 启动 `rag_rerank_9092`，`export RAG_RERANK_SERVICE_URL=http://127.0.0.1:9092`，命令里不要传 `--reranker-model` |
| 看 evidence / diagnostics | `rag query` 加 `--json` |
| 普通制度问答 | `--retrieval-profile auto` |
| Excel/PPT 表格/图片 OCR 资产问题 | `--retrieval-profile asset` |
| 一次性指定模型 | 先 `unset RAG_EMBEDDING_SERVICE_URL`，再传 `--model qwen3_8b_mlx_4bit --embedding-model qwen3_embedding_4b_4bit_dwq` |
| 恢复常驻 embedding | `export RAG_EMBEDDING_SERVICE_URL=http://127.0.0.1:9090` |

### 快速 smoke 测试

```bash
uv run rag ingest \
  --storage-root data/smoke_milvus \
  --vector-backend milvus \
  --vector-dsn "$VECTOR_DSN" \
  --vector-collection-prefix smoke_milvus_v1 \
  --source-type plain_text \
  --location memory://smoke/support-sla \
  --title "示例客服 SLA Smoke" \
  --owner smoke \
  --content "示例客服 SLA：P1 工单首次响应目标为 30 分钟，解决目标为 4 小时。"

uv run rag query \
  --query "P1 工单首次响应目标是多少？" \
  --storage-root data/smoke_milvus \
  --vector-backend milvus \
  --vector-dsn "$VECTOR_DSN" \
  --vector-collection-prefix smoke_milvus_v1 \
  --reranker-model none \
  --retrieval-profile auto
```

说明：CLI 默认 metadata 仍是本地 metadata repo，适合快速验证。正式端到端可以使用下面的 `Postgres + parquet object + Milvus` runtime 配置。

## 真实 Postgres + Milvus 端到端

用正式链路时，显式构造 `StorageConfig`：

```python
from pathlib import Path

from rag import AssemblyRequest, CapabilityRequirements, RAGRuntime, StorageComponentConfig, StorageConfig
from rag.ingest.pipeline import IngestRequest
from rag.models.assembly_adapter import to_assembly_overrides
from rag.models.runtime import resolve_runtime_config
from rag.retrieval.models import QueryOptions
from rag.schema.core import SourceType
from rag.utils.text import load_env_file

load_env_file(".env")

run_id = "manual_run_v1"
root = Path("data/manual_pq_milvus") / run_id
schema = f"rag_{run_id}"
collection_prefix = f"rag_{run_id}"

cfg = resolve_runtime_config()
storage = StorageConfig(
    backend="postgres",
    root=root,
    metadata=StorageComponentConfig(
        backend="postgres",
        dsn="postgresql://user:password@127.0.0.1:5432/postgres",
        namespace=schema,
    ),
    vectors=StorageComponentConfig(
        backend="milvus",
        dsn="http://127.0.0.1:19530",
        collection=collection_prefix,
    ),
    cache=StorageComponentConfig(
        backend="redis",
        dsn="redis://127.0.0.1:6379/0",
        namespace=collection_prefix,
    ),
    object_store=StorageComponentConfig(backend="local"),
)

request = AssemblyRequest(
    requirements=CapabilityRequirements(
        require_chat=True,
        require_rerank=True,
        allow_degraded=False,
    ),
    overrides=to_assembly_overrides(cfg),
)

with RAGRuntime.from_request(storage=storage, request=request) as runtime:
    runtime.insert(
        IngestRequest(
            location="memory://demo/support-sla",
            source_type=SourceType.PLAIN_TEXT,
            owner="demo",
            title="示例客服 SLA",
            content_text="示例客服 SLA：P1 工单首次响应目标为 30 分钟，解决目标为 4 小时。",
        )
    )

    runtime.insert(
        IngestRequest(
            location="/absolute/path/to/sample_sales.xlsx",
            source_type=SourceType.XLSX,
            owner="demo",
            title="示例销售明细",
            file_path=Path("/absolute/path/to/sample_sales.xlsx"),
        )
    )

    result = runtime.query_public(
        "请计算示例销售明细中华北区域 2026-05 的销售额合计是多少？",
        options=QueryOptions(retrieval_profile="asset", top_k=6, retrieval_pool_k=12),
    )
    print(result.answer.answer_text)
```

检查真实后端：

```bash
uv run python - <<'PY'
from pymilvus import connections, utility

prefix = "rag_manual_run_v1"
connections.connect(alias="check", uri="http://127.0.0.1:19530")
try:
    print([name for name in utility.list_collections(using="check") if name.startswith(prefix)])
finally:
    connections.disconnect("check")
PY
```

## 测试命令

完整测试：

```bash
uv run pytest -q
```

模型配置、CLI、registry：

```bash
uv run pytest \
  tests/core/test_cli_runtime_model_loading.py \
  tests/ui/test_cli.py \
  tests/agent/test_llm_registry.py
```

表格计算、grounding、Postgres metadata：

```bash
uv run pytest \
  tests/core/test_table_compute_integration.py \
  tests/service/test_grounding_service.py \
  tests/repo/test_postgres_metadata_repo.py
```

复杂 RAG / Agent 回归：

```bash
uv run pytest \
  tests/agent/test_complex_agent_rag_loop.py \
  tests/service/test_complex_rag_retrieval.py
```

最近一次完整结果：

```text
565 passed, 1 skipped, 2 warnings in 11.51s
```

## 目录地图

下面按文件解释主要代码。这里列的是源码里应该维护的文件，不包含 `__pycache__`、本地 `data/` 产物和一次性诊断输出。

```text
./
├── README.md                          # 项目说明、架构、实验结果、运行命令
├── pyproject.toml                     # Python 项目元数据、依赖、pytest 配置
├── uv.lock                            # uv 锁文件
├── .env.example                       # 环境变量示例
├── .gitignore                         # Git 忽略规则
├── .importlinter                      # import-linter 架构边界规则
├── generate_eval_dataset.py           # 从 SectionRecord 生成 golden eval 数据
└── 100万级企业知识Agent系统_最终整合版.md # 早期整体设计文档
```

```text
rag/
├── __init__.py                        # 包级公开导出
├── benchmarks.py                      # benchmark runtime、下载、入库和评测 helper
├── cli.py                             # 主 CLI：ingest / query / benchmark / service 管理入口
├── embedding_service.py               # 本地 embedding HTTP 服务入口
├── query_pipeline.py                  # 查询端 L3-L6 编排、表格 compute_request 循环
├── rerank_service.py                  # 本地 rerank HTTP 服务入口
├── runtime.py                         # AppRuntime 装配 storage、ingest、retrieval、synthesis
├── agent/
│   ├── __init__.py                    # Agent 层公开导出
│   ├── cli.py                         # Agent CLI 与 resume/approval 参数处理
│   ├── service.py                     # AgentRunRequest / AgentRunResult / AgentService
│   ├── state.py                       # AgentState TypedDict、ToolCallPlan、reducers
│   ├── builtin/
│   │   ├── __init__.py                # 内置 Agent 注册表
│   │   ├── compare.py                 # CompareAgent 定义
│   │   ├── factcheck.py               # FactCheckAgent 定义
│   │   ├── orchestrator.py            # OrchestratorAgent 定义
│   │   ├── research.py                # ResearchAgent 定义与 service factory
│   │   └── synthesize.py              # SynthesizeAgent 定义
│   ├── core/
│   │   ├── __init__.py                # core 层公开导出
│   │   ├── agent_as_tool.py           # 子 Agent 封装为工具的 spec、adapter、runner
│   │   ├── agent_service_factory.py   # 按 AgentDefinition 创建 AgentService
│   │   ├── approval_policy.py         # 工具权限、审批和暂停判断
│   │   ├── checkpointing.py           # LangGraph checkpointer helper
│   │   ├── compiler.py                # AgentDefinition 到 LangGraph runnable 的编译器
│   │   ├── context.py                 # AgentRunConfig、BudgetLedger、RuntimeRegistry
│   │   ├── definition.py              # AgentDefinition、ModelSelectionPolicy、ToolPolicy
│   │   ├── human_input.py             # 人审请求/响应和 pending tool 摘要
│   │   ├── llm_config.py              # Agent 模型配置 schema
│   │   ├── delegation.py              # agent_* 子 Agent 工具委派契约
│   │   ├── llm_prompts.py             # retrieval hint / tool decision prompt 模板
│   │   ├── llm_providers.py           # 节点 LLM provider 协议与实现
│   │   ├── llm_registry.py            # models.yaml 到节点模型的解析和缓存
│   │   ├── registry.py                # AgentDefinition 注册表
│   │   └── subagent_runner.py         # 内置 delegated-agent / synthesis runner
│   ├── graphs/
│   │   ├── __init__.py                # graph 层公开导出
│   │   ├── base.py                    # LangGraph 主图 AgentLoop 装配
│   │   └── nodes/
│   │       ├── __init__.py            # graph node 公开导出
│   │       ├── execute.py             # 工具执行、预算记账、审批阻断、错误结构化
│   │       ├── goal_runtime.py        # 初始化、检查和 observation 归并适配
│   │       ├── llm_decide.py          # 模型选择下一次普通工具调用
│   │       ├── observe.py             # 工具结果观察和状态整理
│   │       ├── pause.py               # interrupt/resume 暂停节点
│   │       ├── retrieval_hint.py      # retrieval hint 元数据生成
│   │       └── synthesize.py          # 最终回答合成与 groundedness 标记
│   ├── memory/
│   │   ├── __init__.py                # memory 层公开导出
│   │   ├── compactor.py               # working memory 压缩和事实抽取
│   │   ├── injector.py                # bounded context 注入
│   │   └── models.py                  # WorkingSummary、ExtractedFact、budget snapshot
│   └── tools/
│       ├── __init__.py                # tools 层公开导出
│       ├── builtin_registry.py        # 内置 RAG/LLM/Agent 工具注册
│       ├── rag_answer_tools.py        # rag_search_answer 普通工具契约
│       ├── llm_tools.py               # llm_generate / llm_summarize / llm_compare 契约
│       ├── rag_tool_runner.py         # RAG runtime/retrieval service 到工具输出的适配
│       ├── rag_tools.py               # vector_search / keyword_search / grounding / rerank 等契约
│       ├── registry.py                # ToolRegistry、runner 执行和输入输出校验
│       └── spec.py                    # ToolSpec、ToolPermissions、ToolResult、ToolError
├── assembly/
│   ├── __init__.py                    # assembly 层公开导出
│   ├── bindings.py                    # provider binding 和 runtime 依赖绑定
│   ├── models.py                      # 装配层数据模型
│   ├── service.py                     # 统一装配服务
│   ├── support.py                     # provider 支持能力检测
│   └── tokenizer.py                   # tokenizer 选择和 token 计数
├── ingest/
│   ├── __init__.py                    # ingest 层公开导出
│   ├── asset_anchors.py               # [ASSET_ANCHOR:...] 生成和解析
│   ├── header_detector.py             # 表头/标题启发式检测
│   ├── pipeline.py                    # L1/L2 入库主链路
│   ├── retrievalsummarizer.py         # doc / section / asset 三类摘要生成
│   ├── section_refiner.py             # token 窗口切分和 section 规范化
│   ├── table_executor.py              # DuckDB 表格计算 sandbox
│   ├── table_sampler.py               # 表格 schema、sample、profile、policy
│   └── parsers/
│       ├── __init__.py                # parser 层公开导出
│       ├── dispatcher.py              # 按文件类型分发 parser
│       ├── docling_parser_repo.py     # PDF / Word / Markdown Docling parser
│       ├── excel_parser_repo.py       # Excel parser 和表格 asset 抽取
│       ├── image_parser_repo.py       # 图片 parser 和 OCR asset 抽取
│       ├── ocr_repos.py               # OCR provider 仓储接口/实现
│       ├── ppt_parser_repo.py         # PPTX parser 和 slide/table/note 抽取
│       ├── util.py                    # parser 公共工具
│       ├── web_fetch_repo.py          # Web 抓取接口
│       └── web_parser_repo.py         # Web 内容解析
├── models/
│   ├── __init__.py                    # model 层公开导出
│   ├── assembly_adapter.py            # models.yaml 到 provider assembly 的适配
│   ├── catalog.py                     # 模型目录和 alias 管理
│   ├── config.py                      # 模型配置 schema
│   ├── guard.py                       # 模型配置校验和安全检查
│   └── runtime.py                     # 模型 runtime 解析
├── providers/
│   ├── __init__.py                    # provider 层公开导出
│   ├── citation_formatter.py          # citation 文本格式化
│   ├── embedding_http.py              # embedding HTTP client
│   ├── fallback.py                    # 显式 fallback provider 组合
│   ├── generation.py                  # Chat/generation provider 和 synthesis prompt
│   ├── rerank_http.py                 # rerank HTTP client
│   ├── telemetry.py                   # provider 调用 telemetry
│   ├── huggingface/
│   │   ├── embedder.py                # Hugging Face embedding provider
│   │   ├── generator.py               # Hugging Face generation provider
│   │   ├── hf_utils.py                # Hugging Face 公共工具
│   │   └── rerank.py                  # Hugging Face rerank provider
│   ├── mlx/
│   │   ├── embedder.py                # MLX embedding provider
│   │   └── generator.py               # MLX generation provider
│   └── ollama/
│       ├── embedder.py                # Ollama embedding provider
│       └── generator.py               # Ollama generation provider
├── retrieval/
│   ├── __init__.py                    # retrieval 层公开导出
│   ├── authorization_service.py       # 访问策略和权限过滤
│   ├── context.py                     # retrieval 上下文对象
│   ├── evidence.py                    # evidence helper
│   ├── fusion.py                      # 多路召回融合和排序
│   ├── graph.py                       # retrieval graph helper
│   ├── grounding_service.py           # L5 原文回读、anchor replacement、表格计算触发
│   ├── l3_l4_engine.py                # L3/L4 engine 协调
│   ├── models.py                      # retrieval 内部模型
│   ├── orchestrator.py                # retrieval 编排器
│   ├── planning_graph.py              # L3 planning graph
│   ├── rerank_service.py              # 候选清洗和 rerank service
│   ├── retrieval_adapter.py           # summary index 检索适配器
│   ├── runtime_coordinator.py         # retrieval runtime 装配协调
│   └── synthesis_service.py           # L6 evidence-only synthesis
├── schema/
│   ├── __init__.py                    # schema 层公开导出
│   ├── core.py                        # Document / SectionRecord / AssetRecord / SummaryRecord
│   ├── graph.py                       # 图谱相关 schema
│   ├── model_protocols.py             # provider protocol 类型
│   ├── query.py                       # GroundingTarget / EvidenceItem / Answer / artifact contract
│   └── runtime.py                     # Runtime contract、diagnostics、vector result
├── storage/
│   ├── __init__.py                    # storage 层公开导出
│   ├── cache.py                       # cache repository 协议
│   ├── data_contract_service.py       # 数据契约和 schema 校验服务
│   ├── index_sync_service.py          # index 同步服务
│   ├── index_sync_worker.py           # index 同步 worker
│   ├── object_store.py                # object store 协议
│   ├── storage_lifecycle_service.py   # 删除、重建、生命周期服务
│   ├── storage_lifecycle_worker.py    # 生命周期 worker
│   ├── graph_backends/
│   │   ├── __init__.py                # graph backend 公开导出
│   │   └── null_graph_repo.py         # 空 graph repo 实现
│   ├── repositories/
│   │   ├── __init__.py                # repository 公开导出
│   │   ├── file_object_store.py       # 本地文件 object store
│   │   ├── postgres_metadata_repo.py  # PostgreSQL metadata repo
│   │   ├── redis_cache_repo.py        # Redis cache repo
│   │   ├── s3_object_store.py         # S3 object store
│   │   └── sqlite_metadata_repo.py    # SQLite metadata repo
│   └── search_backends/
│       ├── __init__.py                # search backend 公开导出
│       ├── in_memory_vector_repo.py   # 测试用内存向量 repo
│       ├── milvus_vector_repo.py      # Milvus vector repo
│       ├── sqlite_vector_repo.py      # SQLite vector repo
│       └── web_search_repo.py         # Web search repo
└── utils/
    ├── __init__.py                    # utils 公开导出
    ├── guard.py                       # 通用安全检查
    ├── telemetry.py                   # 通用 telemetry helper
    └── text.py                        # 文本处理工具
```

```text
configs/
└── models.yaml                        # 默认 chat / embedding / rerank 模型配置
```

```text
scripts/
├── check_anti_patterns.py             # 代码反模式检查
├── demo_llm_agent.py                  # LLM Agent 手动验证入口
├── download_public_benchmark.py       # 下载公开 benchmark
├── evaluate_private_retrieval.py      # 私有 golden set 检索评测
├── export_private_sections.py         # 从 index 导出 SectionRecord JSONL
├── ingest_private_documents.py        # 私有文件夹入库
├── ingest_public_benchmark.py         # 公开 benchmark 入库
├── prepare_public_benchmark.py        # 准备公开 benchmark
└── profile_benchmark_ingest.py        # 入库速度 profiling
```

```text
tests/
├── __init__.py                        # tests 包标记
├── support.py                         # 测试 helper
├── verify_summary_mlx.py              # MLX 摘要验证脚本
├── agent/
│   ├── __init__.py                    # Agent 测试包标记
│   ├── conftest.py                    # Agent 测试 fixtures
│   ├── test_agent_as_tool_runner.py   # Agent-as-tool runner
│   ├── test_agent_cli_resume.py       # CLI resume 参数
│   ├── test_agent_graph_compiler.py   # graph compiler
│   ├── test_agent_service.py          # AgentService run/result
│   ├── test_agent_service_resume.py   # AgentService resume
│   ├── test_approval_policy.py        # 审批策略
│   ├── test_builtin_agents.py         # 内置 AgentDefinition
│   ├── test_builtin_research_agent.py # ResearchAgent service
│   ├── test_builtin_subagent_runner.py # 内置子 Agent runner
│   ├── test_builtin_tool_registry.py  # 内置工具注册
│   ├── test_cli_wiring.py             # Agent CLI wiring
│   ├── test_complex_agent_rag_loop.py # 复杂 Agent + RAG 循环
│   ├── test_context_injector.py       # context injection
│   ├── test_contract_config.py        # Agent config contract
│   ├── test_contract_state.py         # AgentState contract
│   ├── test_contract_tool.py          # ToolSpec contract
│   ├── test_llm_decide_context_integration.py # llm_decide + context
│   ├── test_execute_node_runtime.py   # execute node runtime
│   ├── test_rag_answer_tool.py        # 模型选择 rag_search_answer 工具
│   ├── test_graph_base.py             # base graph routing
│   ├── test_human_input.py            # human input schema
│   ├── test_interrupt_resume.py       # interrupt / resume
│   ├── test_llm_config.py             # LLM config schema
│   ├── test_llm_providers.py          # LLM node providers
│   ├── test_llm_registry.py           # model registry
│   ├── test_llm_tool_specs.py         # LLM tool specs
│   ├── test_public_exports.py         # public exports
│   ├── test_rag_tool_runner.py        # RAG tool runner
│   ├── test_rag_tool_specs.py         # RAG tool specs
│   ├── test_retrieval_signals_loop.py # retrieval signals loop
│   ├── test_synthesize_agent_runner.py # synthesize Agent runner
│   ├── test_tool_registry.py          # ToolRegistry validation
│   └── test_working_memory_compactor.py # working memory compaction
├── core/
│   ├── test_benchmark_contract.py     # benchmark contract
│   ├── test_capability_assembly.py    # capability assembly
│   ├── test_citation_formatter.py     # citation formatting
│   ├── test_cli_runtime_model_loading.py # CLI model loading
│   ├── test_data_contract_service.py  # data contract service
│   ├── test_delete_rebuild.py         # delete/rebuild lifecycle
│   ├── test_evaluate_private_retrieval_script.py # private eval script
│   ├── test_excel_parser_repo.py      # Excel parser
│   ├── test_generate_eval_dataset.py  # eval dataset generation
│   ├── test_guard.py                  # guard checks
│   ├── test_header_detector.py        # header detection
│   ├── test_index_sync_service.py     # index sync service
│   ├── test_index_sync_worker.py      # index sync worker
│   ├── test_ingest_asset_anchors.py   # asset anchors
│   ├── test_ingest_public_benchmark_script.py # public ingest script
│   ├── test_milvus_vector_repo.py     # Milvus vector repo
│   ├── test_model_runtime.py          # model runtime
│   ├── test_postgres_milvus_v1_contract.py # Postgres + Milvus contract
│   ├── test_public_benchmark.py       # public benchmark
│   ├── test_retrieval_summarizer.py   # retrieval summarizer
│   ├── test_runtime_entrypoint.py     # runtime entrypoint
│   ├── test_runtime_query_pipeline_module.py # query pipeline module
│   ├── test_schema_deduplication.py   # schema deduplication
│   ├── test_section_refiner.py        # section refiner
│   ├── test_storage_lifecycle_worker.py # lifecycle worker
│   ├── test_table_compute_integration.py # table compute integration
│   └── test_table_executor.py         # DuckDB table executor
├── provider/
│   ├── __init__.py                    # provider 测试包标记
│   ├── test_embedding_http.py         # embedding HTTP client
│   └── test_rerank_http.py            # rerank HTTP client
├── repo/
│   ├── test_file_object_store.py      # file object store
│   ├── test_huggingface_embedder.py   # Hugging Face embedder
│   └── test_postgres_metadata_repo.py # Postgres metadata repo
├── service/
│   ├── test_answer_generation_contract.py # answer generation contract
│   ├── test_authorization_service.py  # authorization service
│   ├── test_complex_rag_retrieval.py  # complex RAG retrieval
│   ├── test_grounding_service.py      # grounding service
│   ├── test_industrial_rerank_service.py # rerank service
│   ├── test_planning_graph.py         # planning graph
│   ├── test_retrieval_adapter.py      # retrieval adapter
│   ├── test_retrieval_service_config.py # retrieval service config
│   ├── test_summary_hybrid_retriever.py # summary hybrid retrieval
│   ├── test_synthesis_contract.py     # synthesis contract
│   └── test_telemetry_service.py      # telemetry service
└── ui/
    └── test_cli.py                    # 主 CLI 测试
```

## 运行注意事项

- 入库和查询必须使用同一个 embedding space；切换 embedding 模型后必须重建 Milvus collection。
- 每次真实实验建议使用新的 `STORAGE_ROOT` 和 Milvus collection prefix，避免不同 embedding 维度或旧 schema 污染结果。
- `9091` 被 Milvus 占用，rerank 服务使用 `9092`。
- 对表格真实值问题，不要信任 `sample_rows`；正确路径是 `<compute_request>` -> DuckDB -> `TABLE_COMPUTE_RESULT`。
- OpenAI-compatible chat provider 当前没有结构化生成实现，生成链路会先记录一次 structured failure，再走 text fallback；这是可见降级，不是静默 fallback。
- 批量入库脚本支持 `--summary-provider none`，公开 benchmark 可跳过 LLM 摘要生成，直接用 passage 原文入 summary index。

常见问题和处理顺序：

| 现象 | 主要原因 | 处理 |
| --- | --- | --- |
| `Component backend requires a DSN/URI` | `--vector-backend milvus` 但没有传 DSN，或 `VECTOR_DSN` 为空 | `echo "$VECTOR_DSN"`，命令中补 `--vector-dsn "$VECTOR_DSN"` |
| Milvus `vector dimension mismatch` | 查询 embedding 和入库 embedding 不是同一模型/维度 | 换新的 `VECTOR_PREFIX` 和 `STORAGE_ROOT` 重新入库，不要混用旧 collection |
| `Embedding service health check failed: Connection refused` | embedding 服务没启动或端口不对 | `curl http://127.0.0.1:9090/health`，失败就重启 `rag_embedding_9090` |
| `Rerank service health check failed` | 查询要求 rerank，但 rerank 服务没启动；或入库阶段误带 rerank | 入库前 `unset RAG_RERANK_SERVICE_URL`；关闭 rerank 查询时传 `--reranker-model none` |
| `document summary generation returned empty output` | Qwen3 thinking 未关闭，或本地生成服务异常 | 按 README 用 `enable_thinking=false` 重启 Qwen，再重跑入库 |
| 入库 10 分钟进度条不动 | 模型首次加载、Excel 解析大表、embedding batch 太大、内存压力高 | 先 `curl` 检查服务；用 `--batch-size 1 --embedding-batch-size 1`；Excel 用 `scripts/diagnose_ingest_timing.py` 定位 |
| DOCX DrawingML / VML 图片警告 | 缺少 LibreOffice 转换器或文档内图片引用缺失 | 正文通常仍可入库；如果必须抽图，安装 LibreOffice 并设置 `DOCLING_LIBREOFFICE_CMD` |
| RAG 表格答案没读到明显数字 | 没走 asset profile、没触发 compute、或问题口径不清 | 用 `--retrieval-profile asset --json` 看 `context.evidence`、`TABLE_COMPUTE_ONLY`、`TABLE_COMPUTE_RESULT` |
| Agent 输出 `no_pending_tools` / 没有工具结果 | 生成模型没产生工具调用或路由停止 | 先用 `rag query --json` 验证索引；检查 Qwen 服务；再用 `rag agent run --verbose` 看工具执行 |
| Agent 表格问题乱选 sheet/产品/口径 | 用户问题本身有歧义，或 Agent 未做资产 inspect | 正确行为是列候选或请求澄清；不要用硬编码业务关键词修某张表 |

DOCX 图形转换可选配置：

```bash
export DOCLING_LIBREOFFICE_CMD="/Applications/LibreOffice.app/Contents/MacOS/soffice"
```

Excel 入库耗时诊断：

```bash
uv run python scripts/diagnose_ingest_timing.py "$INPUT_PATH"
```

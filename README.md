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

- 2026-05-20：README 补齐本地 Qwen / embedding / rerank 服务管理、私有文档入库、RAG 查询、Agent 查询、JSON diagnostics 和省内存运行手册。
- 2026-05-17：历史默认模型切到 `deepseek-v4-flash`、`mlx-community/Qwen3-Embedding-4B-4bit-DWQ`、`BAAI/bge-reranker-v2-m3`。
- 2026-05-17：README 保留 HTML badge、导航和 Mermaid 版式，同时保留历史 baseline 用来对比后续改进。
- 2026-05-16：完成真实 `PostgreSQL + parquet object store + Milvus + Redis` 端到端验证，表格问题通过 DuckDB 返回 `375`。
- 2026-05-16：README 恢复历史 baseline，并补齐 Agent 设计说明、文件级目录和当前运行命令。

## 能力一览

| 能力 | 当前状态 | 关键实现 |
| --- | --- | --- |
| 多格式入库 | 已支持 PDF、Word、Markdown、Excel、PPT、图片、纯文本 | `rag/ingest/pipeline.py`、`rag/ingest/parsers/*` |
| 多粒度索引 | doc / section / asset 三类 summary index | Milvus collections + summary records |
| 混合检索 | 支持 `fast / auto / deep / asset` profile | L3 planning + L4 retrieval + rerank |
| Grounding | 原文回读、anchor replacement、neighbor expansion | `rag/retrieval/grounding_service.py` |
| 表格计算 | Excel asset 转 parquet，DuckDB 受限 `SELECT` | `table_sampler.py`、`table_executor.py` |
| Agent 编排 | Tool-Centric + LangGraph + TaskDAG + approval pause | `rag/agent/*` |
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
        a1["定义"] --> a2["工具"] --> a3["预算"] --> a4["图执行"] --> a5["子任务"] --> a6["结果"]
    end

    classDef rag fill:#eef6ff,stroke:#7aa7d9,color:#0f172a;
    classDef agent fill:#f4f0ff,stroke:#a78bfa,color:#0f172a;
    class r1,r2,r3,r4,r5,r6,r7,r8 rag;
    class a1,a2,a3,a4,a5,a6 agent;
```

RAG 负责把原始资料变成可引用证据，Agent 运行在 RAG 能力之上，用工具契约、预算和 LangGraph 状态流转处理复杂任务。

## 架构总览

系统由 RAG 流程和 Agent 编排层组成。

```text
L1 Storage
  原始对象、Document、SectionRecord、AssetRecord、locator、权限、版本、处理状态

L2 Indexing
  DocSummary / SectionSummary / AssetSummary -> Embedding -> Milvus

L3 Planning
  complexity gate、semantic route、version gate、predicate push-down

L4 Retrieval
  多粒度 summary 检索、候选清洗、RRF/融合、rerank、召回诊断

L5 Grounding
  原文 range read、局部动态切片、neighbor expansion、asset anchor replacement、预算熔断

L6 Synthesis
  基于 EvidenceItem 生成最终回答、引用、权限/合规复核

Agent Layer
  ToolSpec、AgentDefinition、TaskDAG、LangGraph、BudgetLedger、working memory
```

### L1：事实层

L1 保存事实数据和可追溯定位信息：

- `Document`：文档版本、权限、状态、来源。
- `SectionRecord`：正文窗口，带 `raw_locator`、byte range、token 窗口元数据。
- `AssetRecord`：表格、图片、OCR 区域、PPT 表格等非正文资产。
- Object Store：保存原始文件、visible text、表格对象、schema/sample 和 DuckDB 可读存储指针。

### L2：索引层

L2 保存检索入口。Milvus 中按粒度拆成三类 summary index：

- `doc_summary`：文档级主题召回。
- `section_summary`：正文 section 召回。
- `asset_summary`：表格、图片、OCR、PPT 资产召回。

索引层保存 summary、向量、标量过滤字段和主键映射。原文、表格和权限信息仍由事实层提供。

### L3/L4：规划与检索

L3 判断查询应该如何检索，L4 负责候选召回和排序。系统支持这些 `retrieval_profile`：

- `fast`
- `auto`
- `deep`
- `asset`

规划层处理复杂度、语义路由、版本过滤和谓词下推；检索层对 doc / section / asset summary 做多路召回、候选清洗、融合和 rerank。

### L5：精读与证据层

L5 将 summary 命中的候选重新映射回原始正文或资产对象：

- 命中正文 section 后，通过 `visible_text_key + byte_range` 回读原文。
- 命中含表格锚点的 section 后，通过 `[ASSET_ANCHOR:...]` 找到绑定资产。
- 表格资产通过 DuckDB Text-to-SQL Sandbox 执行受限查询。
- grounding 阶段受 token、目标数、并发和超时预算控制。

### L6：回答合成层

L6 只基于 `EvidenceItem` 合成回答。回答保留 `doc_id / section_id / asset_id`、citation anchor、检索分数、rerank 分数和 evidence metadata，便于追溯和评测。

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

Agent 层采用 Tool-Centric + LangGraph 设计，把 Agent 定义、工具契约、运行配置、状态合并和图执行拆成清晰边界。

### 设计说明

Agent 层不是单独绕开 RAG 的聊天入口，而是把 L3-L6 能力封装为可验证工具，并用 LangGraph 管理复杂任务的状态流转。设计目标是让每一次 Agent run 都能回答这些问题：用了哪个 AgentDefinition、允许哪些工具、消耗多少预算、哪些工具需要审批、召回了哪些 evidence、哪些 citation 被带入最终回答、为什么停止。

核心边界如下：

- `AgentDefinition` 只描述 Agent 类型、系统提示、工具 allowlist、模型选择、最大迭代、最大嵌套深度和工具策略。
- `ToolSpec` 只描述工具契约，包括 Pydantic 输入输出、权限、timeout、retry、预算成本和是否需要确认。
- `AgentRunConfig` 保存可序列化运行配置；`RuntimeRegistry` 保存不可序列化 runtime handles，例如 `BudgetLedger` 和 cancellation event。
- `AgentState` 使用 TypedDict，配合 LangGraph reducer 合并 messages、evidence、citations、tool results、subtask results 和 terminal sets。
- `TaskDAG` 是复杂任务拆解结果，只表达子任务、依赖、优先级和预算估算，不直接执行业务逻辑。
- `AgentService` 是外部调用边界，负责构造初始状态、编译图、注入 request-scoped tool registry，并返回结构化 `AgentRunResult`。

运行链路如下：

```text
AgentRunRequest
  -> AgentRunConfig
  -> AgentService.initial_state()
  -> AgentGraphCompiler.compile(AgentDefinition)
  -> route
  -> fast_path | plan | execute | synthesize
  -> observe
  -> evaluate
  -> execute tool 或 execute_subagent
  -> pause approval 或 synthesize
  -> AgentRunResult
```

工具执行遵循 fail-closed 策略。工具未注册、runner 缺失、输入非法、输出非法、权限被拒、审批未通过、预算不足、timeout 或 runner 异常都会变成结构化 `ToolResult(status="error")` 或显式失败状态，不做静默 fallback。写库、知识图谱变更、外部副作用等高风险工具通过 `ToolPermissions`、`requires_confirmation` 和 `approval_policy` 进入暂停/恢复链路。

子 Agent 通过 agent-as-tool 接入父图。`build_agent_tool_spec()` 会把子 Agent 包成普通工具契约，`AgentAsToolAdapter` 在每个 request 的 runtime registry 里注入，避免并发 run 互相污染预算、深度和访问策略。父 Agent 派发子任务时会通过 `derive_child_config()` 限制嵌套深度，并继承必要的 access policy 和 source scope。

RAG evidence 是事实优先级最高的上下文。Working memory 只用于压缩当前 run 的历史消息和抽取线索，不能覆盖 grounding 后的 evidence。最终回答必须保留 evidence、citation、retrieval score、rerank score 和 grounding metadata，便于复查和评测。

### 已实现

- 工具契约：`ToolSpec / ToolPermissions / ToolResult / ToolError` 描述工具输入、输出、权限、错误、预算和重试策略。
- 工具注册与执行：`ToolRegistry` 支持注册工具 spec 和 callable runner，并执行 Pydantic 输入/输出校验。
- 运行配置：`AgentRunConfig / RuntimeRegistry / BudgetLedger` 区分可序列化 run config 和不可序列化 runtime handles。
- Agent 定义：`AgentDefinition / ModelPolicy / ToolPolicy` 描述 Agent 类型、系统提示、工具 allowlist、模型偏好和工具策略。
- 状态契约：`AgentState` 使用 TypedDict 和 reducer 合并 messages、evidence、citations、tool results、subtask results。
- 子任务图：`TaskDAG` 支持子任务、依赖边、环检测、ready subtask 调度、terminal/successful 分离。
- Agent 组合：`AgentRegistry / AgentToolSpec / AgentAsToolRunner` 支持注册 Agent，并把子 Agent 封装成可调用工具。
- 图执行骨架：LangGraph base graph 已包含 `route -> plan -> execute -> observe -> evaluate -> synthesize`。
- 并行子任务节点：`execute_subagent` 支持通过 `Send()` 扇出子任务，完成后回到 evaluate 继续调度。
- Working memory Phase A：已实现 working summary、extracted facts 和 bounded context injection。
- 内置 ResearchAgent：已提供 research AgentDefinition、RAG/LLM tool specs、builtin tool registry 和 service factory。

### 已具备的能力

- 可以用统一契约定义 Agent、工具、运行预算、访问策略、工具权限和执行位置偏好。
- 可以运行基础 LangGraph Agent 流程，完成路由、计划、工具执行、观察、评估和最终结果合成。
- 可以执行已注册工具，并在工具未注册、未实现、参数非法、输出非法、预算不足、timeout、runner 异常时返回结构化失败结果。
- 可以把复杂任务表示为 `TaskDAG`，按依赖关系选择 ready subtasks，并阻断依赖失败子任务的下游任务。
- 可以把子 Agent 作为工具调用，支持父 run 向子 run 派发任务、继承必要上下文并回收子任务结果。
- 可以在并行子任务返回后合并 evidence、citations、tool results、subtask results 和状态字段。
- 可以对长上下文做 working memory compaction / injection，控制 tail messages、tool results、evidence 和 memory 的上下文预算。
- 可以保证回答事实以 RAG evidence 为最高优先级，memory 只作为当前 run 的上下文线索。
- 可以通过 `AgentService` 返回结构化 `AgentRunResult`，包含 status、final answer、stop reason、tool results、evidence、citations 和 groundedness flags。
- 可以使用内置 ResearchAgent 作为当前可运行 Agent 模板，后续 Agent 可以复用同一套契约和图执行框架。

### 待推进

- 生产级 LLM plan/evaluate provider、prompt 模板和 golden case 评测。
- LangGraph `interrupt()`、checkpointer、`Command(resume=...)` 的完整暂停/恢复链路。
- Orchestrator、CompareAgent、FactCheckAgent、SynthesizeAgent 等更多内置 AgentDefinition。
- RAG tool runner 与 L3-L6 runtime 的完整绑定，包括 vector search、keyword search、grounding、rerank 和 synthesis。
- Agent CLI/API，让外部调用方可以稳定启动 run、查看状态、恢复任务和读取结构化结果。
- 长期 memory Phase B，包括写入策略、去重、冲突标记和与 evidence 的优先级治理。

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

- Chat：`deepseek-v4-flash`
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

表格规则：

- Excel 入库后表格资产会记录 `row_count / column_count / schema / sample_rows / storage_key`。
- 表格资产会转换为 DuckDB 可读的 `.parquet` 对象。
- 涉及真实数据值、筛选、求和、计数、排序、排名、对比或聚合的问题，必须走 `<compute_request>`。
- DuckDB 执行 `SELECT` 后会把 `TABLE_COMPUTE_RESULT` 和 `Executed SQL` 注入证据，再生成最终回答。
- `sample_rows` 只用于识别 schema，不允许被当成完整表格直接回答。

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

下面是当前推荐的本地私有知识库流程：启动服务、入库、RAG 查询、Agent 查询、可选 rerank。

统一设置变量。入库和查询必须使用同一套 `STORAGE_ROOT / VECTOR_DSN / VECTOR_PREFIX`：

```bash
cd "/Users/leixiaoying/LLM/RAG学习"

export STORAGE_ROOT="data/longpai_agent_milvus_v3"
export VECTOR_DSN="http://127.0.0.1:19530"
export VECTOR_PREFIX="longpai_agent_milvus_v3"
export INPUT_DIR="data/longpai_agent_input_docx"

# 让 CLI 复用常驻 embedding 服务，避免每次命令重新加载 embedding 模型。
export RAG_EMBEDDING_SERVICE_URL="http://127.0.0.1:9090"

# 默认省内存：不使用 rerank。
unset RAG_RERANK_SERVICE_URL
```

检查变量，空字符串最容易导致 Milvus DSN 错误：

```bash
echo "$STORAGE_ROOT"
echo "$VECTOR_DSN"
echo "$VECTOR_PREFIX"
echo "$INPUT_DIR"
echo "$RAG_EMBEDDING_SERVICE_URL"
```

确认 Milvus 能连：

```bash
uv run python - <<'PY'
from pymilvus import connections, utility
connections.connect(alias="check", uri="http://127.0.0.1:19530")
try:
    print("milvus ok")
    print(utility.list_collections(using="check"))
finally:
    connections.disconnect("check")
PY
```

### 入库

入库会做：解析文档 -> 切分 section / asset -> 生成 doc / section / asset 摘要 -> embedding -> 写 Milvus summary index。入库阶段不需要 rerank。

```bash
uv run python scripts/ingest_private_documents.py \
  --input "$INPUT_DIR" \
  --storage-root "$STORAGE_ROOT" \
  --batch-size 1 \
  --embedding-batch-size 2 \
  --strict-summary-generation \
  --vector-backend milvus \
  --vector-dsn "$VECTOR_DSN" \
  --vector-collection-prefix "$VECTOR_PREFIX" \
  --output data/longpai_agent_milvus_v3/ingest_result.json
```

内存不足或进度条很久不动时，先用更保守参数：

```bash
uv run python scripts/ingest_private_documents.py \
  --input "$INPUT_DIR" \
  --storage-root "$STORAGE_ROOT" \
  --batch-size 1 \
  --embedding-batch-size 1 \
  --strict-summary-generation \
  --vector-backend milvus \
  --vector-dsn "$VECTOR_DSN" \
  --vector-collection-prefix "$VECTOR_PREFIX" \
  --output data/longpai_agent_milvus_v3/ingest_result.json
```

入库结果检查：

```bash
cat data/longpai_agent_milvus_v3/ingest_result.json

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

普通文本输出，适合人工看效果：

```bash
uv run rag query \
  --storage-root "$STORAGE_ROOT" \
  --vector-backend milvus \
  --vector-dsn "$VECTOR_DSN" \
  --vector-collection-prefix "$VECTOR_PREFIX" \
  --reranker-model none \
  --retrieval-profile auto \
  --query "单笔国内差旅报销金额超过 12000 元需要谁审批？请给出处"
```

JSON 输出，适合看 evidence、citations、diagnostics、generation attempts：

```bash
uv run rag query \
  --storage-root "$STORAGE_ROOT" \
  --vector-backend milvus \
  --vector-dsn "$VECTOR_DSN" \
  --vector-collection-prefix "$VECTOR_PREFIX" \
  --reranker-model none \
  --retrieval-profile auto \
  --query "单笔国内差旅报销金额超过 12000 元需要谁审批？请给出处" \
  --json
```

常用检查字段：

- `answer.answer_text`：最终自然语言答案。
- `answer.citations`：引用位置。
- `context.evidence`：进入生成模型的证据。
- `retrieval_diagnostics.operator_plan`：检索算子计划。
- `retrieval_diagnostics.rerank_skipped`：不启用 rerank 时应为 `true`。
- `generation_attempts`：structured generation / text fallback 是否成功。

### Agent 查询

ResearchAgent 单次运行：

```bash
uv run rag agent run "单笔国内差旅报销金额超过 12000 元需要谁审批？请给出处" \
  --agent research \
  --storage-root "$STORAGE_ROOT" \
  --vector-backend milvus \
  --vector-dsn "$VECTOR_DSN" \
  --vector-collection-prefix "$VECTOR_PREFIX" \
  --reranker-model none \
  --verbose
```

ResearchAgent 交互式运行：

```bash
uv run rag agent chat \
  --agent research \
  --storage-root "$STORAGE_ROOT" \
  --vector-backend milvus \
  --vector-dsn "$VECTOR_DSN" \
  --vector-collection-prefix "$VECTOR_PREFIX" \
  --reranker-model none
```

Agent 正常输出应包含最终回答；`--verbose` 时会显示工具执行摘要。如果看到：

```text
No answer was generated because no tool results were available.
停止原因: no_pending_tools
```

说明 Agent 没有生成工具调用。优先检查 Qwen 生成服务是否启动、`configs/models.yaml` 默认模型是否可访问、以及 `curl http://127.0.0.1:8080/v1/models` 是否正常。

### 是否开启 rerank

默认省内存命令都使用：

```bash
--reranker-model none
unset RAG_RERANK_SERVICE_URL
```

需要开启 rerank 时：

```bash
export RAG_RERANK_SERVICE_URL="http://127.0.0.1:9092"

uv run rag query \
  --storage-root "$STORAGE_ROOT" \
  --vector-backend milvus \
  --vector-dsn "$VECTOR_DSN" \
  --vector-collection-prefix "$VECTOR_PREFIX" \
  --retrieval-profile auto \
  --query "单笔国内差旅报销金额超过 12000 元需要谁审批？请给出处" \
  --json
```

开启 HTTP rerank 时不要再传 `--reranker-model none`。如果同时设置了 `RAG_RERANK_SERVICE_URL` 又传了 `--reranker-model`，CLI 会报冲突。

### 快速 smoke 测试

文本 smoke 入库：

```bash
uv run rag ingest \
  --storage-root data/smoke_milvus \
  --vector-backend milvus \
  --vector-dsn "$VECTOR_DSN" \
  --vector-collection-prefix smoke_milvus_v1 \
  --source-type plain_text \
  --location memory://smoke/travel-policy \
  --title "差旅制度 Smoke" \
  --owner smoke \
  --content "单笔国内差旅报销金额超过 12000 元时，必须由业务线 VP 审批。"
```

文本 smoke 查询：

```bash
uv run rag query \
  --storage-root data/smoke_milvus \
  --vector-backend milvus \
  --vector-dsn "$VECTOR_DSN" \
  --vector-collection-prefix smoke_milvus_v1 \
  --reranker-model none \
  --retrieval-profile auto \
  --query "单笔国内差旅报销金额超过 12000 元需要谁审批？"
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
        dsn="postgresql://leixiaoying:@127.0.0.1:5432/postgres",
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
            location="memory://manual/travel-policy",
            source_type=SourceType.PLAIN_TEXT,
            owner="manual",
            title="差旅制度",
            content_text="单笔国内差旅报销金额超过 12000 元时，必须由业务线 VP 审批。",
        )
    )

    runtime.insert(
        IngestRequest(
            location="/absolute/path/to/开票量明细.xlsx",
            source_type=SourceType.XLSX,
            owner="manual",
            title="开票量明细",
            file_path=Path("/absolute/path/to/开票量明细.xlsx"),
        )
    )

    result = runtime.query_public(
        "请计算华东区域 Q1 的开票量合计是多少？",
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
│   │   ├── llm_prompts.py             # route / plan / evaluate 等节点 prompt 模板
│   │   ├── llm_providers.py           # 节点 LLM provider 协议与实现
│   │   ├── llm_registry.py            # models.yaml 到节点模型的解析和缓存
│   │   ├── registry.py                # AgentDefinition 注册表
│   │   ├── subagent_runner.py         # 内置子 Agent runner
│   │   └── task.py                    # TaskDAG、SubTaskNode、SubTaskResult
│   ├── graphs/
│   │   ├── __init__.py                # graph 层公开导出
│   │   ├── base.py                    # LangGraph 主图 route/plan/execute/evaluate/synthesize
│   │   └── nodes/
│   │       ├── __init__.py            # graph node 公开导出
│   │       ├── evaluate.py            # 评估下一步：继续工具、子任务、暂停或合成
│   │       ├── execute.py             # 工具执行、预算记账、审批阻断、错误结构化
│   │       ├── execute_subagent.py    # Send() 子任务执行结果合并
│   │       ├── fast_path.py           # 单轮 RAG 快路径
│   │       ├── observe.py             # 工具结果观察和状态整理
│   │       ├── pause.py               # interrupt/resume 暂停节点
│   │       ├── plan.py                # 生成或接收 TaskDAG
│   │       ├── route.py               # 初始路由：fast_path / execute / plan / synthesize
│   │       └── synthesize.py          # 最终回答合成与 groundedness 标记
│   ├── memory/
│   │   ├── __init__.py                # memory 层公开导出
│   │   ├── compactor.py               # working memory 压缩和事实抽取
│   │   ├── injector.py                # bounded context 注入
│   │   └── models.py                  # WorkingSummary、ExtractedFact、budget snapshot
│   └── tools/
│       ├── __init__.py                # tools 层公开导出
│       ├── builtin_registry.py        # 内置 RAG/LLM/Agent 工具注册
│       ├── fast_path_tools.py         # rag_search_answer 快路径工具契约
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
│   ├── test_evaluate_context_integration.py # evaluate + context
│   ├── test_evaluate_task_dag.py      # TaskDAG evaluate
│   ├── test_execute_node_runtime.py   # execute node runtime
│   ├── test_execute_subagent_node.py  # execute_subagent node
│   ├── test_fast_path_node.py         # fast path node
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
│   ├── test_subagent_orchestration.py # subagent orchestration
│   ├── test_synthesize_agent_runner.py # synthesize Agent runner
│   ├── test_task_dag_contract.py      # TaskDAG schema/validation
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
- 每次真实实验建议使用新的 Postgres schema 和 Milvus collection prefix，避免不同 embedding 维度或旧 schema 污染结果。
- `9091` 被 Milvus 占用，rerank 服务使用 `9092`。
- 对表格真实值问题，不要信任 `sample_rows`；正确路径是 `<compute_request>` -> DuckDB -> `TABLE_COMPUTE_RESULT`。
- OpenAI-compatible chat provider 当前没有结构化生成实现，生成链路会先记录一次 structured failure，再走 text fallback；这是可见降级，不是静默 fallback。
- 批量入库脚本支持 `--summary-provider none`，公开 benchmark 可跳过 LLM 摘要生成，直接用 passage 原文入 summary index。

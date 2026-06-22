<div align="center">

# 面向私有业务文档的本地知识 Agent

面向私有知识库的 evidence-first RAG 与通用工具型 Agent runtime。

<p>
  <img src="https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.12">
  <img src="https://img.shields.io/badge/Package-uv-2b3137?style=for-the-badge" alt="uv">
  <img src="https://img.shields.io/badge/Agent-generic_loop-1f6feb?style=for-the-badge" alt="generic agent loop">
  <img src="https://img.shields.io/badge/Vector-Milvus-00a1ea?style=for-the-badge" alt="Milvus">
  <img src="https://img.shields.io/badge/Metadata-PostgreSQL-336791?style=for-the-badge&logo=postgresql&logoColor=white" alt="PostgreSQL">
</p>

<p>
  <img src="https://img.shields.io/badge/Chat-mimo--v2.5--pro-111827?style=for-the-badge" alt="mimo-v2.5-pro">
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

这个项目覆盖文档解析、结构化入库、摘要索引、混合检索、原文精读、表格计算、引用回答、离线评测，以及一个单入口的通用 Agent runtime。RAG 负责把私有资料定位成可复查证据；Agent 负责在工具之间循环，处理本地文件、检索、表格分析、代码执行、审批、checkpoint 和最终回答。

当前架构的核心不是“多个角色 Agent 互相编排”，而是：

```text
用户任务 -> generic AgentLoop -> 模型选择工具 -> 工具执行 -> 结果回模型上下文 -> 最终答案
```

RAG 是 Agent 可以调用的一类工具，不是所有任务的默认入口。对本地 Excel、CSV、JSON 等文件分析，优先走 workspace 文件工具和 Python 分析；只有需要知识库证据时才走 RAG 检索。

## 快速开始

```bash
cd "/Users/leixiaoying/LLM/RAG学习"
uv sync
```

准备默认云模型 key：

```bash
cat > .env <<'EOF'
MIMO_API_KEY=your_mimo_key
DEEPSEEK_API_KEY=your_deepseek_key_optional
EOF
```

启动 embedding 服务：

```bash
screen -S rag_embedding_9090 -X quit >/dev/null 2>&1 || true
screen -dmS rag_embedding_9090 zsh -lc '
cd "/Users/leixiaoying/LLM/RAG学习"
uv run rag embedding-service \
  --model mlx-community/Qwen3-Embedding-4B-4bit-DWQ \
  --port 9090 \
  --batch-size 1
'

export RAG_EMBEDDING_SERVICE_URL="http://127.0.0.1:9090"
export VECTOR_DSN="http://127.0.0.1:19530"
export VECTOR_PREFIX="quickstart_v1"
```

最小 RAG smoke：

```bash
uv run rag ingest \
  --storage-root data/quickstart \
  --vector-backend milvus \
  --vector-dsn "$VECTOR_DSN" \
  --vector-collection-prefix "$VECTOR_PREFIX" \
  --source-type plain_text \
  --location memory://quickstart/support-sla \
  --title "示例客服 SLA" \
  --owner quickstart \
  --content "示例客服 SLA：P1 工单首次响应目标为 30 分钟，解决目标为 4 小时。"

uv run rag query \
  --query "P1 工单首次响应目标是多少？请给出处" \
  --storage-root data/quickstart \
  --vector-backend milvus \
  --vector-dsn "$VECTOR_DSN" \
  --vector-collection-prefix "$VECTOR_PREFIX" \
  --reranker-model none \
  --retrieval-profile auto
```

最小 Agent 文件分析 smoke：

```bash
uv run rag agent run \
  "读取这个文件，列出结构并给出摘要" \
  --agent generic \
  --input-file "/absolute/path/to/file.xlsx" \
  --verbose
```

## News

- 2026-06-22：README 重写为当前架构说明：单 `generic` 入口、Claude-like while loop、Tool Search、deferred tools、workspace 文件工具和 `task` 工具。旧的 `ResearchAgent / OrchestratorAgent / agent_*` 默认路径不再作为当前设计描述。
- 2026-06-22：`fix/pr0-tool-result-feedback` 已合并到 `main`。native tool calling 路径会把工具结果写回下一轮模型上下文，为后续清理 `LoopState` 提供基线。
- 2026-06-14：引入 capability runtime：核心工具常驻，RAG、资产、LLM 和结构化分析工具延迟暴露，通过 `tool_search` 和 `activate_tools` 激活。
- 2026-06-14：PR0 升级 prompt/tool 协议：section 化 system prompt、内部 `ModelMessage`、OpenAI-compatible `tools=`、工具结果消息回灌。
- 2026-06-12：单 Agent 内核迁移到 Claude-like while loop。LangGraph 保留为外层复杂 workflow，不再表达普通 `model -> tool -> result` 循环。
- 2026-05：RAG 侧完成多格式入库、summary index、Milvus/PostgreSQL、asset grounding、DuckDB 表格计算和私有检索评测基线。

## 能力一览

| 能力 | 当前状态 | 关键实现 |
| --- | --- | --- |
| 多格式入库 | 支持 PDF、Word、Markdown、Excel、PPT、图片、纯文本 | `rag/ingest/pipeline.py`、`rag/ingest/parsers/*` |
| 多粒度索引 | doc / section / asset 三类 summary index | `SummaryRecord`、Milvus collections |
| 混合检索 | 支持 `fast / auto / deep / asset / bypass` profile | `rag/retrieval/l3_l4_engine.py`、`rag/retrieval/orchestrator.py` |
| Grounding | 原文回读、anchor replacement、neighbor expansion | `rag/retrieval/grounding_service.py` |
| 表格计算 | Excel asset 转 parquet，DuckDB 受限只读查询 | `rag/ingest/table_sampler.py`、`rag/ingest/table_executor.py` |
| 通用资产分析 | asset list / inspect / read slice / analyze | `rag/agent/tools/asset_tools.py` |
| Agent 内核 | 单 generic loop，模型选择工具，运行时负责安全边界 | `rag/agent/loop/runtime.py` |
| 工具发现 | core tools 常驻，deferred tools 搜索后激活 | `rag/agent/capabilities/catalog.py`、`tool_search.py` |
| 本地文件工具 | workspace 导入、列文件、读文件、写文件、Python 执行、结构探测 | `rag/agent/primitive_ops.py` |
| 子任务隔离 | 通用 `task` 工具启动有界 child loop | `rag/agent/tools/task_tool.py` |
| 评测 | 公开 MedicalRetrieval mini + 私有 golden queries | `scripts/evaluate_private_retrieval.py` |

## 系统流程

```mermaid
%%{init: {"theme": "base", "themeVariables": {"fontSize": "11px", "primaryColor": "#f8fafc", "primaryTextColor": "#0f172a", "primaryBorderColor": "#cbd5e1", "lineColor": "#94a3b8", "clusterBkg": "#ffffff", "clusterBorder": "#e2e8f0"}, "flowchart": {"nodeSpacing": 18, "rankSpacing": 22, "curve": "basis", "padding": 4}} }%%
flowchart TB
    subgraph RAG["RAG evidence pipeline"]
        direction LR
        r1["文件"] --> r2["解析"] --> r3["section / asset"] --> r4["summary"] --> r5["embedding"] --> r6["检索"] --> r7["grounding"] --> r8["引用回答"]
    end

    subgraph AG["Generic Agent runtime"]
        direction LR
        a1["用户任务"] --> a2["AgentRuntimePolicy"] --> a3["可见工具"] --> a4["模型 turn"] --> a5["工具执行"] --> a6["工具结果回上下文"] --> a7["最终答案"]
        a3 --> a8["tool_search / activate_tools"]
        a5 --> a9["RAG / asset / files / python / task"]
    end

    classDef rag fill:#eef6ff,stroke:#7aa7d9,color:#0f172a;
    classDef agent fill:#f4f0ff,stroke:#a78bfa,color:#0f172a;
    class r1,r2,r3,r4,r5,r6,r7,r8 rag;
    class a1,a2,a3,a4,a5,a6,a7,a8,a9 agent;
```

RAG 负责可引用证据和检索质量；Agent 负责在工具之间做有界循环。两者共享工具契约、预算、权限、失败显式化和证据保真规则。

## 架构总览

这个项目的主要业务场景是私有资料问答和文件分析：制度/流程文档回答审批规则、费用报销、销售政策；销售日报 Excel 回答提货量、区域汇总、产品口径；PPT/Word/PDF 回答跨文档事实并给出处。系统分成两层：RAG evidence pipeline 和 generic Agent runtime。

```text
L1 Storage：事实层
  原始文件、Document、SectionRecord、AssetRecord、locator、权限、版本、处理状态

L2 Indexing：索引层
  文档摘要、正文 section 摘要、Excel/PPT/图片资产摘要 -> Embedding -> Milvus

L3 Planning：查询规划
  选择 fast / auto / deep / asset / bypass，生成 retrieval signals

L4 Retrieval：候选召回
  多粒度 summary 检索、候选清洗、融合、可选 rerank、召回诊断

L5 Grounding：证据回读
  回读原文、邻近 section、asset anchor、表格对象和计算结果

L6 Synthesis：回答合成
  基于 EvidenceItem 生成回答、引用、权限/合规复核

Agent Runtime：任务执行
  generic AgentLoop、ToolCatalog、DeferredToolStore、ToolExecutionService、Workspace、Memory、Checkpoint
```

### L1：事实层

L1 保存事实数据和可追溯定位信息。制度条款、报销审批规则、销售政策正文落在 `Document / SectionRecord`；Excel sheet、PPT 表格、图片 OCR 区域等非正文内容落在 `AssetRecord`：

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

L3 判断查询应该如何检索，L4 负责候选召回和排序。系统支持这些 `retrieval_profile`：

- `bypass`
- `fast`
- `auto`
- `deep`
- `asset`

普通制度问答通常走 `auto`；销售日报、Excel 数字、PPT 表格和图片 OCR 问题优先走 `asset`；跨多个制度或需要多跳证据时使用 `deep`。

### L5：精读与证据层

L5 将 summary 命中的候选重新映射回原始正文或资产对象，确保最终答案不是只基于摘要猜测：

- 命中正文 section 后，通过 `visible_text_key + byte_range` 回读原文。
- 命中含表格锚点的 section 后，通过 `[ASSET_ANCHOR:...]` 找到绑定资产。
- 表格资产通过 DuckDB sandbox 执行受限只读查询。
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
- 摘要输入输出按 token 裁剪。
- L5 grounding 和 Agent `BudgetLedger` 都按 token 记账。
- 大工具结果进入有界 observation、summary 或外部引用，不直接塞进长期状态。

### Asset-Aware Retrieval

表格、图片、OCR、PPT 表格都作为 `AssetRecord` 独立保存。正文中保留 `[ASSET_ANCHOR:...]`，精读阶段再解析锚点并回填对应资产 evidence。

### DuckDB Table Sandbox

表格资产以 `schema / sample_rows / row_count / column_count / storage_key` 进入上下文。涉及过滤、排序、聚合、排名、交叉对比的问题，由模型生成受限只读查询，交给 DuckDB sandbox 执行，再将计算结果交给合成层。

### Evidence Over Memory

Agent memory 用于 working memory compaction / injection，帮助控制上下文窗口。回答事实优先级为 RAG evidence 高于 memory；当两者冲突时，以 evidence 为准。

## Agent 编排层

Agent 层采用 tool-centric + Python while-loop kernel 设计。LangGraph 保留为外层复杂编排器，不再承担单 Agent 的逐轮控制。

### 设计说明

当前默认 Agent 只有一个内置入口：`generic`。它没有 `ResearchAgent / CompareAgent / FactCheckAgent / SynthesizeAgent` 这样的角色身份。检索、对比、事实核查、生成、文件读取、表格分析和子任务隔离都表达为工具或 runtime capability，由模型在同一个 loop 中选择。

运行链路如下：

```text
AgentRunRequest
  -> AgentRunConfig / AgentRuntimePolicy
  -> AgentService.run()
     -> create workspace
     -> import input_files
     -> inject PrimitiveOps runners
     -> build ToolCatalog + DeferredToolStore
  -> AgentLoop
     -> execute pending tool calls if any
     -> compact context when required
     -> ModelTurnProvider.next_turn()
        -> execute: checkpoint and continue
        -> finish: run stop hooks, then complete
        -> pause: persist pause state for resume
  -> AgentRunResult
```

核心边界如下：

- `AgentRuntimePolicy` 是目标运行契约：系统指令、core tools、deferred tools、预算、迭代上限、模型选择、工具策略和输出 schema。
- `AgentDefinition` 仍存在，但主要是兼容适配层。新设计应逐步以 `AgentRuntimePolicy` 为 source of truth。
- `ToolSpec` 是工具契约来源，包括 Pydantic 输入输出、权限、timeout、retry、预算成本和是否需要确认。
- `ToolCatalog` 只负责可搜索工具元数据；`DeferredToolStore` 负责本次 run 激活了哪些 deferred tools。
- `ToolRegistry` 和 `ToolExecutionService` 才是执行权威，负责输入输出校验、权限、审批、重试、并发和失败结构化。
- `LoopState` 是 bounded mutable state，保存 messages、pending calls、tool results、observations、evidence、citations、pause、terminal 和 diagnostics。
- `ModelTurnProvider` 每轮只能返回 `execute(tool_calls)`、`finish(final_answer)` 或 `pause(reason)`。
- Stop hooks 只检查模型提出的 finish，不参与普通工具选择。

工具可见性分两层：

| 类型 | 默认可见 | 说明 |
| --- | --- | --- |
| Core tools | 是 | `tool_search`、`activate_tools`、`task`、`list_files`、`read_file`、`write_file`、`run_python_inline` |
| Deferred tools | 否 | RAG、asset、LLM、`structured_probe` 等，搜索并激活后才进入下一轮模型工具列表 |
| Internal tools | 否 | 运行时内部能力，不直接暴露给模型 |

本地文件任务优先走 workspace：

```text
--input-file / AgentRunRequest.input_files
  -> workspace/input_files/
  -> list_files
  -> structured_probe or read_file
  -> run_python_inline
  -> write_file when需要产物
```

RAG 任务优先走证据工具：

```text
tool_search("RAG / search documents")
  -> activate_tools(["rag_search_answer", ...])
  -> rag_search_answer or vector_search / grounding / rerank
  -> answer with evidence and citations
```

复杂子任务走 `task` 工具。`task` 启动一个有界 child loop，继承父级 runtime policy、工具注册表、访问策略和预算约束，但默认禁用再次递归调用 `task`。

### 已实现

- Claude-like `AgentLoop`：普通 while loop 表达 `model -> tool -> observation -> model`。
- Native tool calling path：通过 OpenAI-compatible `messages + tools` 调模型，并把 tool result 回灌到下一轮模型上下文。
- Prompt assembly：`AgentMessageAssembler` 将稳定 system sections 和动态 runtime context 分离。
- 工具发现：`ToolCatalog` 用 BM25 搜索 deferred tools，`activate_tools` 激活本次 run 的候选工具。
- 单内置 Agent：`BUILTIN_AGENT_DEFINITIONS` 当前只包含 `generic`。
- Workspace + PrimitiveOps：支持 `list_files`、`read_file`、`write_file`、`run_python`、`run_python_inline`、`structured_probe`。
- 通用 `task` 工具：替代默认 role-specific child agents。
- 工具执行：保留 `ToolExecutionRecord`、审批、幂等恢复、reconciliation、timeout、retry 和结构化错误。
- Memory：支持 working summary、extracted facts、bounded context injection 和 workspace-backed memory store。
- Checkpoint：通过 LangGraph checkpoint saver 兼容现有恢复路径。

### 已具备的能力

- 可以用 `rag agent run` 启动一个 generic Agent run，并返回结构化 `AgentRunResult`。
- 可以把本地文件导入 workspace，让 Agent 直接读取、探测和用 Python 分析，不必先入 RAG。
- 可以让模型用 `tool_search` 找到 RAG、asset、LLM、structured probe 等 deferred tools，再显式激活。
- 可以执行已注册工具，并在工具未注册、runner 缺失、参数非法、输出非法、预算不足、timeout、runner 异常时返回结构化失败结果。
- 可以在工具需要审批、上下文溢出或外部输入时进入 pause，并保留 resume 所需状态。
- 可以把 RAG evidence、citation、retrieval score、rerank score、asset locator 和 artifact path 带到最终回答。

### 下一步规划实现

- 清理 `LoopState`：先明确 tool call / tool result / model message 的唯一 source of truth，再删除 PR0/PR1 遗留字段。
- 让 `AgentRuntimePolicy` 成为 `AgentService`、factory、compiler 的主入口，`AgentDefinition` 降级为兼容适配。
- 收敛 legacy structured-output path，把 native tool calling 作为主线。
- 改善本地表格体验：对有 `--input-file` 的 `.xlsx/.csv` run，考虑预激活或提升 `structured_probe`。
- 完善 Skills / MCP 进入同一个 ToolCatalog 的 adapter，但不增加新的根 Agent 角色。
- 扩展 agent smoke tests，覆盖真实文件分析、tool result feedback、工具激活和 workspace 产物。

## 当前能力

### 文档入库

支持这些文件类型：

- `.pdf`
- `.docx`
- `.md / .markdown`
- `.xlsx / .xls`
- `.pptx`
- `.png / .jpg / .jpeg / .webp`
- `.txt`

解析路径：

- Word / PDF / Markdown：Docling 结构树和标题分段。
- Excel：Pandas / OpenPyXL 读取 sheet，表格作为 asset。
- PPTX：`python-pptx` 解析 slide 文本、表格、备注。
- 图片：OCR 模块抽取 visible text 和 OCR region。

### 检索与回答

- 三类 summary index：doc / section / asset。
- 支持 retrieval profile：`bypass / fast / auto / deep / asset`。
- 支持 rerank、candidate cleanup、neighbor expansion。
- grounding 回读原文 byte range 和 asset anchor。
- 表格查询走 DuckDB sandbox。
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
| chunk window | `480` tokens, overlap `64` tokens |
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

## 当前默认运行配置

模型目录统一在 `configs/models.yaml` 中维护，业务代码不直接写 provider、模型名、base URL 或 API key。

当前默认：

- `defaults.primary_model`：`mimo_cloud`
- `mimo_cloud.model`：`mimo-v2.5-pro`
- Embedding：`mlx-community/Qwen3-Embedding-4B-4bit-DWQ`
- Rerank：`BAAI/bge-reranker-v2-m3`

真实端到端推荐链路：

```text
PostgreSQL metadata
  + local object store / parquet table assets
  + Milvus vector indexes
  + Redis cache
  + Mimo or OpenAI-compatible chat
  + MLX embedding
  + optional BGE rerank
```

表格 / 资产分析规则：

- Excel 入库后表格资产会记录 `row_count / column_count / schema / sample_rows / storage_key`。
- 表格资产会转换为 DuckDB 可读的 `.parquet` 对象。
- 涉及真实数据值、筛选、求和、计数、排序、排名、对比或聚合的问题，必须执行受限计算。
- `sample_rows` 只用于识别 schema，不允许被当成完整表格直接回答。
- 不允许通过“总计/合计/小计”等业务关键词硬编码来修某一张表；如果问题缺少产品、sheet、日期或统计口径，应暴露歧义或要求澄清。

## 已验证端到端结果

下面是历史真实链路验证，保留用于说明系统能力边界。当前模型、索引和评测结果以实际命令输出为准。

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
MIMO_API_KEY=your_mimo_key
DEEPSEEK_API_KEY=your_deepseek_key_optional
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
| Optional local Qwen generation service | `8080` | 本地 OpenAI-compatible chat |
| Embedding service | `9090` | `mlx-community/Qwen3-Embedding-4B-4bit-DWQ` |
| Rerank service | `9092` | `BAAI/bge-reranker-v2-m3` |

## 模型服务管理

当前默认模型配置在 `configs/models.yaml`：

| 能力 | 默认别名 | 实际模型 / 服务 |
| --- | --- | --- |
| 生成 / 摘要 / Agent tool decision | `mimo_cloud` | `mimo-v2.5-pro`，OpenAI-compatible，`MIMO_API_KEY` |
| Embedding | `qwen3_embedding_4b_4bit_dwq` | `mlx-community/Qwen3-Embedding-4B-4bit-DWQ`，HTTP service，`127.0.0.1:9090` |
| Rerank | `bge_reranker_v2_m3` | `BAAI/bge-reranker-v2-m3`，HTTP service，`127.0.0.1:9092` |

内存策略：

- 默认 chat 走 Mimo 云服务，不需要本地常驻 Qwen。
- 入库和查询需要 embedding；建议启动 embedding HTTP 服务，避免每条命令重复加载模型。
- rerank 是可选服务，默认省内存时关闭。
- 切换 embedding 模型后必须换新的 Milvus collection prefix，旧向量不能混用。

先检查是否已经有同模型服务，避免重复常驻占内存：

```bash
ps aux | rg -i 'embedding-service|rerank-service|Qwen3-Embedding|bge-reranker|mlx_lm|vllm|ollama|uvicorn' \
  | rg -v 'rg -i|exec_command'

lsof -nP -iTCP -sTCP:LISTEN \
  | rg ':(8080|8081|8000|8001|9090|9091|9092|11434|19530|5432|6379)\b' || true
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

export RAG_EMBEDDING_SERVICE_URL="http://127.0.0.1:9090"
```

rerank 是可选服务。需要重排时再启动。注意 `9091` 被 Milvus 占用，rerank 用 `9092`：

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

如果你要改用本地 Qwen chat 服务，可以启动本地 OpenAI-compatible server，并在命令里用 `--model qwen3_8b_mlx_4bit`：

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

健康检查：

```bash
curl -sS http://127.0.0.1:9090/health
curl -sS http://127.0.0.1:9092/health
curl -sS http://127.0.0.1:8080/v1/models
screen -ls
```

关闭服务：

```bash
screen -S rag_qwen_8080 -X quit >/dev/null 2>&1 || true
screen -S rag_embedding_9090 -X quit >/dev/null 2>&1 || true
screen -S rag_rerank_9092 -X quit >/dev/null 2>&1 || true
```

## 私有文档端到端运行手册

先启动 embedding 服务；rerank 默认不开，需要时再按“常用开关”打开。默认 chat 走 `configs/models.yaml` 中的 `mimo_cloud`，需要 `.env` 里有 `MIMO_API_KEY`。

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
  --agent generic \
  --storage-root "$STORAGE_ROOT" \
  --vector-backend milvus \
  --vector-dsn "$VECTOR_DSN" \
  --vector-collection-prefix "$VECTOR_PREFIX" \
  --reranker-model none \
  --verbose
```

已入库资产问题：

```bash
uv run rag agent run \
  "日提货总量是多少？请检查相关表格并给出处" \
  --agent generic \
  --storage-root "$STORAGE_ROOT" \
  --vector-backend milvus \
  --vector-dsn "$VECTOR_DSN" \
  --vector-collection-prefix "$VECTOR_PREFIX" \
  --reranker-model none \
  --verbose
```

本地文件直接分析，不需要先入 RAG：

```bash
uv run rag agent run \
  "读取这个 Excel，汇总关键指标，并写一个简短摘要" \
  --agent generic \
  --input-file "/absolute/path/to/report.xlsx" \
  --verbose
```

期望工具链：

```text
本地文件：
  list_files -> structured_probe 或 run_python_inline -> final answer / write_file

知识库问题：
  tool_search -> activate_tools -> rag_search_answer 或 asset_inspect / asset_analyze -> final answer
```

交互式：

```bash
uv run rag agent chat \
  --agent generic \
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
| 开启 HTTP rerank | 启动 `rag_rerank_9092`，`export RAG_RERANK_SERVICE_URL=http://127.0.0.1:9092`，命令里不要传 `--reranker-model none` |
| 看 evidence / diagnostics | `rag query` 加 `--json` |
| 普通制度问答 | `--retrieval-profile auto` |
| Excel/PPT 表格/图片 OCR 资产问题 | `--retrieval-profile asset` |
| Agent 直接读本地文件 | `rag agent run ... --input-file "/path/to/file.xlsx"` |
| 一次性指定模型 | `--model mimo_cloud` 或 `--model qwen3_8b_mlx_4bit` |
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

说明：CLI 默认 metadata 仍可用本地 metadata repo 做快速验证。正式端到端可以使用下面的 `Postgres + parquet object + Milvus` runtime 配置。

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
        require_rerank=False,
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

静态检查：

```bash
uv run ruff check
uv run mypy
uv run lint-imports
```

Agent runtime / tool discovery / task tool：

```bash
uv run pytest -q \
  tests/agent/test_builtin_agents.py \
  tests/test_tool_discovery.py \
  tests/agent/test_task_tool.py \
  tests/agent/test_loop_model_context.py \
  tests/agent/test_agent_loop_runtime.py
```

表格计算、grounding、Postgres metadata：

```bash
uv run pytest -q \
  tests/core/test_table_compute_integration.py \
  tests/service/test_grounding_service.py \
  tests/repo/test_postgres_metadata_repo.py
```

复杂 RAG / Agent 回归：

```bash
uv run pytest -q \
  tests/agent/test_complex_agent_rag_loop.py \
  tests/service/test_complex_rag_retrieval.py
```

## 目录地图

下面按文件解释主要代码。这里列的是源码里应该维护的文件，不包含 `__pycache__`、本地 `data/` 产物和一次性诊断输出。

```text
./
├── README.md                          # 项目说明、架构、实验结果、运行命令
├── pyproject.toml                     # Python 项目元数据、依赖、pytest/ruff/mypy 配置
├── uv.lock                            # uv 锁文件
├── configs/models.yaml                # 默认 chat / embedding / rerank 模型配置
├── docs/agent_naming.md               # Agent 命名和迁移约定
├── docs/superpowers/specs/            # 已批准或历史设计规格
├── docs/superpowers/plans/            # 分阶段实现计划
└── scripts/                           # 入库、评测、诊断、benchmark 脚本
```

```text
rag/
├── cli.py                             # 主 CLI：ingest / query / benchmark / service / agent
├── runtime.py                         # AppRuntime 装配 storage、ingest、retrieval、synthesis
├── query_pipeline.py                  # 查询端 L3-L6 编排、表格 compute_request 循环
├── embedding_service.py               # 本地 embedding HTTP 服务入口
├── rerank_service.py                  # 本地 rerank HTTP 服务入口
├── agent/
│   ├── builtin/generic.py             # 当前唯一内置 Agent 定义
│   ├── builtin_registry.py            # 内置工具 registry composition root
│   ├── capabilities/
│   │   ├── catalog.py                 # ToolCatalog、DeferredToolStore、visible tool resolver
│   │   └── tool_search.py             # tool_search / activate_tools contracts 和 runner
│   ├── core/
│   │   ├── definition.py              # AgentRuntimePolicy、AgentDefinition 兼容层
│   │   ├── llm_providers.py           # native tool calling / legacy structured-output provider
│   │   ├── tool_schema.py             # prompt sections、ModelMessage 和 OpenAI adapter
│   │   ├── tool_execution.py          # ToolExecutionService、records、approval/reconciliation
│   │   ├── checkpointing.py           # checkpoint serde 和 LangGraph saver adapter
│   │   └── context.py                 # AgentRunConfig、BudgetLedger、RunRegistry
│   ├── loop/
│   │   ├── runtime.py                 # Claude-like while-loop kernel
│   │   ├── state.py                   # LoopState、ModelTurnDraft、transitions
│   │   └── stop_hooks.py              # stop hook 协议和 bounded runner
│   ├── memory/                        # working memory compaction / injection / store
│   ├── primitive_ops.py               # workspace 文件和 Python 原语
│   ├── service.py                     # AgentRunRequest / AgentRunResult / AgentService
│   ├── tools/
│   │   ├── asset_tools.py             # asset_list / inspect / read_slice / analyze
│   │   ├── llm_tools.py               # llm_generate / summarize / compare
│   │   ├── primitive_tools.py         # list/read/write/run_python/structured_probe specs
│   │   ├── rag_answer_tools.py        # rag_search_answer composite tool
│   │   ├── rag_tools.py               # vector_search / keyword_search / grounding / rerank
│   │   ├── registry.py                # ToolRegistry 和 runner 校验
│   │   ├── spec.py                    # ToolSpec、ToolPermissions、ToolResult、ToolError
│   │   └── task_tool.py               # generic task tool
│   └── workspace.py                   # per-run workspace runtime
├── assembly/                          # provider 组装、能力检测、token accounting
├── ingest/                            # parser、section、summary、asset、table executor
├── models/                            # models.yaml schema、catalog、runtime 解析
├── providers/                         # generation、embedding、rerank、LLM gateway
├── retrieval/                         # L3-L6 检索、grounding、fusion、synthesis
├── schema/                            # Document / Section / Asset / Query / Runtime schema
├── storage/                           # metadata、object store、cache、vector backend
└── utils/                             # guard、telemetry、text helper
```

```text
tests/
├── agent/                             # Agent loop、tools、memory、checkpoint、CLI、service
├── core/                              # ingest、runtime、model config、table compute、Milvus
├── provider/                          # embedding / rerank / LLM gateway
├── repo/                              # object store、Postgres metadata、HF providers
├── service/                           # retrieval、grounding、synthesis、authorization
├── ui/                                # CLI behavior
└── test_tool_discovery.py             # capability catalog 和 deferred tools
```

## 运行注意事项

- 入库和查询必须使用同一个 embedding space；切换 embedding 模型后必须重建 Milvus collection。
- 每次真实实验建议使用新的 `STORAGE_ROOT` 和 Milvus collection prefix，避免不同 embedding 维度或旧 schema 污染结果。
- `9091` 被 Milvus 占用，rerank 服务使用 `9092`。
- RAG 是 Agent 的一个工具，不是所有文件任务的默认入口。本地文件分析优先用 `--input-file`、workspace、`structured_probe` 和 `run_python_inline`。
- 对表格真实值问题，不要信任 `sample_rows`；正确路径是资产 inspect/read/analyze 或本地 Python 计算。
- OpenAI-compatible chat provider 的结构化输出能力依赖后端；降级必须可见，不能静默吞掉失败。
- 批量入库脚本支持 `--summary-provider none`，可跳过 LLM 摘要生成，直接用原文进入 summary index；质量会低于严格摘要链路。
- Agent CLI 当前默认 `--agent generic`。旧的 `research`、`orchestrator`、`compare`、`factcheck`、`synthesize` 不再是内置默认入口。

常见问题和处理顺序：

| 现象 | 主要原因 | 处理 |
| --- | --- | --- |
| `Component backend requires a DSN/URI` | `--vector-backend milvus` 但没有传 DSN，或 `VECTOR_DSN` 为空 | `echo "$VECTOR_DSN"`，命令中补 `--vector-dsn "$VECTOR_DSN"` |
| Milvus `vector dimension mismatch` | 查询 embedding 和入库 embedding 不是同一模型/维度 | 换新的 `VECTOR_PREFIX` 和 `STORAGE_ROOT` 重新入库，不要混用旧 collection |
| `Embedding service health check failed: Connection refused` | embedding 服务没启动或端口不对 | `curl http://127.0.0.1:9090/health`，失败就重启 `rag_embedding_9090` |
| `Rerank service health check failed` | 查询要求 rerank，但 rerank 服务没启动；或入库阶段误带 rerank | 入库前 `unset RAG_RERANK_SERVICE_URL`；关闭 rerank 查询时传 `--reranker-model none` |
| `document summary generation returned empty output` | 本地 Qwen3 thinking 未关闭，或生成服务异常 | 用 `enable_thinking=false` 重启 Qwen；或改回默认 Mimo 云模型 |
| 入库 10 分钟进度条不动 | 模型首次加载、Excel 解析大表、embedding batch 太大、内存压力高 | 先 `curl` 检查服务；用 `--batch-size 1 --embedding-batch-size 1`；Excel 用 `scripts/diagnose_ingest_timing.py` 定位 |
| DOCX DrawingML / VML 图片警告 | 缺少 LibreOffice 转换器或文档内图片引用缺失 | 正文通常仍可入库；如果必须抽图，安装 LibreOffice 并设置 `DOCLING_LIBREOFFICE_CMD` |
| RAG 表格答案没读到明显数字 | 没走 asset profile、没触发 compute、或问题口径不清 | 用 `--retrieval-profile asset --json` 看 `context.evidence` 和 diagnostics |
| Agent 没有工具结果 | 模型没有产生工具调用，或 deferred tool 未激活 | 用 `--verbose` 看可见工具；必要时让任务明确先 `tool_search` 或先列 workspace 文件 |
| Agent 表格问题乱选 sheet/产品/口径 | 用户问题有歧义，或 Agent 未做结构探测 | 正确行为是列候选或请求澄清；不要用硬编码业务关键词修某张表 |

DOCX 图形转换可选配置：

```bash
export DOCLING_LIBREOFFICE_CMD="/Applications/LibreOffice.app/Contents/MacOS/soffice"
```

Excel 入库耗时诊断：

```bash
uv run python scripts/diagnose_ingest_timing.py "$INPUT_PATH"
```

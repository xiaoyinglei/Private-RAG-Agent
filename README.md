# 企业知识 RAG 与 Agent 编排系统

这是一个面向企业私有知识库的 RAG 与 Agent 编排项目。系统覆盖文档解析、结构化入库、摘要索引、混合检索、原文精读、表格计算、引用回答、离线评测，以及基于 LangGraph 的多 Agent 任务编排。

项目目标是让企业内部制度、流程、销售资料、Word/PDF/Excel/PPT/图片等异构资料进入同一套可检索、可引用、可评测的知识系统，并为复杂问题提供可拆解、可追踪、可失败显式化的 Agent 执行框架。

## 系统流程

```text
原始文件
  -> Parser
  -> Document / SectionRecord / AssetRecord
  -> SectionRefiner token 窗口
  -> Doc / Section / Asset 三类摘要
  -> Embedding
  -> Milvus summary indexes
  -> Planning
  -> Retrieval / Rerank
  -> Grounding / Asset resolution / Table SQL
  -> EvidenceItem
  -> Synthesis with citations
```

Agent 编排层运行在 RAG 能力之上：

```text
AgentDefinition
  -> ToolSpec / ToolRegistry
  -> AgentRunConfig / BudgetLedger
  -> AgentState reducers
  -> LangGraph route / plan / execute / evaluate / synthesize
  -> TaskDAG + Send() 子任务并行
  -> AgentRunResult
```

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

## 评测结果

### 公开数据：MedicalRetrieval mini

| 基线 | 向量后端 | embedding | rerank | Recall@10 | MRR@10 | NDCG@10 | avg_latency_ms |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| `BAAI/bge-m3 + sqlite` | SQLite | `BAAI/bge-m3` | on | 0.776667 | 0.690972 | 0.712199 | 2472.225 |
| `BAAI/bge-m3 + milvus` | Milvus | `BAAI/bge-m3` | on | 0.670000 | 0.588259 | 0.608173 | 563.793 |
| `qwen3-embedding:8b + milvus` | Milvus | `qwen3-embedding:8b` | on | 0.820000 | 0.705854 | 0.733644 | 695.559 |

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

## 目录地图

```text
rag/
├── runtime.py                         # 装配 storage / ingest / retrieval / synthesis
├── cli.py                             # ingest / query / benchmark-download / benchmark-ingest / benchmark-evaluate
├── benchmarks.py                      # benchmark 和 runtime build helper
├── agent/                             # LangGraph Agent 编排层
│   ├── core/                          # AgentRunConfig / AgentDefinition / Registry / AgentAsToolRunner
│   ├── graphs/                        # base graph 与 route/plan/execute/evaluate/synthesize 节点
│   ├── memory/                        # working memory compaction / context injection
│   ├── tools/                         # ToolSpec / ToolRegistry / RAG & LLM tool specs
│   ├── builtin/                       # ResearchAgent 定义与 service factory
│   ├── service.py                     # AgentRunRequest / AgentRunResult / AgentService
│   └── state.py                       # AgentState TypedDict + reducers
├── assembly/                          # provider / tokenizer 装配
├── ingest/
│   ├── pipeline.py                    # L1/L2 入库链路
│   ├── parsers/                       # docling / excel / pptx / image parser
│   ├── section_refiner.py             # token 窗口切分
│   ├── retrievalsummarizer.py         # doc/section/asset 三类摘要
│   ├── table_sampler.py               # 表格 schema/sample/profile/table_policy
│   └── asset_anchors.py               # [ASSET_ANCHOR:...] 工具
├── retrieval/
│   ├── planning_graph.py              # L3 planning
│   ├── retrieval_adapter.py           # L4 summary index retrieval adapter
│   ├── rerank_service.py              # 候选清洗 + rerank
│   ├── grounding_service.py           # L5 raw read / anchor replacement / DuckDB table sandbox trigger
│   └── synthesis_service.py           # L6 synthesis
├── schema/
│   ├── core.py                        # Document / SectionRecord / AssetRecord / SummaryRecord
│   ├── query.py                       # GroundingTarget / EvidenceItem / answer contract
│   └── runtime.py                     # Runtime contract / diagnostics / vector result
└── storage/
    ├── data_contract_service.py       # 数据契约服务
    ├── repositories/                  # sqlite/postgres metadata repo
    └── search_backends/               # milvus/sqlite vector repo
```

```text
scripts/
├── ingest_private_documents.py        # 私有文件夹入库
├── export_private_sections.py         # 从 index 导出 SectionRecord JSONL
├── evaluate_private_retrieval.py      # 私有 golden set 检索评测
├── download_public_benchmark.py       # 下载公开 benchmark
├── prepare_public_benchmark.py        # 准备公开 benchmark
├── ingest_public_benchmark.py         # 公开 benchmark 入库
└── profile_benchmark_ingest.py        # 入库速度 profiling
```

## 环境准备

### 1. 安装依赖

```bash
uv sync
```

### 2. 启动 Milvus

```bash
export MILVUS_URI=http://127.0.0.1:19530
export RAG_MILVUS_URI=$MILVUS_URI
```

### 3. 准备 embedding 模型

```bash
ollama pull qwen3-embedding:8b
ollama serve
```

### 4. 准备摘要模型

```bash
export SUMMARY_MODEL=Qwen/Qwen3-8B-MLX-4bit
```

生成测试题脚本 `generate_eval_dataset.py` 使用 OpenAI-compatible 接口，可以用 MLX server 提供本地模型服务：

```bash
uv run mlx_lm.server \
  --model Qwen/Qwen3-8B-MLX-4bit \
  --host 127.0.0.1 \
  --port 8080 \
  --max-tokens 1024 \
  --temp 0.1 \
  --chat-template-args '{"enable_thinking":false}'
```

## 常用命令

### 私有文档入库

```bash
export INPUT_DIR="/path/to/private_documents"
export STORAGE_ROOT=data/eval_private/company_policy_milvus_v4
export COLLECTION_PREFIX=company_policy_v4
export MILVUS_URI=http://127.0.0.1:19530
export EMBEDDING_MODEL=qwen3-embedding:8b
export SUMMARY_MODEL=Qwen/Qwen3-8B-MLX-4bit
export CHUNK_TOKEN_SIZE=800
export CHUNK_OVERLAP_TOKENS=120

uv run python scripts/ingest_private_documents.py \
  --input "$INPUT_DIR" \
  --storage-root "$STORAGE_ROOT" \
  --owner private \
  --batch-size 8 \
  --continue-on-error \
  --embedding-provider ollama \
  --embedding-model "$EMBEDDING_MODEL" \
  --embedding-batch-size 8 \
  --summary-provider local-hf \
  --summary-model "$SUMMARY_MODEL" \
  --summary-backend mlx \
  --vector-backend milvus \
  --vector-dsn "$MILVUS_URI" \
  --vector-collection-prefix "$COLLECTION_PREFIX" \
  --chunk-token-size "$CHUNK_TOKEN_SIZE" \
  --chunk-overlap-tokens "$CHUNK_OVERLAP_TOKENS" \
  --output data/eval_private/company_policy_ingest_v4.json
```

### 导出 SectionRecord

```bash
uv run python scripts/export_private_sections.py \
  --storage-root "$STORAGE_ROOT" \
  --output data/eval_private/company_policy_sections_v4.jsonl
```

### 生成 golden eval 测试集

```bash
uv run python generate_eval_dataset.py \
  --input data/eval_private/company_policy_sections_v4.jsonl \
  --output data/eval_private/golden_eval_dataset_v4.jsonl \
  --failed-output data/eval_private/golden_eval_failed_v4.jsonl \
  --model "$SUMMARY_MODEL" \
  --base-url http://127.0.0.1:8080/v1 \
  --api-key not-needed \
  --max-window-tokens 700 \
  --window-overlap-tokens 80 \
  --min-window-tokens 120
```

### 检索评测

```bash
uv run python scripts/evaluate_private_retrieval.py \
  --golden-path data/eval_private/golden_eval_dataset_v4.jsonl \
  --storage-root "$STORAGE_ROOT" \
  --retrieval-profile auto \
  --top-k 20 \
  --retrieval-pool-k 20 \
  --neighbor-radius 1 \
  --rerank \
  --rerank-provider local-bge \
  --rerank-model BAAI/bge-reranker-v2-m3 \
  --embedding-provider ollama \
  --embedding-model "$EMBEDDING_MODEL" \
  --vector-backend milvus \
  --vector-dsn "$MILVUS_URI" \
  --vector-collection-prefix "$COLLECTION_PREFIX" \
  --chunk-token-size "$CHUNK_TOKEN_SIZE" \
  --chunk-overlap-tokens "$CHUNK_OVERLAP_TOKENS" \
  --output data/eval_private/private_retrieval_eval_v4_rerank.json \
  --misses-output data/eval_private/private_retrieval_misses_v4_rerank.jsonl
```

### 公开 benchmark

```bash
uv run python scripts/download_public_benchmark.py --dataset medical_retrieval
uv run python scripts/prepare_public_benchmark.py --dataset medical_retrieval

uv run python scripts/ingest_public_benchmark.py \
  --dataset medical_retrieval \
  --variant mini \
  --storage-root data/benchmarks/medical_retrieval/index/mini-milvus-qwen8b-v1 \
  --vector-backend milvus \
  --vector-dsn "$MILVUS_URI" \
  --vector-collection-prefix medical_retrieval_mini_qwen8b_v1 \
  --batch-size 32 \
  --embedding-batch-size 8 \
  --embedding-provider ollama \
  --embedding-model qwen3-embedding:8b \
  --summary-provider local-hf \
  --summary-model Qwen/Qwen3-8B-MLX-4bit \
  --summary-backend mlx \
  --chunk-token-size 480 \
  --chunk-overlap-tokens 64 \
  --skip-graph-extraction

uv run rag benchmark-evaluate \
  --dataset medical_retrieval \
  --variant mini \
  --storage-root data/benchmarks/medical_retrieval/index/mini-milvus-qwen8b-v1 \
  --vector-backend milvus \
  --vector-dsn "$MILVUS_URI" \
  --vector-collection-prefix medical_retrieval_mini_qwen8b_v1 \
  --retrieval-profile auto \
  --top-k 10 \
  --evidence-top-k 20 \
  --rerank \
  --embedding-provider ollama \
  --embedding-model qwen3-embedding:8b \
  --rerank-provider local-bge \
  --rerank-model BAAI/bge-reranker-v2-m3
```

## 质量检查

```bash
uv run ruff check rag scripts tests
uv run pytest -q
```

Agent 模块专项测试：

```bash
uv run pytest tests/agent -q
```

表格与 grounding 相关回归：

```bash
uv run pytest -q \
  tests/core/test_excel_parser_repo.py \
  tests/core/test_ingest_asset_anchors.py \
  tests/core/test_retrieval_summarizer.py \
  tests/service/test_grounding_service.py
```

## 运行注意事项

- 入库与检索的 embedding、tokenizer、`chunk_token_size`、`chunk_overlap_tokens` 必须一致。
- 每次新实验建议使用新的 `STORAGE_ROOT` 和 `COLLECTION_PREFIX`。
- 表格资产保存 schema、sample、summary 和可计算存储指针，计算问题交给 DuckDB Sandbox。
- 生成测试题前先运行 `export_private_sections.py`。
- `generate_eval_dataset.py` 的失败文件只会在出现失败样本时生成。

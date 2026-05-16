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

## 当前默认运行配置

模型目录统一在 `configs/models.yaml` 中维护，业务代码不直接写 provider、模型名、base URL 或 API key。

当前默认：

- Chat：`deepseek-v4-flash`
- Embedding：`mlx-community/Qwen3-Embedding-8B-4bit-DWQ`
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
| Embedding service | `9090` | `mlx-community/Qwen3-Embedding-8B-4bit-DWQ` |
| Rerank service | `9092` | `BAAI/bge-reranker-v2-m3` |

## 模型服务管理

先检查是否已经有同模型服务，避免重复常驻占内存：

```bash
ps aux | rg -i 'embedding-service|rerank-service|Qwen3-Embedding|bge-reranker|mlx_lm|vllm|ollama|uvicorn' \
  | rg -v 'rg -i|exec_command'

lsof -nP -iTCP -sTCP:LISTEN \
  | rg ':(8080|8081|8000|8001|9090|9091|9092|11434|19530|5432|6379)\b' || true
```

如果发现重复的 embedding/rerank 服务，先杀旧进程。不要杀 Milvus、Postgres、Redis：

```bash
kill <old_embedding_pid> <old_rerank_pid>
```

启动唯一一份 embedding 服务：

```bash
screen -S rag_embedding_9090 -X quit >/dev/null 2>&1 || true
screen -dmS rag_embedding_9090 zsh -lc '
cd /Users/leixiaoying/LLM/RAG学习 &&
uv run rag embedding-service \
  --model mlx-community/Qwen3-Embedding-8B-4bit-DWQ \
  --port 9090
'
```

启动唯一一份 rerank 服务：

```bash
screen -S rag_rerank_9092 -X quit >/dev/null 2>&1 || true
screen -dmS rag_rerank_9092 zsh -lc '
cd /Users/leixiaoying/LLM/RAG学习 &&
uv run rag rerank-service \
  --model BAAI/bge-reranker-v2-m3 \
  --port 9092
'
```

健康检查：

```bash
curl -sS http://127.0.0.1:9090/health
curl -sS http://127.0.0.1:9092/health
screen -ls
```

预期返回：

```json
{"model":"mlx-community/Qwen3-Embedding-8B-4bit-DWQ","embedding_space":"default","dimension":4096}
{"model":"BAAI/bge-reranker-v2-m3"}
```

## CLI 快速验证

CLI 会读取 `configs/models.yaml` 的默认模型。若已经启动 `9090/9092` 服务，可以通过 HTTP provider 复用常驻模型，避免 CLI 每次重新加载：

```bash
export RAG_EMBEDDING_SERVICE_URL=http://127.0.0.1:9090
export RAG_RERANK_SERVICE_URL=http://127.0.0.1:9092
```

文本入库：

```bash
uv run rag ingest \
  --storage-root data/smoke_milvus \
  --source-type plain_text \
  --location memory://smoke/travel-policy \
  --title "差旅制度 Smoke" \
  --owner smoke \
  --content "单笔国内差旅报销金额超过 12000 元时，必须由业务线 VP 审批。"
```

Excel 入库：

```bash
uv run rag ingest \
  --storage-root data/smoke_milvus \
  --source-type xlsx \
  --location /absolute/path/to/开票量明细.xlsx \
  --title "开票量明细" \
  --owner smoke
```

查询：

```bash
uv run rag query \
  --storage-root data/smoke_milvus \
  --retrieval-profile auto \
  --query "单笔国内差旅报销金额超过 12000 元需要谁审批？"

uv run rag query \
  --storage-root data/smoke_milvus \
  --retrieval-profile asset \
  --query "请计算华东区域 Q1 的开票量合计是多少？"
```

说明：当前 CLI 的 metadata 默认仍是本地 metadata repo，适合快速验证。正式端到端使用下面的 `Postgres + parquet object + Milvus` runtime 配置。

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

## 运行注意事项

- 入库和查询必须使用同一个 embedding space；切换 embedding 模型后必须重建 Milvus collection。
- 每次真实实验建议使用新的 Postgres schema 和 Milvus collection prefix，避免不同 embedding 维度或旧 schema 污染结果。
- `9091` 被 Milvus 占用，rerank 服务使用 `9092`。
- 对表格真实值问题，不要信任 `sample_rows`；正确路径是 `<compute_request>` -> DuckDB -> `TABLE_COMPUTE_RESULT`。
- OpenAI-compatible chat provider 当前没有结构化生成实现，生成链路会先记录一次 structured failure，再走 text fallback；这是可见降级，不是静默 fallback。
- `scripts/ingest_private_documents.py` 仍保留旧的批量入库入口，但它的 embedding 参数默认是 legacy provider。使用当前默认 MLX embedding 时，优先使用 runtime 配置或先补齐脚本参数支持。

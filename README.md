# 面向私有业务文档的本地知识 Agent

我做这个项目，是为了把企业内部的制度、流程、销售资料、表格、PPT、图片和网页资料，整理成一套可以检索、可以回看原文、可以带引用回答、也可以被 Agent 调用的知识系统。

这个项目不是一个只把文档切块后丢进向量库的 demo。它的核心是把“检索入口”和“事实来源”分开：索引用 summary 快速定位，真正回答时再回到原文、表格或资产对象里取证据。这样做的原因很直接：企业知识问答不能只给一个相似段落，它要能说明答案来自哪份文件、哪一段、哪张表，以及检索和重排过程发生了什么。

## 我怎么讲这个项目

整套系统可以按一条主线理解：

```text
原始资料
  -> parser 解析成文档、章节、表格、图片/OCR 等结构
  -> L1 保存事实记录和原文定位
  -> L2 生成 doc / section / asset 三类 summary index
  -> L3 规划检索策略
  -> L4 多路召回、融合、rerank
  -> L5 回读原文和资产，补齐 evidence
  -> L6 只基于 evidence 生成带引用答案
  -> Agent 把这些能力封装成工具，用 LangGraph 编排复杂任务
```

对应到代码里：

```text
rag/ingest      负责解析、入库、摘要和索引
rag/storage     负责 metadata、object store、cache、vector repo
rag/retrieval   负责 L3/L4 检索、grounding、权限过滤和 synthesis
rag/providers   负责 embedding、rerank、chat、citation formatting
rag/assembly    负责把模型、tokenizer、provider 和 runtime capability 装起来
rag/models      负责 configs/models.yaml 的模型目录和运行时解析
rag/agent       负责 AgentDefinition、ToolSpec、LangGraph、审批、子 Agent 和 memory
scripts         负责公开 benchmark、私有文档入库和离线评测
tests           负责核心契约、RAG、Agent、storage、provider、service 回归
```

## 已经实现了什么

### 1. 多格式入库

入库入口是 `rag/ingest/pipeline.py`。它接收 `IngestRequest`，通过 `ExtractionDispatcher` 选择 parser，然后写入事实层和索引层。

当前支持的资料类型包括：

| 类型 | 处理方式 |
| --- | --- |
| PDF / DOCX / Markdown | 通过 Docling parser 解析结构和正文 |
| Excel | 抽取 sheet、schema、sample rows，并把表格资产转成可计算对象 |
| PPTX | 抽取 slide 文本、表格和备注 |
| 图片 | 走 OCR parser，保存 visible text 和 OCR region |
| Web / 纯文本 / pasted text / browser clip | 作为轻量文本源进入同一套 SectionRecord 流程 |

入库不是直接把全文塞进向量库。主流程会先写：

- `Source`
- `Document`
- `SectionRecord`
- `AssetRecord`
- `LayoutMetaCacheRecord`
- `ProcessingStateRecord`

然后再生成：

- `DocSummaryRecord`
- `SectionSummaryRecord`
- `AssetSummaryRecord`

这些 summary record 才会进入 Milvus 或 SQLite vector repo。

### 2. Summary-first 检索

索引层只保存检索需要的信息：summary、向量、稀疏向量、过滤字段、对象 ID 和 metadata。原文、表格、权限、版本、定位信息仍然在 metadata repo 和 object store 里。

这样查问题时不会先读全文，而是先从三类 summary index 定位：

| index | 用途 |
| --- | --- |
| `doc_summary` | 先判断哪些文档可能相关 |
| `section_summary` | 找具体章节 |
| `asset_summary` | 找表格、图片、OCR、PPT 表格等资产 |

切换 embedding 模型后必须重建索引，因为查询 embedding 和已入库 embedding 必须在同一个 embedding space 里。

### 3. L3/L4 检索编排

检索入口在 `rag/retrieval/orchestrator.py`，核心执行在 `rag/retrieval/l3_l4_engine.py`。

它做几件事：

- 用 `PlanningGraph` 判断检索 profile、复杂度、目标 collection、predicate plan 和 fallback plan。
- 从 vector、section、special、metadata、graph、web 等分支收集候选。
- 对候选做 fusion、cleanup、rerank 和 self-check。
- 在召回不足时按计划补充其他分支。
- 把诊断信息写进 `RetrievalDiagnostics`，方便评测和排查。

当前支持的 retrieval profile：

| profile | 用途 |
| --- | --- |
| `fast` | 快速问答，少量候选 |
| `auto` | 默认策略，按查询选择路径 |
| `deep` | 更重的召回和 grounding |
| `asset` | 偏向表格、图片、OCR、PPT 资产 |
| `bypass` | 不走检索，直接让模型回答 |

### 4. Grounding 和引用

`rag/retrieval/grounding_service.py` 负责把检索命中的 summary 候选重新映射回真实内容。

它会处理：

- 根据 `visible_text_key` 和 byte range 回读正文。
- 根据 section 的上下文补 neighbor section。
- 解析 `[ASSET_ANCHOR:...]`，把正文附近的表格、图片或 OCR 资产补进 evidence。
- 对 evidence 再做局部排序和 token budget 控制。
- 保留 `doc_id`、`section_id`、`asset_id`、page range、citation anchor、retrieval channels 等信息。

最终回答的数据结构是 `GroundedAnswer`，证据结构是 `EvidenceItem`。回答阶段不应该绕开 evidence 直接编造事实。

### 5. 表格计算

表格不是只拿 sample rows 给模型猜。Excel 入库后会保存表格 schema、sample、行列数和 `storage_key`，并在需要时转换为 DuckDB 可读对象。

查询阶段如果模型判断需要真实计算，会生成：

```text
<compute_request>{"asset_id": ..., "sql": "SELECT ..."}</compute_request>
```

`rag/ingest/table_executor.py` 只允许 `SELECT`。
它会拦截 `DROP / INSERT / UPDATE / DELETE / ATTACH / COPY` 等危险语句，并限制超时和返回行数。
执行结果会作为 `TABLE_COMPUTE_RESULT` 注入 evidence，再重新生成最终回答。

这个设计解决的是“表格问题不能靠向量相似度猜答案”的问题。

### 6. Agent 编排

Agent 层在 `rag/agent`。它不是另起一套聊天逻辑，而是把 RAG 能力、LLM 能力和子 Agent 都封成工具，再用 LangGraph 管理任务流转。

核心对象：

| 对象 | 作用 |
| --- | --- |
| `AgentDefinition` | 定义 agent 类型、系统提示、允许工具、预算、最大迭代和最大深度 |
| `ToolSpec` | 定义工具输入输出 schema、权限、timeout、retry、预算成本和是否需要确认 |
| `AgentState` | LangGraph 状态，使用 TypedDict 和 reducer 合并 evidence、citation、tool result 等 |
| `TaskDAG` | 表达复杂任务拆解、依赖和子任务状态 |
| `AgentService` | 对外运行边界，返回结构化 `AgentRunResult` |
| `RuntimeRegistry` | 保存 request-scoped runtime handle，避免并发 run 互相污染 |

当前内置 Agent 包括：

- `research`
- `orchestrator`
- `compare`
- `factcheck`
- `synthesize`

Agent graph 的主流程是：

```text
route
  -> fast_path | plan | execute | synthesize
  -> observe
  -> evaluate
  -> execute tool 或 execute_subagent
  -> pause approval 或 synthesize
```

工具执行采用 fail-closed 策略：工具未注册、runner 缺失、输入非法、输出非法、预算不足、超时、审批未通过，都会变成结构化错误，不做静默 fallback。

需要注意：`RAGRuntime.analyze_task` 和顶层 `rag analyze-task` 仍然是旧入口，当前已禁用。实际 Agent CLI 在 `rag agent run/chat/resume`。

## 当前默认配置

模型配置只从 `configs/models.yaml` 读取。业务代码不应该硬编码模型名、base URL 或 API key。

当前默认值：

| 能力 | alias | 模型 |
| --- | --- | --- |
| Chat | `deepseek_v4_flash` | `deepseek-v4-flash` |
| Embedding | `qwen3_embedding_4b_4bit_dwq` | `mlx-community/Qwen3-Embedding-4B-4bit-DWQ` |
| Rerank | `bge_reranker_v2_m3` | `BAAI/bge-reranker-v2-m3` |

tokenizer 默认配置：

| 配置 | 值 |
| --- | --- |
| `tokenizer_backend` | `auto` |
| `chunk_token_size` | `480` |
| `chunk_overlap_tokens` | `64` |
| `max_context_tokens` | `4096` |
| `prompt_reserved_tokens` | `256` |
| `local_files_only` | `true` |

如果通过 CLI 使用常驻 embedding/rerank 服务，环境变量优先级高于 YAML：

```bash
export RAG_EMBEDDING_SERVICE_URL=http://127.0.0.1:9090
export RAG_RERANK_SERVICE_URL=http://127.0.0.1:9092
```

显式 CLI 参数和这些 service URL 同时出现时会报错，不会偷偷选一个。

## 存储组合

默认 CLI 的存储组合适合本地验证：

```text
metadata: SQLite
object store: local files
vector: Milvus
cache: metadata-backed cache
graph: null graph repo
```

正式端到端可以显式换成：

```text
metadata: PostgreSQL
object store: local / S3 / MinIO
vector: Milvus
cache: Redis
graph: 当前默认 null，图谱 backend 仍需按 EvidenceItem contract 重建
```

`StorageConfig` 会按 component 单独装配 backend，所以同一套 runtime 可以在测试、本地验证和正式链路之间切换。

## 本地运行

安装依赖：

```bash
uv sync
```

准备 `.env`：

```bash
cp .env.example .env
```

至少需要填：

```bash
DEEPSEEK_API_KEY=your-key
```

确认基础设施：

```bash
lsof -nP -iTCP:19530 -sTCP:LISTEN
lsof -nP -iTCP:5432 -sTCP:LISTEN
lsof -nP -iTCP:6379 -sTCP:LISTEN
```

本地端口约定：

| 服务 | 端口 | 说明 |
| --- | ---: | --- |
| Milvus | `19530` | 向量索引 |
| Milvus Web/metrics | `9091` | 已被 Milvus 占用，不要给 rerank 用 |
| Postgres | `5432` | metadata |
| Redis | `6379` | cache |
| Embedding service | `9090` | MLX embedding 常驻服务 |
| Rerank service | `9092` | BGE rerank 常驻服务 |

启动 embedding 服务：

```bash
uv run rag embedding-service \
  --model mlx-community/Qwen3-Embedding-4B-4bit-DWQ \
  --port 9090
```

启动 rerank 服务：

```bash
uv run rag rerank-service \
  --model BAAI/bge-reranker-v2-m3 \
  --port 9092
```

健康检查：

```bash
curl -sS http://127.0.0.1:9090/health
curl -sS http://127.0.0.1:9092/health
```

## 最小 RAG 流程

让 CLI 复用常驻模型服务：

```bash
export RAG_EMBEDDING_SERVICE_URL=http://127.0.0.1:9090
export RAG_RERANK_SERVICE_URL=http://127.0.0.1:9092
```

写入一条文本：

```bash
uv run rag ingest \
  --storage-root data/smoke_milvus \
  --source-type plain_text \
  --location memory://smoke/travel-policy \
  --title "差旅制度 Smoke" \
  --owner smoke \
  --content "单笔国内差旅报销金额超过 12000 元时，必须由业务线 VP 审批。"
```

查询：

```bash
uv run rag query \
  --storage-root data/smoke_milvus \
  --retrieval-profile auto \
  --query "单笔国内差旅报销金额超过 12000 元需要谁审批？"
```

入库 Excel：

```bash
uv run rag ingest \
  --storage-root data/smoke_milvus \
  --source-type xlsx \
  --location /absolute/path/to/开票量明细.xlsx \
  --title "开票量明细" \
  --owner smoke
```

查询表格：

```bash
uv run rag query \
  --storage-root data/smoke_milvus \
  --retrieval-profile asset \
  --query "请计算华东区域 Q1 的开票量合计是多少？"
```

需要完整结构化输出时加 `--json`。

## Agent 运行

单次运行：

```bash
uv run rag agent run "帮我查一下差旅报销超过 12000 元的审批规则" \
  --storage-root data/smoke_milvus \
  --agent research \
  --verbose
```

交互式运行：

```bash
uv run rag agent chat \
  --storage-root data/smoke_milvus \
  --agent research
```

带 checkpoint 的暂停/恢复：

```bash
uv run rag agent run "需要审批的复杂任务" \
  --storage-root data/smoke_milvus \
  --agent orchestrator \
  --checkpoint-db .rag/agent_checkpoints.sqlite

uv run rag agent resume <run_id> \
  --storage-root data/smoke_milvus \
  --agent orchestrator \
  --checkpoint-db .rag/agent_checkpoints.sqlite \
  --decision allow_once
```

Agent CLI 默认不会自动猜 agent 类型，`--agent` 要显式选择。

## 正式 Postgres + Milvus 运行方式

CLI 适合 smoke test。需要完整链路时，用 Python 显式构造 `StorageConfig`：

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

runtime_config = resolve_runtime_config()
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

with RAGRuntime.from_request(
    storage=storage,
    request=AssemblyRequest(
        requirements=CapabilityRequirements(require_chat=True, require_rerank=True, allow_degraded=False),
        overrides=to_assembly_overrides(runtime_config),
    ),
) as runtime:
    runtime.insert(
        IngestRequest(
            location="memory://manual/travel-policy",
            source_type=SourceType.PLAIN_TEXT,
            owner="manual",
            title="差旅制度",
            content_text="单笔国内差旅报销金额超过 12000 元时，必须由业务线 VP 审批。",
        )
    )

    result = runtime.query_public(
        "单笔国内差旅报销金额超过 12000 元需要谁审批？",
        options=QueryOptions(retrieval_profile="auto", top_k=8, retrieval_pool_k=20),
    )
    print(result.answer.answer_text)
```

## 公开 benchmark 和私有评测

公开 benchmark 入口：

```bash
uv run rag benchmark-download --dataset medical_retrieval
uv run rag benchmark-prepare --dataset medical_retrieval --variant mini
./scripts/run_benchmark_ingest.sh medical_retrieval mini
uv run rag benchmark-evaluate --dataset medical_retrieval --variant mini --top-k 10
```

私有文档入库：

```bash
uv run python scripts/ingest_private_documents.py \
  --input /absolute/path/to/private_docs \
  --storage-root data/private_index_milvus \
  --vector-backend milvus \
  --vector-dsn http://127.0.0.1:19530 \
  --vector-collection-prefix private_docs_v1 \
  --summary-provider none \
  --batch-size 8
```

私有 golden set 检索评测：

```bash
uv run python scripts/evaluate_private_retrieval.py \
  --golden-path data/eval_private/golden_eval_dataset.jsonl \
  --storage-root data/private_index_milvus \
  --retrieval-profile auto \
  --top-k 10 \
  --retrieval-pool-k 20 \
  --rerank \
  --vector-backend milvus \
  --vector-dsn http://127.0.0.1:19530 \
  --vector-collection-prefix private_docs_v1 \
  --output data/eval_private/private_retrieval_eval.json
```

## 历史基线和对比口径

这些结果是之前已经跑过的实验基线，用来和后续模型、索引、检索策略的改进做横向对比。
它们是历史快照，不代表当前 `configs/models.yaml` 的默认配置；切换 embedding 模型后必须重建对应索引。

### 公开数据：MedicalRetrieval mini

| 基线 | 向量后端 | embedding | rerank | Recall@10 | MRR@10 | NDCG@10 | avg_latency_ms |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| `BAAI/bge-m3 + sqlite` | SQLite | `BAAI/bge-m3` | on | 0.776667 | 0.690972 | 0.712199 | 2472.225 |
| `BAAI/bge-m3 + milvus` | Milvus | `BAAI/bge-m3` | on | 0.670000 | 0.588259 | 0.608173 | 563.793 |
| `qwen3-embedding:8b + milvus` | Milvus | `qwen3-embedding:8b` | on | 0.820000 | 0.705854 | 0.733644 | 695.559 |

当时的 `qwen3-embedding:8b + milvus` 复现口径：

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

### 私有数据：公司制度 / 销售中心资料

黄金测试集规模：

| 题型 | 数量 |
| --- | ---: |
| `single_section_fact` | 163 |
| `rule_condition` | 105 |
| `process_step` | 61 |
| 合计 | 329 |

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

从这组基线能看出的改进点：

- 文档级召回已经比较稳，`doc_hit@10` 到 `0.9818`。
- section 级 top10 命中到 `0.9027`，后续优化主要看 section top1 和 MRR。
- rerank 对 section 排序有效，`section_mrr` 从 `0.7200` 提升到 `0.7588`。
- 单纯把 `top_k` 从 10 放到 20 没带来额外收益，说明问题不在最后截断，而在候选生成和排序。

旧私有文档实验配置：

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

### 最近一次真实端到端验证

这组不是 benchmark 分数，而是验证正式链路能跑通：

| 项 | 值 |
| --- | --- |
| Postgres schema | `rag_e2e_20260516_150131` |
| Milvus collection prefix | `rag_e2e_20260516_150131` |
| 表格对象 | `data/e2e_agent_pq_milvus/20260516_150131/objects/*.parquet` |
| SQLite vector index | 未使用 |

验证问题：

| 类型 | 问题 | 结果 |
| --- | --- | --- |
| RAG 制度问答 | 单笔国内差旅报销金额超过 12000 元需要谁审批？ | 命中制度原文，回答为业务线 VP |
| RAG SLA 问答 | P0 客户生产故障的首次响应目标和恢复目标分别是多少？ | 命中 `15 分钟` 和 `2 小时` |
| 表格计算 | 请计算华东区域 Q1 的开票量合计是多少？ | 触发 DuckDB SQL，返回 `375` |

表格计算当时执行的 SQL：

```sql
SELECT SUM("开票量") FROM sheet WHERE "区域"='华东' AND "季度"='Q1'
```

返回：

```text
TABLE_COMPUTE_RESULT
sum("开票量") = 375
```

## 测试

常用测试：

```bash
uv run pytest -q
```

只测模型配置和 CLI wiring：

```bash
uv run pytest \
  tests/core/test_cli_runtime_model_loading.py \
  tests/ui/test_cli.py \
  tests/agent/test_llm_registry.py
```

只测表格计算、grounding 和 Postgres metadata：

```bash
uv run pytest \
  tests/core/test_table_compute_integration.py \
  tests/service/test_grounding_service.py \
  tests/repo/test_postgres_metadata_repo.py
```

只测复杂 RAG / Agent 回路：

```bash
uv run pytest \
  tests/agent/test_complex_agent_rag_loop.py \
  tests/service/test_complex_rag_retrieval.py
```

当前测试目录按领域拆分：

| 目录 | 覆盖重点 |
| --- | --- |
| `tests/agent` | Agent contract、graph、tool、approval、subagent、memory |
| `tests/core` | runtime、ingest、model config、Milvus contract、table compute |
| `tests/service` | retrieval、grounding、synthesis、authorization |
| `tests/repo` | storage repo 和 provider repo |
| `tests/provider` | HTTP embedding / rerank client |
| `tests/ui` | CLI |

## 从哪些文件开始读

如果只想快速理解系统，不需要从所有文件开始。

| 想理解 | 先看 |
| --- | --- |
| Runtime 怎么装起来 | `rag/runtime.py` |
| CLI 怎么调 runtime | `rag/cli.py` |
| 模型配置怎么解析 | `configs/models.yaml`、`rag/models/runtime.py`、`rag/models/assembly_adapter.py` |
| 入库主流程 | `rag/ingest/pipeline.py` |
| parser 分发 | `rag/ingest/parsers/dispatcher.py` |
| section 切分 | `rag/ingest/section_refiner.py` |
| summary 生成 | `rag/ingest/retrievalsummarizer.py` |
| 表格计算 | `rag/ingest/table_executor.py` |
| storage 装配 | `rag/storage/__init__.py` |
| Milvus 写入和查询 | `rag/storage/search_backends/milvus_vector_repo.py` |
| 查询总流程 | `rag/query_pipeline.py` |
| L3/L4 检索 | `rag/retrieval/l3_l4_engine.py`、`rag/retrieval/planning_graph.py` |
| grounding | `rag/retrieval/grounding_service.py` |
| 答案生成和引用 | `rag/providers/generation.py`、`rag/providers/citation_formatter.py` |
| Agent 定义 | `rag/agent/core/definition.py` |
| Agent 图 | `rag/agent/graphs/base.py` |
| 工具契约 | `rag/agent/tools/spec.py`、`rag/agent/tools/rag_tools.py` |
| Agent 对外服务 | `rag/agent/service.py` |

## 当前边界

这些点是项目现在明确保留的边界：

- 回答事实以 RAG evidence 为准，working memory 只能做上下文线索。
- summary index 是检索入口，不是事实来源。
- 表格问题必须通过受限计算链路，不能拿 sample rows 当全量数据回答。
- 切换 embedding 模型必须重建索引。
- 高风险工具需要权限和审批，不应该静默执行副作用。
- tool、config、schema、runtime contract 都要有显式类型边界。
- 失败要变成结构化错误、诊断信息或测试失败，不能吞异常后继续装作成功。

仍在推进的部分：

- 长期 memory 写入、去重、冲突治理。
- 更完整的 graph backend。
- 更多生产级 Agent prompt 和 golden case 评测。
- 外部副作用工具的审批、审计和恢复流程。
- `analyze-task` 旧入口的后续替换。

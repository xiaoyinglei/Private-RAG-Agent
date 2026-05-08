# RAG 主线运行手册

这个仓库当前主线已经切到新的企业知识 RAG 架构。系统目标不是做一个简单 Demo，而是把企业私有文档从解析、入库、检索、精读、引用回答到评测闭环全部跑通。

```text
原始文件 -> Parser -> Document / SectionRecord / AssetRecord
        -> SectionRefiner token 窗口
        -> Doc / Section / Asset 三类摘要
        -> Embedding
        -> Milvus 三类 summary index
        -> L3 planning
        -> L4 retrieval / rerank
        -> L5 grounding / anchor replacement
        -> L6 synthesis
```

主线契约只认：

- `Document`
- `SectionRecord`
- `AssetRecord`
- `SummaryRecord`
- `GroundingTarget`
- `EvidenceItem`

旧 `Chunk / Segment / mode mix/local/global` 不是主线。`rag/agent/**` 也不参与当前重构主线。

## 架构总览

当前系统按 6 层主链路组织。

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
```

### L1：事实层

L1 保存事实，不负责相似度检索。这里的事实包括：

- `Document`：一份业务文档的版本、权限、状态、来源。
- `SectionRecord`：可检索和可精读的正文窗口，带 `raw_locator`、byte range、token 窗口元数据。
- `AssetRecord`：表格、图片、OCR 区域、PPT 表格等非正文资产，带 `section_id` 绑定关系。
- Object Store：保存原始文件、visible text、表格对象、schema/sample 与 DuckDB 可读存储指针。

### L2：轻索引层

L2 只保存检索入口，不保存事实主数据。Milvus 默认拆成三类 summary index：

- `doc_summary`：文档级主题召回。
- `section_summary`：默认主召回层，处理制度条款、事实问答、流程问答。
- `asset_summary`：表格、图片、OCR、PPT 资产的语义入口。

Milvus 里只放 summary、向量、标量过滤字段和主键映射，不把大段原文当文本库。

### L3/L4：规划与检索

L3 负责判断“该怎么查”，L4 负责“从哪里召回”。主线不再使用旧 `--mode mix/local/global`，而使用新的 `retrieval_profile`：

- `fast`
- `auto`
- `deep`
- `asset`

规划层会先处理复杂度、语义路由、版本过滤和谓词下推；检索层再对 doc/section/asset summary 做多路召回、候选清洗和 rerank。

### L5：精读与证据层

L5 不相信 summary 本身就是答案。summary 只负责定位，最终回答必须回到原文或资产对象：

- 命中 section 后，通过 `visible_text_key + byte_range` 回读正文。
- 命中含表格锚点的 section 后，按 `[ASSET_ANCHOR:...]` 找回绑定资产。
- 表格一律进入 DuckDB Text-to-SQL Sandbox；schema/sample 只用于生成 SQL 与解释结构，不把表格 Markdown 塞进 prompt。
- 精读阶段受 token、目标数、并发、超时预算控制。

### L6：合成层

L6 只基于 `EvidenceItem` 回答。回答必须能追溯到 `doc_id / section_id / asset_id`，后续 Policy Guard 会作为最终权限与合规闸门。

## 核心设计理念

### 1. Summary-First, Grounding-Later

先用 summary 做轻量召回，再回原文精读。summary 不是事实主库，也不是最终答案来源。

### 2. Facts in Storage, Search in Index

PostgreSQL / Object Store 是事实层，Milvus 是索引层。事实和检索必须分离：

- 原文、表格、定位、权限、版本在 L1。
- 向量、BM25、标量过滤在 L2。

### 3. 主链路只认新契约

主链路不再混用旧 `Chunk / Segment`。新系统以 `Document / SectionRecord / AssetRecord / SummaryRecord / GroundingTarget / EvidenceItem` 为边界，复杂度判断只作为检索规划内部 gate，不进入公共 schema。

旧代码如果不影响主线，只保留到删除；不再为旧契约补测试。

### 4. Token-First

所有切分、窗口、预算、摘要输入输出都按 token 控制，不按字符控制。

当前原则：

- SectionRefiner 按 token 滑动窗口。
- eval 出题按 token 二次窗口。
- 摘要输入输出按 token 裁剪。
- L5 grounding budget 也按 token 记账。

### 5. 资产不混进正文，锚点绑定

表格、图片、OCR、PPT 表格都是 `AssetRecord`，不要把二维结构强行压进普通正文。

正文只保留：

```text
[ASSET_ANCHOR:xxx]
```

资产实际内容独立保存。SectionRefiner 切分后，通过锚点追踪把资产绑定到具体细粒度 section，避免“字到了，表没到”。

### 6. 表格统一走 DuckDB Text-to-SQL Sandbox

表格是最容易诱发幻觉的资产类型，当前 RAG 主线不再按大小分成“短表可直接回填 / 中表摘要 / 长表计算”。官方标准只有一个：

- 所有表格资产统一标记为 `table_policy=compute_only`。
- 严禁 `inline_context`，包括短表。短表直接回填 Markdown 会诱导模型目测、排序、聚合和比较，必须废除。
- `summary_only` 只作为检索摘要语义，不作为表格处理策略。
- 表格只暴露 `schema / sample_rows / row_count / column_count / storage_key` 等结构信息。
- 需要数据值、过滤、排序、聚合、排名、交叉对比时，由 LLM 生成受限 `SELECT`，交给 DuckDB Text-to-SQL Sandbox 执行，再把计算结果交给 L6 合成。

### 7. 摘要必须高密度、结构化、可检索

Section / Asset / Doc 三类摘要都必须稳定模板化。摘要要保留：

- 结构位置
- 语义核心
- 事实锚点
- 数字、金额、日期、制度文号、部门、表头字段、枚举值

这不是为了好看，而是为了让 Milvus hybrid / BM25 / rerank 更容易命中关键事实。

### 8. 默认 Milvus，不再默认 SQLite

SQLite 只作为本地轻量 fallback 或历史参考。当前默认主线是 Milvus，因为设计目标是百万级企业知识库。

### 9. Rerank 是排序增强，不是召回补药

Rerank 只能重排候选，不能解决候选池里没有 gold 的问题。评测时必须同时看：

- `doc_hit`
- `section_hit`
- `parent_section_hit`
- `neighbor_section_hit`
- `returned_candidate_count`
- miss category

### 10. DuckDB Text-to-SQL Sandbox 是表格计算边界

Excel / Word 表格 / PPT 表格 / 业务流水 / 聚合统计不能让 LLM 心算，也不能把 Markdown 表格塞进上下文。正确边界是：

```text
检索命中表格资产 -> 读取 schema/sample -> LLM 生成 SELECT -> DuckDB Sandbox 执行 -> 返回计算结果 -> L6 合成
```

MCP/Pandas 不属于当前 RAG 主线，统一移到 Phase 4 的高级数据分析功能。当前主线只认 DuckDB Text-to-SQL Sandbox。

## 数据处理原则

### 原始文件到 Section

解析阶段优先保留文档天然结构：

- Word/PDF/Markdown：按 Docling 结构树和标题自然分段。
- Excel：每个 sheet 作为带锚点的 section，表格本体作为 asset。
- PPTX：每页 slide 作为 section，文本、表格、备注作为元素。
- 图片：OCR visible text 作为 section，OCR region 作为元素。

当初始 section 超过 token 上限时，`SectionRefiner` 再做 token 滑动窗口。

### 原始文件到 Asset

资产抽离优先于文本拼接：

- 表格不进入正文，不按短表/长表分流，不允许 `inline_context`。
- 表格资产保存 schema/sample/形状信息/可计算存储指针，统一以 `compute_only` 进入 DuckDB Text-to-SQL Sandbox。
- 图片/OCR 保存区域文本与坐标。
- 后续 L5 根据 `section_id` 和锚点关系补齐资产。

### 入库到 Milvus

入 Milvus 的不是原文，而是三类摘要：

- `DocSummaryRecord`
- `SectionSummaryRecord`
- `AssetSummaryRecord`

入库完成后，查询只读 `index_ready` 的对象，避免半成品数据进入检索。

## 当前工程状态

### 已落地

- 默认向量后端：Milvus。
- 默认摘要模型：`Qwen/Qwen3-8B-MLX-4bit`，聊天模型不再默认兼任摘要模型。
- 私有文档入库支持：`.pdf / .docx / .md / .markdown / .xlsx / .xls / .pptx / .png / .jpg / .jpeg / .txt`。
- Word/PDF/Markdown 走 Docling 解析。
- Excel 走原生 Pandas/OpenPyXL 解析。
- PPTX 走原生 `python-pptx` 解析。
- 图片走 OCR repo。
- 表格资产已走 `[ASSET_ANCHOR:...]`，正文只保留锚点，表格 schema/sample/可计算存储指针作为 `AssetRecord`。
- `inline_context` 已废除；短表也不能在 L5 回填 Markdown。
- 所有表格统一 `compute_only`，不把全量 Markdown 入库，`AssetRecord.storage_key` 指向可由 DuckDB 读取的表格对象。
- `AssetRecord` 已增强：`sheet_name / row_count / column_count / sample_rows / schema`。
- `table_policy=compute_only` 或聚合计算意图会进入 DuckDB Text-to-SQL Sandbox。

### 移出当前主线

- MCP/Pandas worker 不再作为当前 RAG 主线目标。
- MCP/Pandas 归档为 Phase 4 的高级数据分析能力，用于未来多表联动、复杂 Python 分析、可视化与长任务编排。

## 当前评测记录

### 公开数据：MedicalRetrieval mini

下面 3 条是从 `main` 分支历史 README 保留下来的 retrieval 基线，作为重构后对齐公开数据评测的基准线。

| 基线 | 向量后端 | embedding | rerank | Recall@10 | MRR@10 | NDCG@10 | avg_latency_ms | 定位 |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| `BAAI/bge-m3 + sqlite` | SQLite | `BAAI/bge-m3` | on | 0.776667 | 0.690972 | 0.712199 | 2472.225 | 历史参考线 |
| `BAAI/bge-m3 + milvus` | Milvus | `BAAI/bge-m3` | on | 0.670000 | 0.588259 | 0.608173 | 563.793 | 低时延线 |
| `qwen3-embedding:8b + milvus` | Milvus | `qwen3-embedding:8b` | on | 0.820000 | 0.705854 | 0.733644 | 695.559 | 质量优先线 |

当前主线选择：

- 质量优先线：`qwen3-embedding:8b + milvus`
- 低时延线：`BAAI/bge-m3 + milvus`
- `BAAI/bge-m3 + sqlite` 只保留为历史参考，不再作为当前默认主线

### 私有数据：公司制度/销售中心资料

私有数据结果不提交原始数据和逐条评测明细，只在 README 记录汇总分数。

黄金测试集：

- golden_path：`data/eval_private/golden_eval_dataset.jsonl`
- query_count：`329`
- 题型分布：
  - `single_section_fact`：163
  - `rule_condition`：105
  - `process_step`：61

评测配置：

- storage_root：`data/company_policy_index_recut`
- retrieval_profile：`auto`
- vector_backend：`sqlite`

整体结果：

| 配置 | top_k | rerank | doc_mrr | section_mrr | doc_hit@1 | doc_hit@3 | doc_hit@5 | doc_hit@10 | section_hit@1 | section_hit@3 | section_hit@5 | section_hit@10 |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| sqlite rerank | 10 | on | 0.9300 | 0.7588 | 0.8967 | 0.9574 | 0.9757 | 0.9818 | 0.6535 | 0.8602 | 0.8875 | 0.9027 |
| sqlite no rerank | 10 | off | 0.9127 | 0.7200 | 0.8663 | 0.9574 | 0.9757 | 0.9818 | 0.6049 | 0.8359 | 0.8723 | 0.9027 |
| sqlite no rerank | 20 | off | 0.9127 | 0.7200 | 0.8663 | 0.9574 | 0.9757 | 0.9818 | 0.6049 | 0.8359 | 0.8723 | 0.9027 |

top_k=20 对比：

| 配置 | doc_hit@10 | doc_hit@20 | section_hit@10 | section_hit@20 |
| --- | ---: | ---: | ---: | ---: |
| sqlite no rerank | 0.9818 | 0.9818 | 0.9027 | 0.9027 |

按题型结果，开启 rerank：

| 题型 | query_count | doc_mrr | section_mrr | doc_hit@1 | doc_hit@3 | doc_hit@10 | section_hit@1 | section_hit@3 | section_hit@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `process_step` | 61 | 0.9727 | 0.7865 | 0.9508 | 1.0000 | 1.0000 | 0.7049 | 0.8689 | 0.9344 |
| `rule_condition` | 105 | 0.9221 | 0.7929 | 0.8857 | 0.9524 | 0.9810 | 0.6762 | 0.8952 | 0.9429 |
| `single_section_fact` | 163 | 0.9192 | 0.7265 | 0.8834 | 0.9448 | 0.9755 | 0.6196 | 0.8344 | 0.8650 |

按题型结果，不开启 rerank：

| 题型 | query_count | doc_mrr | section_mrr | doc_hit@1 | doc_hit@3 | doc_hit@10 | section_hit@1 | section_hit@3 | section_hit@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `process_step` | 61 | 0.9440 | 0.7461 | 0.9016 | 0.9836 | 1.0000 | 0.6393 | 0.8525 | 0.9344 |
| `rule_condition` | 105 | 0.9032 | 0.7426 | 0.8476 | 0.9619 | 0.9810 | 0.6000 | 0.8857 | 0.9429 |
| `single_section_fact` | 163 | 0.9071 | 0.6957 | 0.8650 | 0.9448 | 0.9755 | 0.5951 | 0.7975 | 0.8650 |

消融结论：

- reranker 主要提升 section 级排序质量，不提升 top10 召回。
- `section_mrr`：0.7200 -> 0.7588，提升 +0.0388。
- `section_hit@1`：0.6049 -> 0.6535，提升 +0.0486。
- `section_hit@3`：0.8359 -> 0.8602，提升 +0.0243。
- `section_hit@5`：0.8723 -> 0.8875，提升 +0.0152。
- `section_hit@10`：0.9027 -> 0.9027，无变化。
- top_k 从 10 扩展到 20 未带来新增召回；未命中的正确 section 大概率没有进入 top20。

可对外表述：

> 基于 329 条私有制度文档黄金评测集，对 RAG 检索链路进行离线评估，覆盖 single_section_fact、rule_condition、process_step 三类问题。在开启 rerank 的配置下，系统达到 doc_hit@1=89.7%、doc_hit@10=98.2%、section_hit@1=65.3%、section_hit@3=86.0%、section_hit@10=90.3%、section_mrr=0.759。对比实验显示，reranker 主要提升 section 级排序能力，section_hit@1 从 60.5% 提升至 65.3%，section_mrr 从 0.720 提升至 0.759。

一句话总结：文档级检索很强，section 级召回较好，top1 排序还有优化空间；rerank 有效，但主要改善排序，不改善 top10 召回。

## 目录地图

```text
rag/
├── runtime.py                         # 唯一组合根：装配 storage / ingest / retrieval / synthesis
├── cli.py                             # 新 CLI：ingest / query / benchmark-download / benchmark-ingest / benchmark-evaluate
├── benchmarks.py                      # benchmark 和 runtime build helper
├── assembly/                          # provider/profile/tokenizer 装配
├── ingest/
│   ├── pipeline.py                    # L1/L2 入库主线
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
    ├── data_contract_service.py       # 新数据契约服务
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

如果你已经有本地 Milvus，只需要确认地址：

```bash
export MILVUS_URI=http://127.0.0.1:19530
export RAG_MILVUS_URI=$MILVUS_URI
```

### 3. 准备 embedding 模型

当前默认建议：

```bash
ollama pull qwen3-embedding:8b
```

如果 Ollama 没启动，另开一个终端：

```bash
ollama serve
```

### 4. 准备摘要模型

入库摘要默认走本地 MLX：

```bash
export SUMMARY_MODEL=Qwen/Qwen3-8B-MLX-4bit
```

入库脚本用 `--summary-provider local-hf --summary-backend mlx` 时会在当前进程加载这个模型，不需要 OpenAI-compatible server。

生成测试题脚本 `generate_eval_dataset.py` 使用 OpenAI-compatible 接口，所以需要另开一个终端启动 MLX server：

```bash
uv run mlx_lm.server \
  --model Qwen/Qwen3-8B-MLX-4bit \
  --host 127.0.0.1 \
  --port 8080 \
  --max-tokens 1024 \
  --temp 0.1 \
  --chat-template-args '{"enable_thinking":false}'
```

## 私有数据全链路命令

下面这组命令用于你的公司制度/Word/Excel/PDF/PPTX 私有数据。

先设变量：

```bash
export INPUT_DIR="/Users/leixiaoying/Desktop/2026-04-27销售中心归口管理的制度及文件"
export STORAGE_ROOT=data/eval_private/company_policy_milvus_v4
export COLLECTION_PREFIX=company_policy_v4
export MILVUS_URI=http://127.0.0.1:19530
export EMBEDDING_MODEL=qwen3-embedding:8b
export SUMMARY_MODEL=Qwen/Qwen3-8B-MLX-4bit
export CHUNK_TOKEN_SIZE=800
export CHUNK_OVERLAP_TOKENS=120
```

### 1. 重新切分 + 重新入库 + 生成摘要 + 写 Milvus

重切没有单独脚本。当前主线里，重切就是重新跑 ingest。

推荐每次新实验换新的 `STORAGE_ROOT` 和 `COLLECTION_PREFIX`，避免 Milvus 旧 collection 混入。

```bash
uv run python scripts/ingest_private_documents.py \
  --input "$INPUT_DIR" \
  --storage-root "$STORAGE_ROOT" \
  --profile local_full \
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

如果你必须复用同一个 `COLLECTION_PREFIX`，先清 Milvus 旧 collection：

```bash
uv run python - <<'PY'
from pymilvus import connections, utility

uri = "http://127.0.0.1:19530"
prefix = "company_policy_v4"
alias = "cleanup"

connections.connect(alias=alias, uri=uri)
for name in list(utility.list_collections(using=alias)):
    if name.startswith(prefix + "__"):
        utility.drop_collection(name, using=alias)
        print("dropped", name)
connections.disconnect(alias)
PY
```

### 2. 导出 SectionRecord JSONL

这个文件用于逆向出题和人工检查切分结果。

```bash
uv run python scripts/export_private_sections.py \
  --storage-root "$STORAGE_ROOT" \
  --output data/eval_private/company_policy_sections_v4.jsonl
```

快速看数量：

```bash
wc -l data/eval_private/company_policy_sections_v4.jsonl
```

### 3. 生成 golden eval 测试集

先做 smoke：

```bash
uv run python generate_eval_dataset.py \
  --input data/eval_private/company_policy_sections_v4.jsonl \
  --output data/eval_private/golden_eval_dataset_v4_smoke.jsonl \
  --failed-output data/eval_private/golden_eval_failed_v4_smoke.jsonl \
  --model "$SUMMARY_MODEL" \
  --base-url http://127.0.0.1:8080/v1 \
  --api-key not-needed \
  --max-window-tokens 700 \
  --window-overlap-tokens 80 \
  --min-window-tokens 120 \
  --limit-windows 3 \
  --limit-tasks 9
```

确认没问题后全量生成：

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

### 4. 检索评测，不开 rerank

注意：`--chunk-token-size / --chunk-overlap-tokens` 必须和入库一致，否则 runtime contract 会拒绝运行。

```bash
uv run python scripts/evaluate_private_retrieval.py \
  --golden-path data/eval_private/golden_eval_dataset_v4.jsonl \
  --storage-root "$STORAGE_ROOT" \
  --profile local_full \
  --retrieval-profile auto \
  --top-k 20 \
  --retrieval-pool-k 20 \
  --neighbor-radius 1 \
  --no-rerank \
  --embedding-provider ollama \
  --embedding-model "$EMBEDDING_MODEL" \
  --vector-backend milvus \
  --vector-dsn "$MILVUS_URI" \
  --vector-collection-prefix "$COLLECTION_PREFIX" \
  --chunk-token-size "$CHUNK_TOKEN_SIZE" \
  --chunk-overlap-tokens "$CHUNK_OVERLAP_TOKENS" \
  --output data/eval_private/private_retrieval_eval_v4_no_rerank.json \
  --misses-output data/eval_private/private_retrieval_misses_v4_no_rerank.jsonl
```

### 5. 检索评测，开启 rerank

```bash
uv run python scripts/evaluate_private_retrieval.py \
  --golden-path data/eval_private/golden_eval_dataset_v4.jsonl \
  --storage-root "$STORAGE_ROOT" \
  --profile local_full \
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

### 6. 单条问题检索调试

如果使用自定义 `COLLECTION_PREFIX`，用 Python snippet 最稳：

```bash
uv run python - <<'PY'
from pathlib import Path

from rag.benchmarks import build_runtime_for_benchmark
from rag.retrieval.models import QueryOptions
from rag.schema.runtime import AccessPolicy

runtime = build_runtime_for_benchmark(
    storage_root=Path("data/eval_private/company_policy_milvus_v4"),
    profile_id="local_full",
    require_chat=False,
    require_rerank=True,
    embedding_provider_kind="ollama",
    embedding_model="qwen3-embedding:8b",
    rerank_provider_kind="local-bge",
    rerank_model="BAAI/bge-reranker-v2-m3",
    vector_backend="milvus",
    vector_dsn="http://127.0.0.1:19530",
    vector_collection_prefix="company_policy_v4",
    chunk_token_size=800,
    chunk_overlap_tokens=120,
)

try:
    payload = runtime.retrieval_service.retrieve_payload(
        "这里换成你的问题",
        access_policy=AccessPolicy.default(),
        query_options=QueryOptions(
            retrieval_profile="auto",
            top_k=10,
            evidence_top_k=10,
            max_evidence_items=10,
            retrieval_pool_k=20,
            rerank_pool_k=20,
            enable_rerank=True,
        ),
    )
    for index, item in enumerate(payload.clean_items or payload.evidence.all, start=1):
        print(index, item.item_id, item.record_type, item.score, item.metadata)
finally:
    runtime.close()
PY
```

## 公开 benchmark 命令

### 1. 下载与准备 MedicalRetrieval mini

```bash
uv run python scripts/download_public_benchmark.py --dataset medical_retrieval
uv run python scripts/prepare_public_benchmark.py --dataset medical_retrieval
```

### 2. 入库：Milvus + qwen3-embedding:8b

```bash
uv run python scripts/ingest_public_benchmark.py \
  --dataset medical_retrieval \
  --variant mini \
  --profile local_full \
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
```

### 3. 评测：不开 rerank

```bash
uv run rag benchmark-evaluate \
  --dataset medical_retrieval \
  --variant mini \
  --profile local_full \
  --storage-root data/benchmarks/medical_retrieval/index/mini-milvus-qwen8b-v1 \
  --vector-backend milvus \
  --vector-dsn "$MILVUS_URI" \
  --vector-collection-prefix medical_retrieval_mini_qwen8b_v1 \
  --retrieval-profile auto \
  --top-k 10 \
  --evidence-top-k 20 \
  --no-rerank \
  --embedding-provider ollama \
  --embedding-model qwen3-embedding:8b
```

### 4. 评测：开启 rerank

```bash
uv run rag benchmark-evaluate \
  --dataset medical_retrieval \
  --variant mini \
  --profile local_full \
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

## 快速代码质量检查

全量：

```bash
uv run ruff check rag scripts tests
uv run pytest -q
```

本次 Excel/表格资产主线重点回归：

```bash
uv run pytest -q \
  tests/core/test_excel_parser_repo.py \
  tests/core/test_ingest_asset_anchors.py \
  tests/core/test_retrieval_summarizer.py \
  tests/service/test_grounding_service.py
```

## 关键注意事项

- 以后默认用 Milvus，不再默认 SQLite。
- 入库与检索的 embedding、tokenizer、`chunk_token_size`、`chunk_overlap_tokens` 必须一致。
- 私有数据每次重跑建议换新的 `COLLECTION_PREFIX`。
- 表格统一只入 schema/sample/summary/可计算存储指针，不入全量 Markdown。
- `inline_context` 已废除；短表也只走 `compute_only`，统一由 DuckDB Text-to-SQL Sandbox 计算。
- MCP/Pandas 不在当前 RAG 主线内，归入 Phase 4 高级数据分析功能。
- 生成测试题前必须先 `export_private_sections.py`。
- `generate_eval_dataset.py` 的失败文件只有在真正失败时才会生成。

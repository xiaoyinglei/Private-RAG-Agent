# 《100 万级企业知识 Agent 系统总体架构与存储检索蓝图（最终整合版）》

## 1. 文档目的

本文档用于定义一套面向 **100 万级企业私有文档** 的知识 Agent 系统总体方案。

系统目标不是构建一个只能做摘要问答的 RAG Demo，而是构建一个能够在企业场景下实现以下能力的工业级知识引擎：

- 私有知识安全管理
- 多粒度高效检索
- 原文级证据取证
- 带引用的稳定回答
- 版本治理与权限控制
- 可观测、可扩展、可落地运行

本文档整合了：

- 总体架构设计
- 存储与索引设计
- L3 规划层与 L4 检索层最终方案
- 性能优化、预算控制、实施建议与避坑清单

可作为后续：

- 数据模型设计
- 索引模型设计
- 查询运行时链路设计
- 工程模块拆分
- POC 与正式版实施

的统一基线。

---

## 2. 核心设计哲学

### 2.1 Summary-First, Grounding-Later

系统不对全量文档进行细粒度预切块并全量入向量库，而采用：

- 轻索引先定位
- 原文再精读
- 最终证据回答

的两阶段架构。

### 2.2 Facts in Storage, Search in Index

系统明确区分：

- **事实层**：PostgreSQL + S3 / Object Store
- **检索层**：Milvus

其中：

- PostgreSQL / S3 负责存储真实数据、元数据、原文与定位信息
- Milvus 只负责高效检索，不作为事实主库

### 2.3 Retrieval Is Not Answering

检索层的职责是快速找到候选对象，而不是直接产出最终答案。

最终回答必须建立在 **原文级 grounding** 基础上。

### 2.4 Governance Is First-Class

版本治理、权限校验、预算熔断、审计留痕不是附加功能，而是系统主链路的一部分。

### 2.5 轻重分离、分层检索

不要把百万文档全量切块后全部塞进向量库，而是采用：

- **轻索引（Milvus）**：文档、章节、资产摘要向量，作为检索入口
- **重存储（PostgreSQL / S3）**：原文、元数据、定位信息，作为证据事实源

---

## 3. 适用范围

本架构面向以下企业知识对象：

- PDF / DOCX / Markdown / Excel / 图片 / 扫描件
- 制度文件
- 报表与经营数据说明
- 会议纪要
- 表格、图示、流程图、图片说明
- 版本化内部知识材料

不直接覆盖：

- 通用互联网搜索引擎场景
- 完全开放式网络问答
- 无权限边界的消费者产品

---

## 4. 总体架构概览

系统采用 **6 层主架构 + 1 个横切基础设施层**。

### 4.1 六层主架构

1. L1 数据层（Storage）
2. L2 索引层（Indexing）
3. L3 规划层（Planning）
4. L4 检索层（Retrieval）
5. L5 精读层（Grounding）
6. L6 合成层（Synthesis）

### 4.2 横切基础设施层（Infra）

横跨所有主层，提供：

- CDC
- 异步任务
- 缓存
- 监控
- tracing
- 审计
- 重试与失败恢复
- 限流与熔断

---

## 5. 分层架构设计

## 5.1 L1 数据层（Storage）

### 5.1.1 职责

数据层是系统事实主库，负责保存：

- 原始文档
- 结构化元数据
- 版本信息
- 权限信息
- 摘要文本
- 原文定位信息
- 处理状态信息

### 5.1.2 组成

#### PostgreSQL

存储：

- 文档主记录
- 章节记录
- 资产记录
- 版本链
- 权限标签
- 定位信息
- 摘要文本
- 处理状态表

#### S3 / Object Store

存储：

- 原始 PDF / DOCX / XLSX / 图片
- 解析后的 Markdown / 结构化文本
- 中间转换产物
- Layout JSON

### 5.1.3 设计要求

- PostgreSQL 是业务真相来源
- S3 是原文真相来源
- 所有检索对象都必须能回溯到 PG / S3 中的原始记录

### 5.1.4 数据质量与合规增强

#### 1）全局去重与哈希校验

在 `documents` 表中增加 `file_hash`（MD5 / SHA-256）。

作用：

- 入库前去重
- 避免重复摘要生成与重复向量化
- 降低无效算力消耗

#### 2）结构化解析缓存（Layout Meta-Cache）

在 S3 原文之外，额外保存 Layout JSON，用于记录：

- 文本块位置
- 表格位置
- 图片位置
- OCR 区域坐标

作用：

- 精读阶段避免重复 OCR / 解析
- 支持坐标级渲染与精准提取

#### 3）隐私与脱敏层（PII Masking）

对包含手机号、身份证号等敏感信息的文档：

- 离线路径先做脱敏
- 再进入摘要与向量化流程

防止敏感信息进入向量库。

### 5.1.5 单机流水线阶段增强

#### 状态追踪表（Processing State Table）

建议维护：

- `PENDING`
- `PARSED`
- `SUMMARY_GEN`
- `INDEXED`
- `FAILED`

用于支持：

- 断点续跑
- 崩溃恢复
- 幂等重试

#### 幂等性写入（Idempotent Writes）

建议使用：

- `INSERT ... ON CONFLICT DO UPDATE`

保证解析与入库脚本可重入。

#### 原子提交锚点（Atomic Commit Anchor）

建议流程：

1. 先写 PostgreSQL
2. 再写 Milvus
3. 最后在 PostgreSQL 标记 `is_indexed = true`

查询阶段只读 `is_indexed = true` 的对象，避免半成品数据进入主链路。

---

## 5.2 L2 索引层（Indexing）

### 5.2.1 职责

索引层负责构建系统的轻量检索入口，不保存完整事实，只保存：

- 检索向量
- 检索必需的标量字段
- 多粒度对象映射

### 5.2.2 三层索引对象

系统采用三类摘要索引对象：

#### 1）Doc Summary Index

用于：

- 文档级召回
- 宏观主题定位
- 总结类与综述类问题

#### 2）Section Summary Index

用于：

- 制度条款定位
- 事实型问题
- 细粒度语义召回

**Section Summary 是默认主检索层，也是 90% 查询的首选。**

#### 3）Asset Summary Index

用于：

- 表格
- 图片
- 图示
- OCR 区域
- caption / figure 摘要定位

### 5.2.3 技术选型

使用 **Milvus** 作为索引层底座。

启用能力：

- Dense 向量检索
- Sparse / BM25 检索
- Hybrid Search
- 标量过滤

### 5.2.4 索引边界

Milvus 中仅存储：

- `id`
- `vector`
- 检索必要标量字段

例如：

- `doc_id`
- `section_id`
- `asset_id`
- `version_group_id`
- `version_no`
- `doc_status`
- `effective_date`
- `updated_at`
- `is_active`
- `department_id`
- `auth_tag`
- `embedding_model_id`
- `source_type`

不在 Milvus 中保存大段业务原文，不把 Milvus 当事实主库。

### 5.2.5 索引对象状态模型

所有 Collection 必须包含：

- `version_group_id`
- `version_no`
- `doc_status`
- `effective_date`
- `updated_at`
- `is_active`

用于支撑：

- 版本治理
- 当前生效版本过滤
- 历史版本隔离

### 5.2.6 Multi-Collection 设计

建议采用 **物理集合分离**：

- `doc_summary`
- `section_summary`
- `asset_summary`

理由：

- 文档摘要、章节摘要、资产摘要的语义密度不同
- 检索策略不对称
- 有利于按计划进行非对称召回

### 5.2.7 多粒度位置对齐

索引层必须维护：

- Doc -> Section
- Section -> Asset
- Section -> Neighbor Assets
- Section -> Page Range
- Asset -> Raw Locator

命中某一层对象后，系统应能顺带补齐相关邻近证据。

### 5.2.8 向量空间冷热隔离

建议通过 Partition 对最近 3 个月的活跃文档与历史文档做物理隔离：

- 热分区：高频使用文档
- 冷分区：低频历史文档

优势：

- 降低内存压力
- 提高热数据查询性能

### 5.2.9 Embedding 模型版本标签

在 Schema 中增加：

- `embedding_model_id`

作用：

- 支持模型升级
- 支持灰度检索
- 支持新旧模型并存

### 5.2.10 标量索引建议

建议明确：

- `is_active` 使用 BitMap 索引
- `department_id` 使用 Inverted Index

避免大规模过滤时全表扫描。

### 5.2.11 向量量化与容量规划

在千万级 Section 索引下，建议：

- 使用 `IVF_SQ8` 或 `HNSW`
- 摘要向量统一使用 Cosine Similarity

原因：

- 降低内存开销
- 保持较好精度
- 方便后续 Rerank 分值归一化

### 5.2.12 延迟编码（Lazy Encoding）

对于海量冷门文档：

- 先只进主库
- 当文档变成高优先级或首次被命中时再异步向量化入库

### 5.2.13 批处理与延迟建索引

单机流水线阶段建议：

- 每 500~1000 条向量做一次 bulk insert
- 全量 Row 写入完成后，再统一 `create_index`

避免边写边建索引带来的稳定性问题。

### 5.2.14 Bulk Insert 红线

百万级 / 千万级向量写入严禁逐条调用 API `insert`。
必须使用：

- Milvus BulkInsert

从 S3 导入 Parquet / JSON。

---

## 5.3 L3 规划层（Planning）

### 5.3.1 职责

规划层负责把用户自然语言问题翻译成系统可执行的 **Execution Graph**。

这一层：

- 不直接访问原文
- 不负责生成答案
- 只负责理解问题、拆解问题、选择路径、生成执行计划

### 5.3.2 核心子能力

#### 1）Intent Recognition

识别用户问题属于：

- 制度问答
- 数据查询
- 图表类问题
- 总结类问题
- 对比类问题
- 多跳复合问题

#### 2）Query Decomposition

将复杂问题拆成多个子问题。

例如：

“对比 A 制度和 B 制度的公积金比例”

可拆为：

- 查 A 制度相关章节
- 查 B 制度相关章节
- 提取比例字段
- 做对比总结

#### 3）Operator Plan

生成算子序列，例如：

`VersionGate -> EntitlementFilter -> SectionSearch -> TopK -> RawRead -> Grounding -> Synthesis`

#### 4）Predicate Push-down

执行 **分层谓词下推**：

- **强约束**：权限、部门、租户、地域等，先由 PostgreSQL 生成白名单
- **弱约束**：类型、状态、时间范围等，再交由 Milvus 标量过滤

### 5.3.3 动态谓词下推策略

#### 策略 A（小规模白名单）

若 `doc_id_whitelist < 1000`：

- 直接下推 `doc_id in [...]`

#### 策略 B（大规模白名单）

若白名单过大：

- 自动改为属性级过滤
- 下推 `department_id == 'X'` 或 `auth_tag == 'Y'`

作用：

- 防止长表达式拖垮 Milvus
- 控制表达式复杂度

### 5.3.4 Complexity Gate

规划层引入复杂度栅栏，对查询按复杂度分流：

#### Fast-Track

- 正则匹配
- 短 Query
- 高频模板问题

直接生成检索指令，跳过 LLM 拆解。

#### Standard

- 中等长度语义查询

仅执行 Query Rewrite。

#### Complex

- 长难句
- 对比类
- 多跳复合问题

启动 LangGraph 任务分解。

### 5.3.5 Semantic Router

利用语义路由判断意图支线：

- 文本主线
- 资产支线（图、表、OCR 区域）

### 5.3.6 Version Gate

规划层默认第一步必须插入：

- `Filter(current_active_version)`

从源头阻断旧版本与失效版本误召回。

### 5.3.7 退出与降级协议

规划层需要为复杂场景提供：

- 非线性降级
- 退出协议
- 空召回时的备选策略

避免系统进入高成本无效路径。

---

## 5.4 L4 检索层（Retrieval）

### 5.4.1 职责

检索层是索引层的调度核心，负责：

- 多路召回
- 索引编排
- Hybrid 融合
- Rerank
- 召回补盲
- 退出审计

### 5.4.2 检索路径

#### Dense Path

负责：

- 语义近似
- 模糊表达
- 自然语言相关性召回

建议使用：

- BGE-M3 Embedding

#### Sparse Path

负责：

- 专有名词
- 制度文号
- 数字
- 日期
- 型号
- 表格字段名
- 专业术语

方案：

- 默认：Milvus 原生 BM25
- 升级：BGE-M3 Sparse

#### Hybrid Path

通过 RRF 或加权融合实现统一候选池。

建议：

- 权重不要硬编码
- 在配置中心维护 `alpha`
- 支持上线后动态调参

### 5.4.3 主检索策略

默认优先检索：

- `Section Summary Index`

若 Section 召回不足，再补充：

- `Doc Summary Index`
- `Asset Summary Index`

### 5.4.4 候选裁剪协议（Pre-Rerank Protocol）

送入 Reranker 前必须经过物理清洗：

- **去重**：基于 `section_id` 唯一化
- **版本清洗**：剔除非活跃、已过期版本
- **噪声剪枝**：剔除极低分候选

### 5.4.5 Rerank 与置信度审计

建议流程：

- 多路召回 Top 50
- Rerank 至 Top 3~5

建议模型：

- BGE-Reranker-v2-m3

同时引入：

- Top1 置信度审计
- 低置信度触发降级或空召回响应

### 5.4.6 空召回与断层补救

#### 1）Section 召回不到

自动降级至：

- Doc Summary 层
- 或全文关键词检索

#### 2）摘要召回过弱

扩大到资产层或宏观文档层，避免直接返回“我不知道”。

### 5.4.7 检索目标

检索层解决的是：

**“在哪找”**

不负责：

- 证据是否最终充分
- 答案是否能安全生成
- 最终权限是否允许输出

这些由后续层处理。

---

## 5.5 L5 精读层（Grounding）

### 5.5.1 职责

精读层负责将摘要相关性转化为原文证据确定性，是整个系统的灵魂层。

### 5.5.2 核心流程

#### 1）Raw Read

根据：

- `doc_id`
- `section_id`
- `asset_id`
- `page_range`
- `raw_locator`
- `byte_range`

从 PostgreSQL / S3 回读原文。

#### 2）Streaming Read

采用：

- S3 Range Request

实现流式局部读取，而不是整篇加载大文档。

#### 3）Dynamic Chunking

对局部原文做 JIT（Just-in-Time）动态切块。

这些 local chunks：

- 只在当前查询中存在
- 不进入长期索引层

#### 4）Evidence Scoring

对 local chunks 做：

- 相关性过滤
- 局部 rerank
- 冲突检查
- 证据打分

#### 5）邻近资产补全

若命中的 section 存在邻近 asset，应顺带捞取：

- 表格
- 图片
- caption
- 相邻说明文本

避免“文字到了，表没到”。

### 5.5.3 精读层性能防御

#### 1）S3 网络并发限制

使用信号量（Semaphore）限制单次查询的并发读取数，避免网络带宽被耗尽。

#### 2）超长 Section 的二次物理拆分

若某 Section 超过 1.5 万字，应在离线期做二次物理分段。

否则 L5 动态切片与评估容易突破熔断阈值。

### 5.5.4 原文读取失败兜底

若 S3 定位偏移量失效：

- 尝试读取当前页全文
- 或返回“原文暂时不可用”
- 或降级为基于摘要的风险提示性回答

### 5.5.5 预算与熔断

建议 v1 固定以下阈值：

- `MAX_SECTIONS_PER_QUERY = 3 ~ 5`
- `MAX_LOCAL_CHUNKS = 15 ~ 20`
- `MAX_GROUNDING_TOKENS = 6k ~ 10k`
- `READ_LATENCY_CIRCUIT = 5s`
- `MAX_PAGES = 10`

一旦超过预算，应：

- 缩减证据范围
- 触发熔断
- 返回“请缩小范围”的系统提示

### 5.5.6 精读层目标

精读层解决的是：

**“找得准”**

即：

- 证据是否真支持问题
- 召回是否可信
- 是否应进入回答阶段

---

## 5.6 L6 合成层（Synthesis）

### 5.6.1 职责

合成层负责基于 evidence pack 输出最终结果。

输出形式可以包括：

- 自然语言回答
- 制度对比说明
- 数据分析结论
- 结构化 JSON
- 报告草稿
- 风险提示

### 5.6.2 子模块

#### 1）Response Generator

负责：

- 总结
- 对比
- 推理
- 回答生成

#### 2）Citation Generator

强制生成引用信息，例如：

- `[doc_id, section_id, page_no]`

#### 3）Policy Guard

合成前执行最终死守：

- 权限复核
- 合规阻断
- 脱敏处理
- 输出审计

### 5.6.3 双闸门机制

系统必须采用双闸门安全机制：

- 第一道闸：L3 / L4 阶段的权限过滤
- 第二道闸：L6 合成前的静态权限与合规复核

即使前序阶段漏掉权限，L6 也必须能拦截输出。

### 5.6.4 合成层目标

合成层解决的是：

**“答得稳”**

---

## 6. 横切基础设施层（Infra）

### 6.1 职责

基础设施层横切所有业务层，负责：

- CDC
- 异步任务
- 索引刷新
- 缓存
- 队列
- 监控
- tracing
- 审计日志
- 失败重试
- 限流与熔断

### 6.2 关键能力

#### CDC 一致性监控

保证：

- PostgreSQL 更新
- S3 更新
- Milvus 更新

之间状态一致。

#### 异步索引 Worker

负责：

- 摘要生成
- 摘要向量化
- 索引写入
- 重建与修复

**绝不在线重建大规模索引。**

#### 热数据缓存

对于最近 30 天更新的文档，可以：

- 缓存 Markdown 结构
- 缓存 section 文本
- 减少 S3 IO

#### CDC 延迟防御

在 L1 维护：

- `index_ready`

只有当 CDC 确认 Milvus 索引已刷新后，才将其置为 `true`，避免“刚上传却搜不到”的不一致体验。

---

## 7. Section Summary 标准化规范

Section Summary 是默认主检索对象，其生成逻辑必须标准化。

### 7.1 必须包含三个硬维度

#### 1）语义核心（Semantic Core）

一句话说明本章节讲什么。

#### 2）事实锚点（Fact Anchors）

强制保留：

- 数字
- 日期
- 制度文号
- 专有名词
- 标准编号
- 指标名

#### 3）结构位置（Structural Hint）

例如：

- 第三章 > 第一节 > 附表 1

### 7.2 摘要模板要求

必须固定：

- 摘要长度
- 关键信息保留规则
- 数字是否强制出现
- 表格引用方式
- 结构路径格式

---

## 8. 核心查询生命周期

### Step 1: Plan

规划层决定：

- 用户在问什么
- 该查哪一类索引
- 该走哪些路径
- 该使用哪些过滤条件

### Step 2: Retrieve

检索层进行：

- Dense 检索
- Sparse 检索
- Hybrid 融合
- 候选清洗
- Rerank

### Step 3: Read

精读层根据命中的对象：

- 回 PostgreSQL / S3
- 读取局部原文
- 邻近资产补全

### Step 4: Ground

精读层进行：

- 动态切片
- 证据过滤
- 局部 rerank
- 冲突检测
- 预算控制

### Step 5: Synthesize

合成层：

- 生成回答
- 插入引用
- 执行权限与合规复核
- 输出结果

---

## 9. 系统边界与避坑清单

### 9.1 不做的事情

- 不全量预切全部原文细块
- 不把 Milvus 当文本库
- 不在线重建百万级索引
- 不无限制回读原文
- 不跳过最终权限复核
- 不把所有粒度摘要混在同一个 Collection 里

### 9.2 必做的事情

- 固定 Section Summary 模板
- 强制 Version Gate
- Predicate Push-down
- L5 预算熔断
- L6 双闸门权限校验
- CDC 一致性监控
- 候选清洗协议
- 置信度审计与降级协议

### 9.3 典型深水区风险

#### 1）白名单过大风险

若 PG 筛出的 `doc_id` 白名单超过 10 万：

- 不应直接拼巨大 `IN` 子句
- 应切换为属性过滤或分批查询

#### 2）Milvus 内存保卫战

- 采用向量量化
- 只给高频过滤字段建标量索引
- 摘要文本不留在 Milvus

#### 3）热冷不分导致成本爆炸

- 文档冷热分区
- 向量冷热隔离
- 原文缓存分层

#### 4）CDC 成为无声杀手

- 主库版本失效后，Milvus 必须同步失效
- 防止旧版本继续被召回

---

## 10. 性能与容量规划

### 10.1 量级估算

- 100 万 Doc -> 约 1000 万 Section 记录
- Milvus 内存占用（按 768 维估算）约 40GB ~ 60GB（含索引开销）

### 10.2 PG 热冷分层

- 活跃文档（`is_active = true`）的元数据常驻热索引
- 历史版本使用表分区

### 10.3 S3 缓存策略

高频访问热文档的 Markdown / Section 片段缓存至：

- Redis
- 本地 SSD

减少 S3 I/O 延迟。

---

## 11. POC 与工程落地建议

### 第一阶段：底层跑通

完成：

- PostgreSQL 文档 / 章节 / 资产 / 版本模型
- S3 原文与 Markdown 存储
- Milvus 三类摘要索引
- Section Summary 标准化模板
- `is_indexed` / `index_ready` 状态位

### 第二阶段：L3 / L4 能力落地

完成：

- Query Planning
- Complexity Gate
- Semantic Router
- Dense + Sparse Hybrid Retrieval
- 候选清洗协议
- Rerank
- Version Gate
- Predicate Push-down

### 第三阶段：L5 Grounding

完成：

- Streaming Raw Read
- Dynamic Chunking
- Local Grounding
- 邻近资产补全
- 预算熔断机制

### 第四阶段：L6 与治理闭环

完成：

- Citation Engine
- Policy Guard
- CDC 监控
- 索引异步刷新与修复
- 输出审计

### 单机流水线阶段建议

- 使用 MinIO 模拟对象存储
- 限制 `max_workers <= CPU 核心数`
- 先保证断点续跑、幂等、批量写入稳定
- 不要一开始就追求全分布式

---

## 12. 三个优先实现的核心模块

### 12.1 `planning_graph`

任务：

- 定义 State 结构
- 定义 LangGraph 节点与状态迁移
- 输出包含 `query`、`filters`、`target_collection` 的指令包

### 12.2 `retrieval_adapter`

任务：

- 封装 Milvus `hybrid_search`
- 实现动态谓词转换逻辑
- 实现多路召回合并算法

### 12.3 `rerank_service`

任务：

- 独立负责候选清洗
- 处理模型打分与截断逻辑
- 防止 GPU / 内存 OOM

---

## 13. 系统评估指标

建议在 POC 阶段重点拷打以下三个指标：

### 13.1 Grounding 准确率（Faithfulness）

最终回答中，有多少内容能在引用原文中找到直接证据。

目标：

- >95%

### 13.2 摘要召回偏差（Recall Bias）

比较：

- 摘要检索路径
- 全量细 Chunk 检索路径

两者命中的核心文档重合度。

用于验证摘要质量。

### 13.3 首字响应延迟（TTFT）

在 L5 流式精读场景下：

- 第一个 Token 是否能在 2s 左右吐出

---

## 14. 一句话架构定义

系统以 PostgreSQL / S3 作为事实与原文层，以 Milvus 的 Dense + Sparse 多粒度摘要索引作为轻量召回层，通过 L3 规划层的复杂度栅栏、语义路由与谓词下推缩小搜索空间，再由 L5 流式精读层完成原文级 grounding，最终由 L6 合成层在权限、合规与审计双重约束下输出带引用答案。

---

## 15. 总结

这套最终整合版架构的核心价值在于：

- 不贪心地预处理全部文档
- 不把检索误当回答
- 不牺牲回答阶段的证据真实性
- 不把治理当作附属能力
- 不只解决“重”，也解决“贵”和“不稳”

它适合支撑：

- 百万级企业知识管理
- 高合规私有知识问答
- 证据驱动的分析与生成
- 面向后续 Agent 化扩展的正式底座

这份文档可以直接作为第一阶段工程开发的统一主文档。

---

## 16. 当前工程落地状态与运行命令（2026-05-02）

本节记录当前项目主线已经落地的工程状态，以及本地重跑私有数据、生成测试集、检索评测的标准命令。

### 16.1 当前主线契约

当前新系统主线只认以下公共契约：

- `Document`
- `SectionRecord`
- `AssetRecord`
- `DocSummaryRecord / SectionSummaryRecord / AssetSummaryRecord`
- `GroundingTarget`
- `EvidenceItem`

旧 `Chunk / Segment / mode mix/local/global` 不再作为主线设计对象。`rag/agent/**` 也不纳入当前重构主线。

### 16.2 当前支持的文件类型

当前私有入库脚本支持：

- `.pdf`
- `.docx`
- `.md`
- `.markdown`
- `.xlsx`
- `.xls`
- `.pptx`
- `.png`
- `.jpg`
- `.jpeg`
- `.txt`

解析策略：

- PDF / DOCX / Markdown：Docling 结构化解析。
- Excel：Pandas/OpenPyXL 原生表格解析。
- PPTX：python-pptx 原生解析。
- 图片：OCR repo。
- TXT：纯文本解析。

### 16.3 Word/PDF/Markdown 表格处理标准

表格不再直接拼进 Section 正文。

标准流程：

1. Parser 遇到表格，生成独立 `ParsedElement(kind="table")`。
2. 正文位置只写入 `[ASSET_ANCHOR:...]`。
3. `SectionRefiner` token 窗口切分后，后处理扫描锚点。
4. 锚点在哪个细粒度 `SectionRecord`，表格资产就绑定到哪个 `section_id`。
5. L5 命中 section 后，不回填 Markdown 表格，只暴露表格 schema/sample 和计算入口。
6. 任何表格数据值、过滤、排序、聚合、排名、对比问题，都必须进入 DuckDB Text-to-SQL Sandbox。

这解决两个问题：

- `normalize_whitespace()` 不会压扁 Markdown 表格源码。
- 超长 section 被切成多个窗口后，不会出现“字到了，表没到”的资产错位。

### 16.4 表格处理官方标准：DuckDB Text-to-SQL Sandbox

Excel / Word 表格 / PPT 表格不走“全表转 Markdown 后切块”，也不允许“短表直接回填 Markdown”。表格是最容易造成 LLM 目测、误算、排序幻觉的资产类型，因此当前 RAG 主线采用单一标准：

**所有表格一律 `table_policy=compute_only`，统一进入 DuckDB Text-to-SQL Sandbox。**

#### 废除项

- 废除 `inline_context`：短表也不能直接回填 Markdown。
- 废除“让模型看 sample rows 回答具体数据问题”的做法：sample 只用于理解字段、生成 SQL 和解释结构。
- `summary_only` 只保留为检索摘要语义，不再作为表格处理策略。

#### 当前主线标准

- Parser 遇到表格，生成独立 `AssetRecord`。
- 正文只保留 `[ASSET_ANCHOR:...]`，不拼接 Markdown 表格。
- `AssetRecord` 记录：
  - `sheet_name`
  - `row_count`
  - `column_count`
  - `sample_rows`
  - `schema`
  - `estimated_tokens`
  - `table_policy=compute_only`
  - `storage_key`
- L5 命中表格锚点后，只向 L6 暴露结构说明、schema/sample 和计算指令，不暴露全量 Markdown。
- L6 如需回答数据值、筛选、排序、聚合、排名、对比问题，必须生成受限 DuckDB `SELECT`。
- DuckDB Sandbox 执行 SQL 后，将结果表作为 `TABLE_COMPUTE_RESULT` 证据交回 L6 合成。

#### Phase 4：高级数据分析能力

MCP/Pandas 不属于当前 RAG 主线，统一移出到 Phase 4。Phase 4 只面向更高级的数据分析场景，例如多表联动、复杂 Python 分析、可视化、长任务编排和交互式数据探索。

当前 RAG 主线只认：

```text
表格资产 -> schema/sample -> Text-to-SQL -> DuckDB Sandbox -> 计算结果 -> L6 合成
```

### 16.5 默认模型与后端

当前默认建议：

- 向量模型：`qwen3-embedding:8b`
- 摘要模型：`Qwen/Qwen3-8B-MLX-4bit`
- Rerank 模型：`BAAI/bge-reranker-v2-m3`
- 默认向量后端：Milvus

注意：

- 摘要模型不能默认复用聊天模型。
- 入库和检索必须保持 embedding / tokenizer / chunk 参数一致。
- 当前所有拆分和预算计数都按 token，而不是字符。

### 16.6 环境准备命令

安装依赖：

```bash
uv sync
```

设置 Milvus：

```bash
export MILVUS_URI=http://127.0.0.1:19530
export RAG_MILVUS_URI=$MILVUS_URI
```

准备 embedding：

```bash
ollama pull qwen3-embedding:8b
ollama serve
```

生成测试题需要启动 MLX OpenAI-compatible server：

```bash
uv run mlx_lm.server \
  --model Qwen/Qwen3-8B-MLX-4bit \
  --host 127.0.0.1 \
  --port 8080 \
  --max-tokens 1024 \
  --temp 0.1 \
  --chat-template-args '{"enable_thinking":false}'
```

### 16.7 私有数据重切、入库、生成摘要、写 Milvus

先设置变量：

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

重切没有独立脚本，重新跑 ingest 即重新解析和重新切分：

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

如果复用同一个 Milvus collection prefix，先清旧 collection：

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

### 16.8 导出 SectionRecord JSONL

```bash
uv run python scripts/export_private_sections.py \
  --storage-root "$STORAGE_ROOT" \
  --output data/eval_private/company_policy_sections_v4.jsonl
```

检查数量：

```bash
wc -l data/eval_private/company_policy_sections_v4.jsonl
```

### 16.9 生成私有 golden eval 测试集

先 smoke：

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

再全量：

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

### 16.10 私有检索评测命令

不开 rerank：

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

开启 rerank：

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

### 16.11 公开 MedicalRetrieval mini 命令

下载与准备：

```bash
uv run python scripts/download_public_benchmark.py --dataset medical_retrieval
uv run python scripts/prepare_public_benchmark.py --dataset medical_retrieval
```

入库：

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

不开 rerank：

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

开启 rerank：

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

### 16.12 回归测试命令

全量：

```bash
uv run ruff check rag scripts tests
uv run pytest -q
```

Excel / 表格资产 / 摘要 / Grounding 重点回归：

```bash
uv run pytest -q \
  tests/core/test_excel_parser_repo.py \
  tests/core/test_ingest_asset_anchors.py \
  tests/core/test_retrieval_summarizer.py \
  tests/service/test_grounding_service.py
```

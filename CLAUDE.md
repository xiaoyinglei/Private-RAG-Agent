# CLAUDE.md — AI Coding Agent Reference

> 面向 AI coding agent 的项目入口。设计说明见 [README.md](README.md)。

## 环境

```bash
cd "/Users/leixiaoying/LLM/RAG学习"
uv sync          # 安装依赖
```

Python 3.12、包管理用 `uv`（`uv run`、`uv pip`、`uv add`）。

## 关键命令

```bash
# 完整测试（忽略需 LLM 环境的 pre-existing 失败）
uv run pytest tests/agent/ -q \
  --ignore=tests/agent/test_agent_service_resume.py \
  --ignore=tests/agent/test_agent_service_loop_boundary.py \
  --ignore=tests/agent/test_agent_graph_compiler.py

# 全部测试
uv run pytest -q

# 静态检查
uv run ruff check
uv run mypy                              # 当前 0 error
uv run lint-imports
```

## 架构要点

- **Agent 内核**：单入口 `generic`、Claude-like while-loop、`GENERIC_SYSTEM_PROMPT`
- **工具系统**：core tools 常驻 / deferred tools 搜索激活
  - Core：`tool_search`、`activate_tools`、`task`、`list_files`、`read_file`、`write_file`、`run_python_inline`
  - Deferred：RAG、asset、LLM 等，搜索激活后才暴露
  - MCP 工具通过 Adapter 进入统一 Catalog
- **RAG** 是 Agent 的工具子系统，不是默认入口
- **ToolSpec** 即工具契约（`rag/agent/tools/spec.py`）
- **ToolRegistry + ToolExecutionService** 是执行权威
- **BaseTool**：9/9 workspace 工具已迁移完成

## 关键路径

| 组件 | 路径 |
|------|------|
| Agent Loop | `rag/agent/loop/runtime.py` |
| System Prompt | `rag/agent/builtin/generic.py` → `GENERIC_SYSTEM_PROMPT` |
| Tool Catalog | `rag/agent/capabilities/catalog.py` |
| Tool Spec | `rag/agent/tools/spec.py` |
| Workspace Ops | `rag/agent/primitive_ops.py` |
| RAG Pipeline | `rag/retrieval/`、`rag/ingest/` |
| CLI | `rag/cli.py` |
| Config | `configs/models.yaml` |
| Docs | `docs/RUNBOOK.md`、`docs/EVALUATION.md`、`docs/TROUBLESHOOTING.md` |

## 约束

- **uv 管理环境**：只用 `uv run`、`uv pip`、`uv add`，不用裸 pip/python
- **嵌入模型切换**：必须重建 Milvus collection（不同维度不能混用）
- **表格真实值**：不信任 `sample_rows`，走 DuckDB sandbox 或本地 Python 计算
- **Agent CLI**：默认 `--agent generic`，旧角色名（`research`/`orchestrator`等）已废弃
- **9091 端口**：被 Milvus 占用，rerank 服务用 9092
- **mypy 零容忍**：保持 0 错误

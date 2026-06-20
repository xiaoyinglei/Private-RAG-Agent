from __future__ import annotations

from collections.abc import Mapping

from langgraph.checkpoint.base import BaseCheckpointSaver

from rag.agent.builtin_registry import create_builtin_tool_registry
from rag.agent.core.definition import AgentDefinition, ModelSelectionPolicy, ToolPolicy
from rag.agent.core.delegation import DelegatedAgentRunner
from rag.agent.core.llm_registry import ModelRegistry
from rag.agent.core.runtime_diagnostics import RuntimeDiagnostic
from rag.agent.core.runtime_ports import RetrievalHintProvider
from rag.agent.loop.runtime import ModelTurnProvider
from rag.agent.service import AgentService
from rag.agent.tools.registry import ToolRunner

_SENTINEL = object()


RESEARCH_AGENT_SYSTEM_PROMPT = """You are the ResearchAgent for deep single-topic research.

Your task workflow:
1. If the task involves a local file (xlsx, csv, json, etc.), use list_files to find it,
   then use run_python_inline to read and analyze it directly. Skip RAG tools entirely.
2. If the task requires searching indexed documents, first call tool_search to discover
   retrieval tools (vector_search, keyword_search, etc.), then activate_tools to load them.
   Only after activation can you use those tools.
3. For structured data analysis, use tool_search to find asset tools, activate them,
   then use asset_inspect/asset_read_slice/asset_analyze.

Use retrieved evidence as the factual authority. Preserve evidence ids, citations,
retrieval scores, citation anchors, and grounding metadata whenever available.
Do not invent facts. When evidence is insufficient, state insufficient evidence.

For local files:
- Use the workspace-relative path (e.g. 'input_files/filename.xlsx'), NOT absolute paths.
- For .xlsx: run_python_inline with openpyxl
- For .csv: run_python_inline with pandas
- NEVER try to index or search for a file that is already local.
"""


RESEARCH_AGENT = AgentDefinition(
    agent_type="research",
    description="Deep single-topic research with grounded evidence and citations.",
    system_prompt=RESEARCH_AGENT_SYSTEM_PROMPT,
    # TODO: agent_* / rag_* / llm_* tool names must match ToolRegistry registration.
    allowed_tools=[
        # core — always visible
        "tool_search",
        "activate_tools",
        "list_files",
        "read_file",
        "write_file",
        "run_python_inline",
        # deferred — visible after tool_search + activate_tools
        "vector_search",
        "keyword_search",
        "grounding",
        "rerank",
        "asset_list",
        "asset_inspect",
        "asset_read_slice",
        "asset_analyze",
        "llm_summarize",
        "rag_search_answer",
        "structured_probe",
    ],
    # TODO: migrate estimated_token_budget / max_iterations / max_depth to runtime config
    estimated_token_budget=96_000,
    estimated_work_budget=20_000,
    model_selection=ModelSelectionPolicy(
        thinking=True,
        retrieval_hint_max_tokens=256,
        tool_decision_max_tokens=2048,
    ),
    max_iterations=10,
    max_depth=2,
    tool_policy=ToolPolicy(max_parallel_calls=4),
)


def create_research_agent_service(
    *,
    runners: Mapping[str, ToolRunner] | None = None,
    model_turn_provider: ModelTurnProvider | None = None,
    retrieval_hint_provider: RetrievalHintProvider | None = None,
    subagent_runner: DelegatedAgentRunner | None = None,
    model_registry: ModelRegistry | None | object = _SENTINEL,
    checkpointer: BaseCheckpointSaver[str] | None = None,
) -> AgentService:
    # 默认自动加载 models.yaml，测试环境可显式传 None 跳过
    registry: ModelRegistry | None
    runtime_diagnostics: tuple[RuntimeDiagnostic, ...] = ()
    if model_registry is _SENTINEL:
        try:
            registry = ModelRegistry.from_env()
        except Exception as exc:
            registry = None
            runtime_diagnostics = (
                RuntimeDiagnostic.from_exception(
                    code="model_registry_initialization_failed",
                    component="model_registry",
                    error=exc,
                ),
            )
    else:
        registry = model_registry  # type: ignore[assignment]
    return AgentService(
        definition=RESEARCH_AGENT,
        tool_registry=create_builtin_tool_registry(runners=runners),
        model_turn_provider=model_turn_provider,
        retrieval_hint_provider=retrieval_hint_provider,
        subagent_runner=subagent_runner,
        model_registry=registry,
        checkpointer=checkpointer,
        runtime_diagnostics=runtime_diagnostics,
    )

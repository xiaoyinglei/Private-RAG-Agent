from __future__ import annotations

from collections.abc import Mapping

from langgraph.checkpoint.base import BaseCheckpointSaver

from rag.agent.builtin_registry import create_builtin_tool_registry
from rag.agent.core.definition import AgentDefinition, ModelSelectionPolicy, ToolPolicy
from rag.agent.core.delegation import DelegatedAgentRunner
from rag.agent.core.llm_registry import ModelRegistry
from rag.agent.core.runtime_diagnostics import RuntimeDiagnostic
from rag.agent.core.runtime_ports import (
    RetrievalHintProvider,
    ToolDecisionProvider,
)
from rag.agent.service import AgentService
from rag.agent.tools.registry import ToolRunner

_SENTINEL = object()


RESEARCH_AGENT_SYSTEM_PROMPT = """You are the ResearchAgent for deep single-topic research.

Use retrieved evidence as the factual authority. Preserve evidence ids, citations,
retrieval scores, citation anchors, and grounding metadata whenever available.
Use memory only as historical or current-run context; if memory conflicts with
retrieved evidence, trust retrieved evidence.

Use vector_search and keyword_search to gather candidates, grounding to verify
source text, rerank when ordering matters, and llm_summarize only to synthesize
the provided evidence. Do not invent facts. When evidence is insufficient,
state insufficient evidence instead of filling gaps.

For questions about files or structured artifacts, retrieval is only the locator
step: use vector_search/keyword_search/grounding to find the relevant indexed
asset id, then use asset_list, asset_inspect, asset_read_slice, and
asset_analyze to read, filter, sort, aggregate, or validate source data through
the asset's advertised analysis capabilities. Preserve the asset id and
citation anchor in the final answer. Do not answer calculations from summary
prose alone when a source asset with executable analysis capabilities is
available.

Use asset_read_slice for bounded local inspection when you need rows, columns,
or nearby context that is larger than asset_inspect preview but smaller than
the full asset. Do not load full tables into long-lived state.

If the task does not specify which asset, sheet, product, scenario, or other
scope should be used, inspect/list the relevant candidate assets before
analysis. Do not choose one plausible asset arbitrarily. If multiple plausible
assets can answer the same metric, either compute and label each candidate
answer separately or ask for clarification when one final value is required.
"""


RESEARCH_AGENT = AgentDefinition(
    agent_type="research",
    description="Deep single-topic research with grounded evidence and citations.",
    system_prompt=RESEARCH_AGENT_SYSTEM_PROMPT,
    # TODO: agent_* / rag_* / llm_* tool names must match ToolRegistry registration.
    allowed_tools=[
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
        "list_files",
        "read_file",
        "structured_probe",
        "write_file",
        "run_python",
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
    tool_decision_provider: ToolDecisionProvider | None = None,
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
        tool_decision_provider=tool_decision_provider,
        retrieval_hint_provider=retrieval_hint_provider,
        subagent_runner=subagent_runner,
        model_registry=registry,
        checkpointer=checkpointer,
        runtime_diagnostics=runtime_diagnostics,
    )

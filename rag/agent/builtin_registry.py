from __future__ import annotations

from collections.abc import Mapping

from rag.agent.tools.asset_tools import ALL_ASSET_TOOLS
from rag.agent.tools.formatter import ToolOutputFormatter
from rag.agent.tools.formatters.generic_tools import (
    ApplyPatchFormatter,
    RunCommandFormatter,
    SearchTextFormatter,
    ToolReplFormatter,
    UpdatePlanFormatter,
)
from rag.agent.tools.formatters.asset_tools import (
    AssetAnalyzeFormatter,
    AssetInspectFormatter,
    AssetListFormatter,
    AssetReadSliceFormatter,
)
from rag.agent.tools.formatters.file_tools import (
    ListFilesFormatter,
    ReadFileFormatter,
    RunPythonFormatter,
    StructuredProbeFormatter,
    WriteFileFormatter,
)
from rag.agent.tools.formatters.llm_tools import (
    LLMCompareFormatter,
    LLMGenerateFormatter,
    LLMSummarizeFormatter,
)
from rag.agent.tools.formatters.rag_retrieval import (
    GraphExpandFormatter,
    GroundingFormatter,
    KeywordSearchFormatter,
    RAGSearchAnswerFormatter,
    RerankFormatter,
    SearchAssetsFormatter,
    SearchKnowledgeFormatter,
    VectorSearchFormatter,
)
from rag.agent.tools.generic_tools import ALL_GENERIC_TOOLS
from rag.agent.tools.llm_tools import ALL_LLM_TOOLS
from rag.agent.tools.primitive_tools import ALL_PRIMITIVE_TOOLS
from rag.agent.tools.rag_answer_tools import ALL_RAG_ANSWER_TOOLS
from rag.agent.tools.rag_semantic_tools import ALL_RAG_SEMANTIC_TOOLS
from rag.agent.tools.rag_tools import ALL_RAG_TOOLS
from rag.agent.tools.registry import ContextualToolRunner, ToolRegistry, ToolRunner


def create_builtin_tool_registry(
    *,
    runners: Mapping[str, ToolRunner] | None = None,
    contextual_runners: Mapping[str, ContextualToolRunner] | None = None,
) -> ToolRegistry:
    registry = ToolRegistry()
    all_tools = [
        *ALL_RAG_TOOLS,
        *ALL_ASSET_TOOLS,
        *ALL_LLM_TOOLS,
        *ALL_RAG_ANSWER_TOOLS,
        *ALL_RAG_SEMANTIC_TOOLS,
        *ALL_PRIMITIVE_TOOLS,
        *ALL_GENERIC_TOOLS,
    ]

    runner_by_name = runners or {}
    contextual_runner_by_name = contextual_runners or {}
    duplicate_runners = sorted(set(runner_by_name) & set(contextual_runner_by_name))
    if duplicate_runners:
        raise ValueError(f"runners cannot be both ordinary and contextual: {', '.join(duplicate_runners)}")
    for spec in all_tools:
        registry.register(spec, runner=runner_by_name.get(spec.name))
        contextual_runner = contextual_runner_by_name.get(spec.name)
        if contextual_runner is not None:
            registry.register_contextual_runner(spec.name, contextual_runner)

    # Register RAG retrieval formatters
    retrieval_formatters: tuple[ToolOutputFormatter, ...] = (
        VectorSearchFormatter(),
        KeywordSearchFormatter(),
        GroundingFormatter(),
        RerankFormatter(),
        GraphExpandFormatter(),
    )
    for formatter in retrieval_formatters:
        registry.register_formatter(formatter)

    # Register file/workspace tool formatters
    file_formatters: tuple[ToolOutputFormatter, ...] = (
        ListFilesFormatter(),
        ReadFileFormatter(),
        WriteFileFormatter(),
        RunPythonFormatter(),
        StructuredProbeFormatter(),
    )
    for formatter in file_formatters:
        registry.register_formatter(formatter)

    # Register asset tool formatters (PR6)
    asset_formatters: tuple[ToolOutputFormatter, ...] = (
        AssetListFormatter(),
        AssetInspectFormatter(),
        AssetReadSliceFormatter(),
        AssetAnalyzeFormatter(),
    )
    for formatter in asset_formatters:
        registry.register_formatter(formatter)

    # Register LLM tool formatters (PR6)
    llm_formatters: tuple[ToolOutputFormatter, ...] = (
        LLMGenerateFormatter(),
        LLMSummarizeFormatter(),
        LLMCompareFormatter(),
    )
    for formatter in llm_formatters:
        registry.register_formatter(formatter)

    # Register RAG search answer formatter (PR6)
    registry.register_formatter(RAGSearchAnswerFormatter())

    # Register semantic RAG tool formatters
    registry.register_formatter(SearchKnowledgeFormatter())
    registry.register_formatter(SearchAssetsFormatter())

    # Register generic coding-agent tool formatters
    generic_formatters: tuple[ToolOutputFormatter, ...] = (
        SearchTextFormatter(),
        ApplyPatchFormatter(),
        RunCommandFormatter(),
        UpdatePlanFormatter(),
        ToolReplFormatter(),
    )
    for formatter in generic_formatters:
        registry.register_formatter(formatter)

    known_tool_names = {spec.name for spec in all_tools}
    unknown_runners = sorted((set(runner_by_name) | set(contextual_runner_by_name)) - known_tool_names)
    if unknown_runners:
        raise ValueError(f"unknown builtin tool runners: {', '.join(unknown_runners)}")
    return registry

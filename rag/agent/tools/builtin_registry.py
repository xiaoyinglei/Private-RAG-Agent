from __future__ import annotations

from collections.abc import Mapping

from rag.agent.tools.fast_path_tools import ALL_FAST_PATH_TOOLS
from rag.agent.tools.llm_tools import ALL_LLM_TOOLS
from rag.agent.tools.rag_tools import ALL_RAG_TOOLS
from rag.agent.tools.registry import ToolRegistry, ToolRunner


def create_builtin_tool_registry(
    *,
    runners: Mapping[str, ToolRunner] | None = None,
) -> ToolRegistry:
    registry = ToolRegistry()
    all_tools = [*ALL_RAG_TOOLS, *ALL_LLM_TOOLS, *ALL_FAST_PATH_TOOLS]
    runner_by_name = runners or {}
    for spec in all_tools:
        registry.register(spec, runner=runner_by_name.get(spec.name))
    unknown_runners = sorted(set(runner_by_name) - {spec.name for spec in all_tools})
    if unknown_runners:
        raise ValueError(f"unknown builtin tool runners: {', '.join(unknown_runners)}")
    return registry

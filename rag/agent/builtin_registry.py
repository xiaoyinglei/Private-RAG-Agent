from __future__ import annotations

from collections.abc import Mapping

from rag.agent.tools.asset_tools import ALL_ASSET_TOOLS
from rag.agent.tools.llm_tools import ALL_LLM_TOOLS
from rag.agent.tools.primitive_tools import ALL_PRIMITIVE_TOOLS
from rag.agent.tools.rag_answer_tools import ALL_RAG_ANSWER_TOOLS
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
        *ALL_PRIMITIVE_TOOLS,
    ]

    runner_by_name = runners or {}
    contextual_runner_by_name = contextual_runners or {}
    duplicate_runners = sorted(set(runner_by_name) & set(contextual_runner_by_name))
    if duplicate_runners:
        raise ValueError(
            "runners cannot be both ordinary and contextual: "
            f"{', '.join(duplicate_runners)}"
        )
    for spec in all_tools:
        registry.register(spec, runner=runner_by_name.get(spec.name))
        contextual_runner = contextual_runner_by_name.get(spec.name)
        if contextual_runner is not None:
            registry.register_contextual_runner(spec.name, contextual_runner)
    known_tool_names = {spec.name for spec in all_tools}
    unknown_runners = sorted(
        (set(runner_by_name) | set(contextual_runner_by_name)) - known_tool_names
    )
    if unknown_runners:
        raise ValueError(
            f"unknown builtin tool runners: {', '.join(unknown_runners)}"
        )
    return registry

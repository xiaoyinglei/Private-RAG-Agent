from __future__ import annotations

from collections.abc import Mapping

from rag.agent.core.agent_as_tool import build_agent_tool_spec
from rag.agent.tools.asset_tools import ALL_ASSET_TOOLS
from rag.agent.tools.fast_path_tools import ALL_FAST_PATH_TOOLS
from rag.agent.tools.llm_tools import ALL_LLM_TOOLS
from rag.agent.tools.rag_tools import ALL_RAG_TOOLS
from rag.agent.tools.registry import ToolRegistry, ToolRunner


def create_builtin_tool_registry(
    *,
    runners: Mapping[str, ToolRunner] | None = None,
) -> ToolRegistry:
    # Lazy import to avoid circular: builtin_registry → builtin → research → builtin_registry
    from rag.agent.builtin import (  # noqa: PLC0415
        COMPARE_AGENT,
        FACTCHECK_AGENT,
        RESEARCH_AGENT,
        SYNTHESIZE_AGENT,
    )

    registry = ToolRegistry()

    # 1. 标准工具（RAG + Indexed assets + LLM + Fast Path）
    all_tools = [*ALL_RAG_TOOLS, *ALL_ASSET_TOOLS, *ALL_LLM_TOOLS, *ALL_FAST_PATH_TOOLS]

    # 2. Agent-as-tool 工具（静态注册 ToolSpec，不带 runner）
    agent_defs = [RESEARCH_AGENT, COMPARE_AGENT, FACTCHECK_AGENT, SYNTHESIZE_AGENT]
    agent_tool_specs = [
        build_agent_tool_spec(definition).tool_spec for definition in agent_defs
    ]
    agent_tool_names = {spec.name for spec in agent_tool_specs}

    runner_by_name = runners or {}
    for spec in all_tools + agent_tool_specs:
        registry.register(spec, runner=runner_by_name.get(spec.name))
    unknown_runners = sorted(
        set(runner_by_name) - {spec.name for spec in all_tools} - agent_tool_names
    )
    if unknown_runners:
        raise ValueError(f"unknown builtin tool runners: {', '.join(unknown_runners)}")
    return registry

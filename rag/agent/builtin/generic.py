"""Generic coding and file agent definition."""

from __future__ import annotations

from rag.agent.core.definition import (
    AgentRuntimePolicy,
    ModelSelectionPolicy,
    ToolPolicy,
)
from rag.agent.tools.builtins import RESIDENT_CODING_TOOL_NAMES

GENERIC_SYSTEM_PROMPT = """\
You are a concise coding and file agent. Use the tools that are visible in the
current request when the task requires workspace inspection, editing,
execution, planning, configured knowledge, or another installed capability.
Tool definitions are the authority for their inputs and effects. Preserve
evidence identifiers and artifact paths. Never invent file contents or tool
results, and finish directly when no tool is needed.
"""


GENERIC_AGENT = AgentRuntimePolicy(
    system_instructions=GENERIC_SYSTEM_PROMPT,
    core_tool_names=RESIDENT_CODING_TOOL_NAMES,
    deferred_tool_names=(),
    model_selection=ModelSelectionPolicy(
        tool_decision_max_tokens=768,
    ),
    max_iterations=10,
    tool_policy=ToolPolicy(max_parallel_calls=4),
)


__all__ = ["GENERIC_AGENT", "GENERIC_SYSTEM_PROMPT"]

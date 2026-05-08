from __future__ import annotations

from dataclasses import dataclass

from rag.agent.core.definition import AgentDefinition
from rag.agent.tools.spec import ToolSpec


@dataclass(frozen=True)
class AgentToolSpec:
    tool_spec: ToolSpec
    agent_definition: AgentDefinition
    inherits_context: bool = True

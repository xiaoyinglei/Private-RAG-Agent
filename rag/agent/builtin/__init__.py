"""Built-in agent definitions."""

from rag.agent.builtin.generic import GENERIC_AGENT
from rag.agent.core.definition import AgentRuntimePolicy
from rag.agent.core.registry import AgentRegistry

BUILTIN_AGENT_DEFINITIONS: dict[str, AgentRuntimePolicy] = {
    "generic": GENERIC_AGENT,
}


def register_builtin_agents(registry: AgentRegistry) -> None:
    for definition in BUILTIN_AGENT_DEFINITIONS.values():
        registry.register(definition)


def create_builtin_agent_registry() -> AgentRegistry:
    registry = AgentRegistry()
    register_builtin_agents(registry)
    return registry


__all__ = [
    "BUILTIN_AGENT_DEFINITIONS",
    "GENERIC_AGENT",
    "create_builtin_agent_registry",
    "register_builtin_agents",
]

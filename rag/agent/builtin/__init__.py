"""Built-in agent definitions."""

from rag.agent.builtin.compare import COMPARE_AGENT
from rag.agent.builtin.factcheck import FACTCHECK_AGENT
from rag.agent.builtin.generic import GENERIC_AGENT
from rag.agent.builtin.orchestrator import ORCHESTRATOR_AGENT
from rag.agent.builtin.research import RESEARCH_AGENT, create_research_agent_service
from rag.agent.builtin.synthesize import SYNTHESIZE_AGENT
from rag.agent.core.definition import AgentDefinition
from rag.agent.core.registry import AgentRegistry

BUILTIN_AGENT_DEFINITIONS: dict[str, AgentDefinition] = {
    "generic": GENERIC_AGENT,
    "research": RESEARCH_AGENT,
    "orchestrator": ORCHESTRATOR_AGENT,
    "compare": COMPARE_AGENT,
    "factcheck": FACTCHECK_AGENT,
    "synthesize": SYNTHESIZE_AGENT,
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
    "COMPARE_AGENT",
    "FACTCHECK_AGENT",
    "GENERIC_AGENT",
    "ORCHESTRATOR_AGENT",
    "RESEARCH_AGENT",
    "SYNTHESIZE_AGENT",
    "create_builtin_agent_registry",
    "create_research_agent_service",
    "register_builtin_agents",
]

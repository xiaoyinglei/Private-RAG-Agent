from __future__ import annotations

from rag.agent.core.definition import AgentDefinition


class AgentRegistry:
    _agents: dict[str, AgentDefinition] = {}

    @classmethod
    def register(cls, definition: AgentDefinition) -> None:
        cls._agents[definition.agent_type] = definition

    @classmethod
    def get(cls, agent_type: str) -> AgentDefinition:
        if agent_type not in cls._agents:
            raise KeyError(f"Agent type '{agent_type}' not found in registry")
        return cls._agents[agent_type]

    @classmethod
    def list_all(cls) -> list[AgentDefinition]:
        return list(cls._agents.values())

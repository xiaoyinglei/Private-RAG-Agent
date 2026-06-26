from __future__ import annotations

from rag.agent.core.definition import AgentRuntimePolicy


class AgentRegistry:
    def __init__(self) -> None:
        self._agents: dict[str, AgentRuntimePolicy] = {}

    def register(self, definition: AgentRuntimePolicy, *, replace: bool = False) -> None:
        if not replace and definition.agent_type in self._agents:
            raise ValueError(f"Agent type '{definition.agent_type}' already registered")
        self._agents[definition.agent_type] = definition

    def get(self, agent_type: str) -> AgentRuntimePolicy:
        if agent_type not in self._agents:
            raise KeyError(f"Agent type '{agent_type}' not found in registry")
        return self._agents[agent_type]

    def list_all(self) -> list[AgentRuntimePolicy]:
        return list(self._agents.values())

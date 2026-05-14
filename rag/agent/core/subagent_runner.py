from __future__ import annotations

from dataclasses import replace

from rag.agent.core.agent_service_factory import AgentServiceFactory
from rag.agent.core.context import derive_child_config
from rag.agent.core.registry import AgentRegistry
from rag.agent.core.task import SubTaskNode
from rag.agent.service import AgentRunResult
from rag.agent.state import AgentState


class BuiltinSubAgentRunner:
    def __init__(
        self,
        *,
        agent_registry: AgentRegistry,
        service_factory: AgentServiceFactory,
    ) -> None:
        self._agent_registry = agent_registry
        self._service_factory = service_factory

    async def run_subtask(
        self,
        *,
        subtask: SubTaskNode,
        parent_state: AgentState,
    ) -> AgentRunResult:
        child_definition = self._agent_registry.get(subtask.agent_type)
        child_config = derive_child_config(parent_state["run_config"], child_definition)
        if subtask.estimated_tokens is not None:
            child_config = replace(child_config, budget_total=subtask.estimated_tokens)

        child_service = self._service_factory.create(child_definition)
        return await child_service.run_with_config(
            task=subtask.prompt,
            run_config=child_config,
        )

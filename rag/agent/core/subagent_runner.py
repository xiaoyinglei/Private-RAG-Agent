from __future__ import annotations

from dataclasses import replace

from rag.agent.core.agent_service_factory import AgentServiceFactory
from rag.agent.core.context import derive_child_config
from rag.agent.core.delegation import (
    AgentDelegationRequest,
    ParentAgentContext,
)
from rag.agent.core.registry import AgentRegistry
from rag.agent.service import AgentRunResult


class BuiltinSubAgentRunner:
    def __init__(
        self,
        *,
        agent_registry: AgentRegistry,
        service_factory: AgentServiceFactory,
    ) -> None:
        self._agent_registry = agent_registry
        self._service_factory = service_factory

    async def run_delegated_task(
        self,
        *,
        request: AgentDelegationRequest,
        parent_state: ParentAgentContext,
    ) -> AgentRunResult:
        child_definition = self._agent_registry.get(request.agent_type)
        child_config = derive_child_config(parent_state["run_config"], child_definition)
        if request.estimated_tokens is not None:
            child_config = replace(child_config, budget_total=request.estimated_tokens)

        child_service = self._service_factory.create(child_definition)
        return await child_service.run_with_config(
            task=request.prompt,
            run_config=child_config,
        )

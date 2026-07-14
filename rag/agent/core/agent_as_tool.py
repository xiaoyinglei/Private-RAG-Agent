from __future__ import annotations

from dataclasses import replace

from rag.agent.core.agent_tool_contract import (
    AgentAsToolAdapter,
    AgentToolInput,
    AgentToolOutput,
    DelegatedEvidenceRef,
)
from rag.agent.core.context import derive_child_config
from rag.agent.core.delegation import (
    AgentAsToolExecutionError,
    AgentDelegationRequest,
    DelegatedAgentResult,
    ParentAgentContext,
)
from rag.agent.core.registry import AgentRegistry
from rag.agent.core.runtime_ports import RetrievalHintProvider
from rag.agent.loop.runtime import ModelTurnProvider
from rag.agent.tools.registry import ToolRegistry


class AgentAsToolRunner:
    def __init__(
        self,
        *,
        tool_registry: ToolRegistry,
        agent_registry: AgentRegistry,
        model_turn_provider: ModelTurnProvider | None = None,
        retrieval_hint_provider: RetrievalHintProvider | None = None,
    ) -> None:
        self._tool_registry = tool_registry
        self._agent_registry = agent_registry
        self._model_turn_provider = model_turn_provider
        self._retrieval_hint_provider = retrieval_hint_provider

    async def run_delegated_task(
        self,
        *,
        request: AgentDelegationRequest,
        parent_state: ParentAgentContext,
    ) -> DelegatedAgentResult:
        from rag.agent.service import AgentService

        parent_config = parent_state["run_config"]
        child_definition = self._agent_registry.get(request.agent_type)
        child_config = derive_child_config(parent_config, child_definition)
        if request.llm_budget_total is not None:
            child_config = replace(
                child_config,
                llm_budget_total=request.llm_budget_total,
            )
        if request.max_turns is not None:
            child_config = replace(child_config, max_turns=request.max_turns)

        service = AgentService(
            definition=child_definition,
            tool_registry=self._tool_registry,
            model_turn_provider=self._model_turn_provider,
            retrieval_hint_provider=self._retrieval_hint_provider,
            subagent_runner=self,
        )
        return await service.run_with_config(
            task=request.prompt,
            run_config=child_config,
        )


__all__ = [
    "AgentAsToolAdapter",
    "AgentAsToolExecutionError",
    "AgentAsToolRunner",
    "AgentToolInput",
    "AgentToolOutput",
    "DelegatedEvidenceRef",
]

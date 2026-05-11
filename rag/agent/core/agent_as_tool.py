from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from rag.agent.core.context import derive_child_config
from rag.agent.core.definition import AgentDefinition
from rag.agent.core.registry import AgentRegistry
from rag.agent.core.task import SubTaskNode
from rag.agent.graphs.nodes.evaluate import EvaluateDecisionProvider
from rag.agent.graphs.nodes.plan import PlanProvider
from rag.agent.state import AgentState
from rag.agent.tools.registry import ToolRegistry
from rag.agent.tools.spec import ToolSpec

if TYPE_CHECKING:
    from rag.agent.service import AgentRunResult


@dataclass(frozen=True)
class AgentToolSpec:
    tool_spec: ToolSpec
    agent_definition: AgentDefinition
    inherits_context: bool = True


class AgentAsToolRunner:
    def __init__(
        self,
        *,
        tool_registry: ToolRegistry,
        agent_registry: type[AgentRegistry] = AgentRegistry,
        evaluate_decision_provider: EvaluateDecisionProvider | None = None,
        plan_provider: PlanProvider | None = None,
    ) -> None:
        self._tool_registry = tool_registry
        self._agent_registry = agent_registry
        self._evaluate_decision_provider = evaluate_decision_provider
        self._plan_provider = plan_provider

    async def run_subtask(
        self,
        *,
        subtask: SubTaskNode,
        parent_state: AgentState,
    ) -> AgentRunResult:
        from rag.agent.service import AgentService

        parent_config = parent_state["run_config"]
        child_definition = self._agent_registry.get(subtask.agent_type)
        child_config = derive_child_config(parent_config, child_definition)
        if subtask.estimated_tokens is not None:
            child_config = replace(child_config, budget_total=subtask.estimated_tokens)

        service = AgentService(
            definition=child_definition,
            tool_registry=self._tool_registry,
            evaluate_decision_provider=self._evaluate_decision_provider,
            plan_provider=self._plan_provider,
            subagent_runner=self,
        )
        return await service.run_with_config(
            task=subtask.prompt,
            run_config=child_config,
        )

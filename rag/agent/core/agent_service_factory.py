from __future__ import annotations

from langgraph.checkpoint.base import BaseCheckpointSaver

from rag.agent.core.definition import AgentDefinition
from rag.agent.core.llm_registry import ModelRegistry
from rag.agent.graphs.nodes.evaluate import EvaluateDecisionProvider
from rag.agent.graphs.nodes.execute_subagent import SubAgentRunner
from rag.agent.graphs.nodes.plan import PlanProvider
from rag.agent.graphs.nodes.route import RouteProvider
from rag.agent.graphs.nodes.synthesize import SynthesisRunner
from rag.agent.service import AgentService
from rag.agent.tools.registry import ToolRegistry


class AgentServiceFactory:
    def __init__(
        self,
        *,
        tool_registry: ToolRegistry,
        model_registry: ModelRegistry | None = None,
        route_provider: RouteProvider | None = None,
        evaluate_decision_provider: EvaluateDecisionProvider | None = None,
        plan_provider: PlanProvider | None = None,
        checkpointer: BaseCheckpointSaver[str] | None = None,
    ) -> None:
        self._tool_registry = tool_registry
        self._model_registry = model_registry
        self._route_provider = route_provider
        self._evaluate_decision_provider = evaluate_decision_provider
        self._plan_provider = plan_provider
        self._checkpointer = checkpointer
        self._subagent_runner: SubAgentRunner | None = None
        self._synthesis_runner: SynthesisRunner | None = None

    def bind_subagent_runner(self, runner: SubAgentRunner) -> None:
        self._subagent_runner = runner

    def bind_synthesis_runner(self, runner: SynthesisRunner) -> None:
        self._synthesis_runner = runner

    def create(self, definition: AgentDefinition) -> AgentService:
        if definition.agent_type == "synthesize":
            return AgentService(
                definition=definition,
                tool_registry=self._tool_registry,
                evaluate_decision_provider=None,
                plan_provider=None,
                route_provider=None,
                subagent_runner=self._subagent_runner,
                synthesis_runner=None,
                model_registry=None,
                checkpointer=self._checkpointer,
            )
        return AgentService(
            definition=definition,
            tool_registry=self._tool_registry,
            evaluate_decision_provider=self._evaluate_decision_provider,
            plan_provider=self._plan_provider,
            route_provider=self._route_provider,
            subagent_runner=self._subagent_runner,
            synthesis_runner=self._synthesis_runner,
            model_registry=self._model_registry,
            checkpointer=self._checkpointer,
        )

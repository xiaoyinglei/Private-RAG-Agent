from __future__ import annotations

from langgraph.checkpoint.memory import MemorySaver

from rag.agent.core.definition import AgentDefinition
from rag.agent.core.llm_providers import create_default_providers
from rag.agent.core.llm_registry import ModelRegistry
from rag.agent.graphs.base import build_agent_graph
from rag.agent.graphs.nodes.evaluate import EvaluateDecisionProvider
from rag.agent.graphs.nodes.execute_subagent import SubAgentRunner
from rag.agent.graphs.nodes.plan import PlanProvider
from rag.agent.graphs.nodes.route import RouteProvider
from rag.agent.tools.registry import ToolRegistry


class AgentGraphCompiler:
    """Compile an AgentDefinition into a LangGraph runnable."""

    def __init__(
        self,
        *,
        tool_registry: ToolRegistry,
        evaluate_decision_provider: EvaluateDecisionProvider | None = None,
        plan_provider: PlanProvider | None = None,
        route_provider: RouteProvider | None = None,
        subagent_runner: SubAgentRunner | None = None,
        model_registry: ModelRegistry | None = None,
    ) -> None:
        self._tool_registry = tool_registry
        self._evaluate_decision_provider = evaluate_decision_provider
        self._plan_provider = plan_provider
        self._route_provider = route_provider
        self._subagent_runner = subagent_runner
        self._model_registry = model_registry
        self._checkpointer = MemorySaver()

    def compile(self, definition: AgentDefinition) -> object:
        missing_tools = self._missing_allowed_tools(definition)
        if missing_tools:
            raise ValueError(f"unregistered tools: {', '.join(missing_tools)}")

        route_provider = self._route_provider
        evaluate_provider = self._evaluate_decision_provider
        plan_provider = self._plan_provider

        if self._model_registry is not None and (
            route_provider is None or evaluate_provider is None or plan_provider is None
        ):
            try:
                router, evaluator, planner = create_default_providers(
                    self._model_registry, definition.model_selection
                )
            except Exception:
                pass
            else:
                if route_provider is None:
                    route_provider = router
                if evaluate_provider is None:
                    evaluate_provider = evaluator
                if plan_provider is None:
                    plan_provider = planner

        return build_agent_graph(
            definition=definition,
            tool_registry=self._tool_registry,
            evaluate_decision_provider=evaluate_provider,
            plan_provider=plan_provider,
            route_provider=route_provider,
            subagent_runner=self._subagent_runner,
            checkpointer=self._checkpointer,
        )

    def _missing_allowed_tools(self, definition: AgentDefinition) -> list[str]:
        registered_tools = {tool.name for tool in self._tool_registry.list_all()}
        missing: list[str] = []
        seen: set[str] = set()
        for tool_name in definition.allowed_tools:
            if tool_name in registered_tools or tool_name in seen:
                continue
            missing.append(tool_name)
            seen.add(tool_name)
        return missing

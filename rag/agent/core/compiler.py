from __future__ import annotations

from rag.agent.core.definition import AgentDefinition
from rag.agent.graphs.base import build_agent_graph
from rag.agent.graphs.nodes.evaluate import EvaluateDecisionProvider
from rag.agent.graphs.nodes.execute_subagent import SubAgentRunner
from rag.agent.graphs.nodes.plan import PlanProvider
from rag.agent.tools.registry import ToolRegistry


class AgentGraphCompiler:
    """Compile an AgentDefinition into a LangGraph runnable."""

    def __init__(
        self,
        *,
        tool_registry: ToolRegistry,
        query_understanding_service: object | None = None,
        evaluate_decision_provider: EvaluateDecisionProvider | None = None,
        plan_provider: PlanProvider | None = None,
        subagent_runner: SubAgentRunner | None = None,
    ) -> None:
        self._tool_registry = tool_registry
        self._query_understanding_service = query_understanding_service
        self._evaluate_decision_provider = evaluate_decision_provider
        self._plan_provider = plan_provider
        self._subagent_runner = subagent_runner

    def compile(self, definition: AgentDefinition) -> object:
        missing_tools = self._missing_allowed_tools(definition)
        if missing_tools:
            raise ValueError(f"unregistered tools: {', '.join(missing_tools)}")
        return build_agent_graph(
            definition=definition,
            tool_registry=self._tool_registry,
            query_understanding_service=self._query_understanding_service,
            evaluate_decision_provider=self._evaluate_decision_provider,
            plan_provider=self._plan_provider,
            subagent_runner=self._subagent_runner,
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

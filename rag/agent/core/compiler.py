from __future__ import annotations

from langgraph.checkpoint.base import BaseCheckpointSaver

from rag.agent.core.checkpointing import create_agent_checkpointer
from rag.agent.core.definition import AgentDefinition
from rag.agent.core.llm_providers import (
    create_default_providers,
    create_goal_contract_provider,
)
from rag.agent.core.llm_registry import ModelRegistry
from rag.agent.core.output_finalizer import (
    StructuredOutputFinalizer,
    create_model_structured_output_finalizer,
)
from rag.agent.graphs.base import build_agent_graph
from rag.agent.graphs.nodes.goal_runtime import GoalContractProvider
from rag.agent.graphs.nodes.llm_decide import ToolDecisionProvider
from rag.agent.graphs.nodes.retrieval_hint import RetrievalHintProvider
from rag.agent.graphs.nodes.synthesize import SynthesisRunner
from rag.agent.tools.registry import ToolRegistry


class GraphCompiler:
    """Compile an AgentDefinition into a LangGraph runnable."""

    def __init__(
        self,
        *,
        tool_registry: ToolRegistry,
        tool_decision_provider: ToolDecisionProvider | None = None,
        goal_contract_provider: GoalContractProvider | None = None,
        retrieval_hint_provider: RetrievalHintProvider | None = None,
        synthesis_runner: SynthesisRunner | None = None,
        output_finalizer: StructuredOutputFinalizer | None = None,
        model_registry: ModelRegistry | None = None,
        checkpointer: BaseCheckpointSaver[str] | None = None,
    ) -> None:
        self._tool_registry = tool_registry
        self._tool_decision_provider = tool_decision_provider
        self._goal_contract_provider = goal_contract_provider
        self._retrieval_hint_provider = retrieval_hint_provider
        self._synthesis_runner = synthesis_runner
        self._output_finalizer = output_finalizer
        self._model_registry = model_registry
        self._checkpointer = checkpointer or create_agent_checkpointer(None)

    def compile(self, definition: AgentDefinition) -> object:
        missing_tools = self._missing_allowed_tools(definition)
        if missing_tools:
            raise ValueError(f"unregistered tools: {', '.join(missing_tools)}")

        retrieval_hint_provider = self._retrieval_hint_provider
        tool_decision_provider = self._tool_decision_provider
        goal_contract_provider = self._goal_contract_provider
        output_finalizer = self._output_finalizer
        needs_default_retrieval_hint_provider = (
            retrieval_hint_provider is None
            and definition.model_selection.retrieval_hint_model is not None
        )

        if self._model_registry is not None and (
            needs_default_retrieval_hint_provider
            or tool_decision_provider is None
        ):
            try:
                hint_provider, decision_provider = create_default_providers(
                    self._model_registry,
                    definition.model_selection,
                    definition,
                )
            except Exception:
                pass
            else:
                if needs_default_retrieval_hint_provider:
                    retrieval_hint_provider = hint_provider
                if tool_decision_provider is None:
                    tool_decision_provider = decision_provider

        if (
            output_finalizer is None
            and definition.output_model is not None
            and self._model_registry is not None
        ):
            try:
                output_finalizer = create_model_structured_output_finalizer(
                    self._model_registry
                )
            except Exception:
                output_finalizer = None

        if (
            goal_contract_provider is None
            and self._model_registry is not None
        ):
            try:
                goal_contract_provider = create_goal_contract_provider(
                    self._model_registry,
                    definition,
                )
            except Exception:
                goal_contract_provider = None

        return build_agent_graph(
            definition=definition,
            tool_registry=self._tool_registry,
            tool_decision_provider=tool_decision_provider,
            goal_contract_provider=goal_contract_provider,
            retrieval_hint_provider=retrieval_hint_provider,
            synthesis_runner=self._synthesis_runner,
            output_finalizer=output_finalizer,
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


__all__ = ["GraphCompiler"]

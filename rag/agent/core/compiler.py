from __future__ import annotations

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver

from rag.agent.core.definition import AgentDefinition
from rag.agent.core.llm_providers import create_default_providers
from rag.agent.core.llm_registry import ModelRegistry
from rag.agent.graphs.base import build_agent_graph
from rag.agent.graphs.nodes.llm_decide import ToolDecisionProvider
from rag.agent.graphs.nodes.retrieval_hint import RetrievalHintProvider
from rag.agent.graphs.nodes.synthesize import SynthesisRunner
from rag.agent.tools.registry import ToolRegistry


class AgentGraphCompiler:
    """Compile an AgentDefinition into a LangGraph runnable."""

    def __init__(
        self,
        *,
        tool_registry: ToolRegistry,
        tool_decision_provider: ToolDecisionProvider | None = None,
        retrieval_hint_provider: RetrievalHintProvider | None = None,
        synthesis_runner: SynthesisRunner | None = None,
        model_registry: ModelRegistry | None = None,
        checkpointer: BaseCheckpointSaver[str] | None = None,
    ) -> None:
        self._tool_registry = tool_registry
        self._tool_decision_provider = tool_decision_provider
        self._retrieval_hint_provider = retrieval_hint_provider
        self._synthesis_runner = synthesis_runner
        self._model_registry = model_registry
        self._checkpointer = checkpointer or MemorySaver()

    def compile(self, definition: AgentDefinition) -> object:
        missing_tools = self._missing_allowed_tools(definition)
        if missing_tools:
            raise ValueError(f"unregistered tools: {', '.join(missing_tools)}")

        retrieval_hint_provider = self._retrieval_hint_provider
        tool_decision_provider = self._tool_decision_provider
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
                )
            except Exception:
                pass
            else:
                if needs_default_retrieval_hint_provider:
                    retrieval_hint_provider = hint_provider
                if tool_decision_provider is None:
                    tool_decision_provider = decision_provider

        return build_agent_graph(
            definition=definition,
            tool_registry=self._tool_registry,
            tool_decision_provider=tool_decision_provider,
            retrieval_hint_provider=retrieval_hint_provider,
            synthesis_runner=self._synthesis_runner,
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

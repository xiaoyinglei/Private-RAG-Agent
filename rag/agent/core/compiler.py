from __future__ import annotations

from collections.abc import Sequence

from langgraph.checkpoint.base import BaseCheckpointSaver

from rag.agent.core.checkpointing import create_agent_checkpointer
from rag.agent.core.definition import AgentDefinition
from rag.agent.core.delegation import DelegatedAgentRunner
from rag.agent.core.finalization import CompatibilitySynthesisRunner
from rag.agent.core.llm_registry import ModelRegistry
from rag.agent.core.output_finalizer import StructuredOutputFinalizer
from rag.agent.core.runtime_diagnostics import RuntimeDiagnostic
from rag.agent.core.runtime_ports import (
    RetrievalHintProvider,
    ToolDecisionProvider,
)
from rag.agent.graphs.base import build_outer_agent_graph
from rag.agent.loop.runtime import ModelTurnProvider
from rag.agent.service import AgentService
from rag.agent.tools.registry import ToolRegistry


class GraphCompiler:
    """Compile one AgentLoop invocation as a coarse outer LangGraph node."""

    def __init__(
        self,
        *,
        tool_registry: ToolRegistry,
        model_turn_provider: ModelTurnProvider | None = None,
        tool_decision_provider: ToolDecisionProvider | None = None,
        retrieval_hint_provider: RetrievalHintProvider | None = None,
        subagent_runner: DelegatedAgentRunner | None = None,
        synthesis_runner: CompatibilitySynthesisRunner | None = None,
        output_finalizer: StructuredOutputFinalizer | None = None,
        model_registry: ModelRegistry | None = None,
        checkpointer: BaseCheckpointSaver[str] | None = None,
        runtime_diagnostics: Sequence[RuntimeDiagnostic] = (),
    ) -> None:
        self._tool_registry = tool_registry
        self._model_turn_provider = model_turn_provider
        self._tool_decision_provider = tool_decision_provider
        self._retrieval_hint_provider = retrieval_hint_provider
        self._subagent_runner = subagent_runner
        self._synthesis_runner = synthesis_runner
        self._output_finalizer = output_finalizer
        self._model_registry = model_registry
        self._checkpointer = (
            checkpointer or create_agent_checkpointer(None)
        )
        self._runtime_diagnostics = tuple(runtime_diagnostics)

    def compile(self, definition: AgentDefinition) -> object:
        missing_tools = self._missing_allowed_tools(definition)
        if missing_tools:
            raise ValueError(
                f"unregistered tools: {', '.join(missing_tools)}"
            )

        service = AgentService(
            definition=definition,
            tool_registry=self._tool_registry,
            model_turn_provider=self._model_turn_provider,
            tool_decision_provider=self._tool_decision_provider,
            retrieval_hint_provider=self._retrieval_hint_provider,
            subagent_runner=self._subagent_runner,
            synthesis_runner=self._synthesis_runner,
            output_finalizer=self._output_finalizer,
            model_registry=self._model_registry,
            checkpointer=self._checkpointer,
            runtime_diagnostics=self._runtime_diagnostics,
        )
        return build_outer_agent_graph(
            run_kernel=service.run,
            checkpointer=self._checkpointer,
        )

    def _missing_allowed_tools(
        self,
        definition: AgentDefinition,
    ) -> list[str]:
        registered_tools = {
            tool.name for tool in self._tool_registry.list_all()
        }
        missing: list[str] = []
        seen: set[str] = set()
        for tool_name in definition.allowed_tools:
            if tool_name in registered_tools or tool_name in seen:
                continue
            missing.append(tool_name)
            seen.add(tool_name)
        return missing


__all__ = ["GraphCompiler"]

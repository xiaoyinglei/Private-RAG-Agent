from __future__ import annotations

from collections.abc import Sequence

from langgraph.checkpoint.base import BaseCheckpointSaver

from rag.agent.core.definition import AgentDefinition
from rag.agent.core.delegation import DelegatedAgentRunner
from rag.agent.core.llm_registry import ModelRegistry
from rag.agent.core.runtime_diagnostics import RuntimeDiagnostic
from rag.agent.graphs.nodes.goal_runtime import GoalContractProvider
from rag.agent.graphs.nodes.llm_decide import ToolDecisionProvider
from rag.agent.graphs.nodes.retrieval_hint import RetrievalHintProvider
from rag.agent.graphs.nodes.synthesize import SynthesisRunner
from rag.agent.loop.runtime import ModelTurnProvider
from rag.agent.service import AgentService
from rag.agent.tools.registry import ToolRegistry


class AgentServiceFactory:
    def __init__(
        self,
        *,
        tool_registry: ToolRegistry,
        model_turn_provider: ModelTurnProvider | None = None,
        model_registry: ModelRegistry | None = None,
        retrieval_hint_provider: RetrievalHintProvider | None = None,
        tool_decision_provider: ToolDecisionProvider | None = None,
        goal_contract_provider: GoalContractProvider | None = None,
        checkpointer: BaseCheckpointSaver[str] | None = None,
        runtime_diagnostics: Sequence[RuntimeDiagnostic] = (),
    ) -> None:
        self._tool_registry = tool_registry
        self._model_turn_provider = model_turn_provider
        self._model_registry = model_registry
        self._retrieval_hint_provider = retrieval_hint_provider
        self._tool_decision_provider = tool_decision_provider
        self._goal_contract_provider = goal_contract_provider
        self._checkpointer = checkpointer
        self._runtime_diagnostics = tuple(runtime_diagnostics)
        self._subagent_runner: DelegatedAgentRunner | None = None
        self._synthesis_runner: SynthesisRunner | None = None

    def bind_subagent_runner(self, runner: DelegatedAgentRunner) -> None:
        self._subagent_runner = runner

    def bind_synthesis_runner(self, runner: SynthesisRunner) -> None:
        self._synthesis_runner = runner

    def create(self, definition: AgentDefinition) -> AgentService:
        if definition.agent_type == "synthesize":
            return AgentService(
                definition=definition,
                tool_registry=self._tool_registry,
                model_turn_provider=self._model_turn_provider,
                tool_decision_provider=None,
                goal_contract_provider=self._goal_contract_provider,
                retrieval_hint_provider=None,
                subagent_runner=self._subagent_runner,
                synthesis_runner=None,
                model_registry=self._model_registry,
                checkpointer=self._checkpointer,
                runtime_diagnostics=self._runtime_diagnostics,
            )
        return AgentService(
            definition=definition,
            tool_registry=self._tool_registry,
            model_turn_provider=self._model_turn_provider,
            tool_decision_provider=self._tool_decision_provider,
            goal_contract_provider=self._goal_contract_provider,
            retrieval_hint_provider=self._retrieval_hint_provider,
            subagent_runner=self._subagent_runner,
            synthesis_runner=self._synthesis_runner,
            model_registry=self._model_registry,
            checkpointer=self._checkpointer,
            runtime_diagnostics=self._runtime_diagnostics,
        )

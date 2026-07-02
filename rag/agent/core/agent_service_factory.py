from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from langgraph.checkpoint.base import BaseCheckpointSaver

from rag.agent.core.definition import AgentRuntimePolicy
from rag.agent.core.delegation import DelegatedAgentRunner
from rag.agent.core.llm_registry import ModelResolver
from rag.agent.core.runtime_diagnostics import RuntimeDiagnostic
from rag.agent.core.runtime_ports import RetrievalHintProvider
from rag.agent.loop.runtime import ModelTurnProvider
from rag.agent.service import AgentService
from rag.agent.tools.registry import ToolRegistry

if TYPE_CHECKING:
    from rag.agent.skills.catalog import SkillCatalog


class AgentServiceFactory:
    def __init__(
        self,
        *,
        tool_registry: ToolRegistry,
        model_turn_provider: ModelTurnProvider | None = None,
        model_registry: ModelResolver | None = None,
        retrieval_hint_provider: RetrievalHintProvider | None = None,
        checkpointer: BaseCheckpointSaver[str] | None = None,
        runtime_diagnostics: Sequence[RuntimeDiagnostic] = (),
        skill_catalog: SkillCatalog | None = None,
        strict_model_provider: bool = False,
    ) -> None:
        self._tool_registry = tool_registry
        self._model_turn_provider = model_turn_provider
        self._model_registry = model_registry
        self._retrieval_hint_provider = retrieval_hint_provider
        self._checkpointer = checkpointer
        self._runtime_diagnostics = tuple(runtime_diagnostics)
        self._subagent_runner: DelegatedAgentRunner | None = None
        self._skill_catalog = skill_catalog
        self._strict_model_provider = strict_model_provider

    def bind_subagent_runner(self, runner: DelegatedAgentRunner) -> None:
        self._subagent_runner = runner

    def create(self, definition: AgentRuntimePolicy) -> AgentService:
        return AgentService(
            definition=definition,
            tool_registry=self._tool_registry,
            model_turn_provider=self._model_turn_provider,
            retrieval_hint_provider=self._retrieval_hint_provider,
            subagent_runner=self._subagent_runner,
            model_registry=self._model_registry,
            checkpointer=self._checkpointer,
            runtime_diagnostics=self._runtime_diagnostics,
            skill_catalog=self._skill_catalog,
            strict_model_provider=self._strict_model_provider,
        )

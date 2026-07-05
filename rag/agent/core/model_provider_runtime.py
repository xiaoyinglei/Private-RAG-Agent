"""Model turn provider resolution for AgentService."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from rag.agent.capabilities.catalog import ToolCatalog
from rag.agent.capabilities.context import deferred_store_var
from rag.agent.core.definition import AgentRuntimePolicy
from rag.agent.core.llm_providers import create_loop_model_turn_provider
from rag.agent.core.llm_registry import ModelResolver
from rag.agent.core.runtime_diagnostics import RuntimeDiagnostic
from rag.agent.loop.runtime import ModelTurnProvider
from rag.agent.loop.state import (
    LoopState,
    ModelTurnDraft,
    append_loop_diagnostic,
)
from rag.agent.tools.registry import ToolRegistry


class ResultDrivenModelTurnProvider:
    """Minimal fallback provider for explicit compatibility paths."""

    async def next_turn(
        self,
        state: LoopState,
        *,
        definition: AgentRuntimePolicy,
        budget_remaining: int,
    ) -> ModelTurnDraft:
        del definition, budget_remaining
        if state["finish_state"].feedback:
            return ModelTurnDraft(
                action="pause",
                pause_reason=(
                    "No model turn provider is available to address "
                    "stop-hook feedback."
                ),
            )
        if state["tool_results"]:
            latest = state["tool_results"][-1]
            if latest.error is not None:
                return ModelTurnDraft(action="finish")
            if latest.output is not None:
                text = getattr(latest.output, "text", None) or getattr(
                    latest.output,
                    "result",
                    None,
                )
                if text:
                    return ModelTurnDraft(
                        action="finish",
                        final_answer=str(text),
                    )
            return ModelTurnDraft(
                action="pause",
                pause_reason="Tool execution produced no extractable answer.",
            )
        return ModelTurnDraft(
            action="pause",
            pause_reason="No model turn provider is configured.",
        )


@dataclass
class ModelProviderResolver:
    """Build the per-loop model provider from service-scoped dependencies."""

    model_turn_provider: ModelTurnProvider | None
    model_registry: ModelResolver | None
    policy: AgentRuntimePolicy
    base_tool_registry: ToolRegistry
    catalog: ToolCatalog
    strict_model_provider: bool = True
    stream_sink: Any = None
    skill_context_provider: Callable[[LoopState], str] | None = None

    def resolve(
        self,
        state: LoopState | None,
        *,
        tool_registry: ToolRegistry | None = None,
    ) -> ModelTurnProvider:
        if self.model_turn_provider is not None:
            return self.model_turn_provider
        if self.model_registry is not None:
            try:
                store = deferred_store_var.get(None)
                if store is None:
                    raise RuntimeError(
                        "DeferredToolStore is not bound — AgentLoop must set "
                        "deferred_store_var before creating provider"
                    )
                effective_registry = tool_registry or self.base_tool_registry
                formatter_resolver = (
                    (lambda name: tool_registry.get_formatter(name))
                    if tool_registry is not None
                    else None
                )
                return create_loop_model_turn_provider(
                    self.model_registry,
                    self.policy.model_selection,
                    tool_registry=effective_registry,
                    definition=self.policy,
                    catalog=self.catalog,
                    deferred_store=store,
                    stream_sink=self.stream_sink,
                    formatter_resolver=formatter_resolver,
                    skill_context_provider=self.skill_context_provider,
                )
            except Exception as exc:
                if self.strict_model_provider:
                    raise
                if state is not None:
                    append_loop_diagnostic(
                        state,
                        RuntimeDiagnostic.from_exception(
                            code="default_providers_initialization_failed",
                            component="model_providers",
                            error=exc,
                        ),
                    )
        return ResultDrivenModelTurnProvider()


__all__ = ["ModelProviderResolver", "ResultDrivenModelTurnProvider"]

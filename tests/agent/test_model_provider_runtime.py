from __future__ import annotations

import pytest

from rag.agent.builtin.generic import GENERIC_AGENT
from rag.agent.builtin_registry import create_builtin_tool_registry
from rag.agent.capabilities.catalog import DeferredToolStore, ToolCatalog
from rag.agent.capabilities.context import deferred_store_var
from rag.agent.core.context import AgentRunConfig
from rag.agent.core.model_provider_runtime import (
    ModelProviderResolver,
    ResultDrivenModelTurnProvider,
)
from rag.agent.loop.state import create_loop_state
from rag.schema.runtime import AccessPolicy


class _FailingModelRegistry:
    default_model = "broken"
    fallback_model = None

    def resolve_for_node(self, *, node_model: str | None, node_name: str) -> object:
        del node_model, node_name
        raise RuntimeError("provider unavailable")


def _state():
    return create_loop_state(
        task="Explain policy",
        run_config=AgentRunConfig(
            run_id="model-provider-runtime",
            thread_id="model-provider-runtime",
            max_depth=1,
            access_policy=AccessPolicy.default(),
        ),
    )


@pytest.mark.anyio
async def test_model_provider_resolver_records_diagnostic_when_non_strict() -> None:
    state = _state()
    token = deferred_store_var.set(
        DeferredToolStore(max_active=GENERIC_AGENT.max_active_deferred_tools),
    )
    try:
        provider = ModelProviderResolver(
            model_turn_provider=None,
            model_registry=_FailingModelRegistry(),
            policy=GENERIC_AGENT,
            base_tool_registry=create_builtin_tool_registry(),
            catalog=ToolCatalog(),
            strict_model_provider=False,
        ).resolve(
            state,
            tool_registry=create_builtin_tool_registry(),
        )
    finally:
        deferred_store_var.reset(token)

    assert isinstance(provider, ResultDrivenModelTurnProvider)
    assert state["runtime_diagnostics"][0].code == "default_providers_initialization_failed"


@pytest.mark.anyio
async def test_model_provider_resolver_raises_when_strict() -> None:
    state = _state()
    token = deferred_store_var.set(
        DeferredToolStore(max_active=GENERIC_AGENT.max_active_deferred_tools),
    )
    try:
        resolver = ModelProviderResolver(
            model_turn_provider=None,
            model_registry=_FailingModelRegistry(),
            policy=GENERIC_AGENT,
            base_tool_registry=create_builtin_tool_registry(),
            catalog=ToolCatalog(),
            strict_model_provider=True,
        )
        with pytest.raises(RuntimeError, match="provider unavailable"):
            resolver.resolve(
                state,
                tool_registry=create_builtin_tool_registry(),
            )
    finally:
        deferred_store_var.reset(token)


def test_model_provider_resolver_prefers_explicit_provider() -> None:
    provider = ResultDrivenModelTurnProvider()

    resolved = ModelProviderResolver(
        model_turn_provider=provider,
        model_registry=object(),
        policy=GENERIC_AGENT,
        base_tool_registry=create_builtin_tool_registry(),
        catalog=ToolCatalog(),
        strict_model_provider=True,
    ).resolve(
        None,
        tool_registry=create_builtin_tool_registry(),
    )

    assert resolved is provider

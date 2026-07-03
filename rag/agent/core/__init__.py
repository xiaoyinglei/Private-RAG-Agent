"""Lazy public exports for agent core contracts."""

from __future__ import annotations

from typing import Any

_EXPORTS: dict[str, tuple[str, str]] = {
    "AgentRuntimePolicy": ("rag.agent.core.definition", "AgentRuntimePolicy"),
    "GraphCompiler": ("rag.agent.core.compiler", "GraphCompiler"),
    "AgentRegistry": ("rag.agent.core.registry", "AgentRegistry"),
    "AgentRunConfig": ("rag.agent.core.context", "AgentRunConfig"),
    "AgentServiceFactory": ("rag.agent.core.agent_service_factory", "AgentServiceFactory"),
    "AgentAsToolRunner": ("rag.agent.core.agent_as_tool", "AgentAsToolRunner"),
    "AgentDelegationRequest": ("rag.agent.core.delegation", "AgentDelegationRequest"),
    "AgentToolSpec": ("rag.agent.core.agent_as_tool", "AgentToolSpec"),
    "BuiltinSubAgentRunner": ("rag.agent.core.subagent_runner", "BuiltinSubAgentRunner"),
    "aclose_agent_checkpointer": ("rag.agent.core.checkpointing", "aclose_agent_checkpointer"),
    "create_agent_checkpointer": ("rag.agent.core.checkpointing", "create_agent_checkpointer"),
    "ModelSelectionPolicy": ("rag.agent.core.definition", "ModelSelectionPolicy"),
    "RunRegistry": ("rag.agent.core.context", "RunRegistry"),
    "DelegatedAgentRunner": ("rag.agent.core.delegation", "DelegatedAgentRunner"),
    "ToolPolicy": ("rag.agent.core.definition", "ToolPolicy"),
    "derive_child_config": ("rag.agent.core.context", "derive_child_config"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    try:
        module_name, attr_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module 'rag.agent.core' has no attribute {name!r}") from exc
    from importlib import import_module

    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value

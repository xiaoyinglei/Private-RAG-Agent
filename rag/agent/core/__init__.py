"""Lazy public exports for agent core contracts."""

from __future__ import annotations

_EXPORTS: dict[str, tuple[str, str]] = {
    "AgentRuntimePolicy": ("rag.agent.core.definition", "AgentRuntimePolicy"),
    "AgentRunConfig": ("rag.agent.core.context", "AgentRunConfig"),
    "aclose_agent_checkpointer": ("rag.agent.core.checkpointing", "aclose_agent_checkpointer"),
    "create_agent_checkpointer": ("rag.agent.core.checkpointing", "create_agent_checkpointer"),
    "ModelSelectionPolicy": ("rag.agent.core.definition", "ModelSelectionPolicy"),
    "TurnRegistry": ("rag.agent.core.context", "TurnRegistry"),
    "ToolPolicy": ("rag.agent.core.definition", "ToolPolicy"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> object:
    try:
        module_name, attr_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module 'rag.agent.core' has no attribute {name!r}") from exc
    from importlib import import_module

    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value

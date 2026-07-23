"""Lazy public exports for the agent orchestration package."""

from __future__ import annotations

from importlib import import_module

_EXPORTS: dict[str, tuple[str, str]] = {
    "AgentRuntimePolicy": ("rag.agent.core.definition", "AgentRuntimePolicy"),
    "AgentRunConfig": ("rag.agent.core.context", "AgentRunConfig"),
    "AgentRunRequest": ("rag.agent.service", "AgentRunRequest"),
    "AgentRunResult": ("rag.agent.service", "AgentRunResult"),
    "AgentService": ("rag.agent.service", "AgentService"),
    "AgentState": ("rag.agent.loop.state", "LoopState"),
    "ContextBudgetSnapshot": ("rag.agent.memory.models", "ContextBudgetSnapshot"),
    "ExtractedFact": ("rag.agent.memory.models", "ExtractedFact"),
    "InterruptBehavior": ("rag.agent.tools.tool", "InterruptBehavior"),
    "MemoryPolicy": ("rag.agent.memory.models", "MemoryPolicy"),
    "ModelSelectionPolicy": ("rag.agent.core.definition", "ModelSelectionPolicy"),
    "TurnRegistry": ("rag.agent.core.context", "TurnRegistry"),
    "ToolCallPlan": ("rag.agent.core.turn_contracts", "ToolCallPlan"),
    "ToolPolicy": ("rag.agent.core.definition", "ToolPolicy"),
    "Tool": ("rag.agent.tools.tool", "Tool"),
    "ToolRegistry": ("rag.agent.tools.registry", "ToolRegistry"),
    "ToolResult": ("rag.agent.tools.tool", "ToolResult"),
    "WorkingSummary": ("rag.agent.memory.models", "WorkingSummary"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> object:
    try:
        module_name, attr_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module 'rag.agent' has no attribute {name!r}") from exc
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value

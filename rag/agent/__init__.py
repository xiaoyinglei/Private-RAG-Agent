"""Lazy public exports for the agent orchestration package."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS: dict[str, tuple[str, str]] = {
    "AgentRuntimePolicy": ("rag.agent.core.definition", "AgentRuntimePolicy"),
    "ExecutionCategory": ("rag.agent.tools.spec", "ExecutionCategory"),
    "GraphCompiler": ("rag.agent.core.compiler", "GraphCompiler"),
    "AgentRegistry": ("rag.agent.core.registry", "AgentRegistry"),
    "AgentRunConfig": ("rag.agent.core.context", "AgentRunConfig"),
    "AgentRunRequest": ("rag.agent.service", "AgentRunRequest"),
    "AgentRunResult": ("rag.agent.service", "AgentRunResult"),
    "AgentServiceFactory": ("rag.agent.core.agent_service_factory", "AgentServiceFactory"),
    "AgentService": ("rag.agent.service", "AgentService"),
    "AgentState": ("rag.agent.loop.state", "LoopState"),
    "AgentPlan": ("rag.agent.planning", "AgentPlan"),
    "AgentAsToolRunner": ("rag.agent.core.agent_as_tool", "AgentAsToolRunner"),
    "AgentDelegationRequest": ("rag.agent.core.delegation", "AgentDelegationRequest"),
    "AgentToolSpec": ("rag.agent.core.agent_as_tool", "AgentToolSpec"),
    "BuiltinSubAgentRunner": ("rag.agent.core.subagent_runner", "BuiltinSubAgentRunner"),
    "ContextBudgetSnapshot": ("rag.agent.memory.models", "ContextBudgetSnapshot"),
    "ExtractedFact": ("rag.agent.memory.models", "ExtractedFact"),
    "InterruptBehavior": ("rag.agent.tools.spec", "InterruptBehavior"),
    "MemoryPolicy": ("rag.agent.memory.models", "MemoryPolicy"),
    "ModelSelectionPolicy": ("rag.agent.core.definition", "ModelSelectionPolicy"),
    "PlanEvent": ("rag.agent.planning", "PlanEvent"),
    "PlanStep": ("rag.agent.planning", "PlanStep"),
    "PlanStepPatch": ("rag.agent.planning", "PlanStepPatch"),
    "PlanTracker": ("rag.agent.planning", "PlanTracker"),
    "PlanUpdate": ("rag.agent.planning", "PlanUpdate"),
    "RiskLevel": ("rag.agent.tools.spec", "RiskLevel"),
    "RunRegistry": ("rag.agent.core.context", "RunRegistry"),
    "DelegatedAgentRunner": ("rag.agent.core.delegation", "DelegatedAgentRunner"),
    "ToolCallPlan": ("rag.agent.core.turn_contracts", "ToolCallPlan"),
    "ToolError": ("rag.agent.tools.spec", "ToolError"),
    "ToolPermissions": ("rag.agent.tools.spec", "ToolPermissions"),
    "ToolPolicy": ("rag.agent.core.definition", "ToolPolicy"),
    "ToolRegistry": ("rag.agent.tools.registry", "ToolRegistry"),
    "ToolResult": ("rag.agent.tools.spec", "ToolResult"),
    "ToolSpec": ("rag.agent.tools.spec", "ToolSpec"),
    "WorkingSummary": ("rag.agent.memory.models", "WorkingSummary"),
    "derive_child_config": ("rag.agent.core.context", "derive_child_config"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    try:
        module_name, attr_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module 'rag.agent' has no attribute {name!r}") from exc
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value

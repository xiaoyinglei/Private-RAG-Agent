"""Core RAG library public exports."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

__all__ = [
    "AgentRuntimePolicy",
    "AgentRunConfig",
    "AgentRunRequest",
    "AgentService",
    "AgentState",
    "ToolRegistry",
    "ToolSpec",
    "AssemblyConfig",
    "AssemblyDiagnostics",
    "AssemblyOverrides",
    "AssemblyRequest",
    "CapabilityAssemblyService",
    "CapabilityRequirements",
    "RAGRuntime",
    "StorageComponentConfig",
    "StorageConfig",
]

if TYPE_CHECKING:
    from rag.agent import AgentRuntimePolicy, AgentRunConfig, AgentRunRequest, AgentService, AgentState, ToolRegistry, ToolSpec  # noqa: F401
    from rag.assembly import AssemblyConfig, AssemblyDiagnostics, AssemblyOverrides, AssemblyRequest, CapabilityAssemblyService, CapabilityRequirements  # noqa: F401, E501
    from rag.runtime import RAGRuntime  # noqa: F401
    from rag.storage import StorageComponentConfig, StorageConfig  # noqa: F401

_EXPORTS = {
    "AgentRuntimePolicy": ("rag.agent", "AgentRuntimePolicy"),
    "AgentRunConfig": ("rag.agent", "AgentRunConfig"),
    "AgentRunRequest": ("rag.agent", "AgentRunRequest"),
    "AgentService": ("rag.agent", "AgentService"),
    "AgentState": ("rag.agent", "AgentState"),
    "ToolRegistry": ("rag.agent", "ToolRegistry"),
    "ToolSpec": ("rag.agent", "ToolSpec"),
    "AssemblyConfig": ("rag.assembly", "AssemblyConfig"),
    "AssemblyDiagnostics": ("rag.assembly", "AssemblyDiagnostics"),
    "AssemblyOverrides": ("rag.assembly", "AssemblyOverrides"),
    "AssemblyRequest": ("rag.assembly", "AssemblyRequest"),
    "CapabilityAssemblyService": ("rag.assembly", "CapabilityAssemblyService"),
    "CapabilityRequirements": ("rag.assembly", "CapabilityRequirements"),
    "RAGRuntime": ("rag.runtime", "RAGRuntime"),
    "StorageComponentConfig": ("rag.storage", "StorageComponentConfig"),
    "StorageConfig": ("rag.storage", "StorageConfig"),
}


def __getattr__(name: str) -> object:
    export = _EXPORTS.get(name)
    if export is None:
        raise AttributeError(f"module 'rag' has no attribute {name!r}")
    module_name, attr_name = export
    module = import_module(module_name)
    return getattr(module, attr_name)

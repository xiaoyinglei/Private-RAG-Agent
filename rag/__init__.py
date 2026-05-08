"""Core RAG library public exports."""

from __future__ import annotations

from importlib import import_module

__all__ = [
    "AgentDefinition",
    "AgentRunConfig",
    "AgentState",
    "ToolRegistry",
    "ToolSpec",
    "AssemblyConfig",
    "AssemblyDiagnostics",
    "AssemblyOverrides",
    "AssemblyProfileSpec",
    "AssemblyRequest",
    "CapabilityAssemblyService",
    "CapabilityRequirements",
    "RAGRuntime",
    "StorageComponentConfig",
    "StorageConfig",
]

_EXPORTS = {
    "AgentDefinition": ("rag.agent", "AgentDefinition"),
    "AgentRunConfig": ("rag.agent", "AgentRunConfig"),
    "AgentState": ("rag.agent", "AgentState"),
    "ToolRegistry": ("rag.agent", "ToolRegistry"),
    "ToolSpec": ("rag.agent", "ToolSpec"),
    "AssemblyConfig": ("rag.assembly", "AssemblyConfig"),
    "AssemblyDiagnostics": ("rag.assembly", "AssemblyDiagnostics"),
    "AssemblyOverrides": ("rag.assembly", "AssemblyOverrides"),
    "AssemblyProfileSpec": ("rag.assembly", "AssemblyProfileSpec"),
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

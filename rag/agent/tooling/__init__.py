"""Claude-like tool boundary for the main agent path."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rag.agent.tooling.discovery import DiscoveryPolicy, ToolDiscoveryState
    from rag.agent.tooling.executor import (
        CanUseToolResult,
        ToolExecutor,
        ToolExecutorLoopAdapter,
        canUseTool,
    )
    from rag.agent.tooling.registry import ToolRegistry, install_minimal_workspace_tools
    from rag.agent.tooling.request_builder import ModelRequest, ModelRequestBuilder
    from rag.agent.tooling.spec import (
        ToolCall,
        ToolDomain,
        ToolExposure,
        ToolResult,
        ToolRisk,
        ToolSpec,
    )
    from rag.agent.tooling.surface import (
        ProviderCapability,
        ToolSurfaceDecision,
        ToolSurfacePolicy,
        ToolSurfaceRequest,
    )
    from rag.agent.tooling.trace import ModelRequestTrace, ToolExecutionTrace

__all__ = [
    "DiscoveryPolicy",
    "CanUseToolResult",
    "ModelRequest",
    "ModelRequestBuilder",
    "ModelRequestTrace",
    "ProviderCapability",
    "ToolDiscoveryState",
    "ToolCall",
    "ToolDomain",
    "ToolExecutionTrace",
    "ToolExecutor",
    "ToolExecutorLoopAdapter",
    "ToolExposure",
    "ToolRegistry",
    "ToolResult",
    "ToolRisk",
    "ToolSpec",
    "ToolSurfaceDecision",
    "ToolSurfacePolicy",
    "ToolSurfaceRequest",
    "install_minimal_workspace_tools",
    "canUseTool",
]

_EXPORTS = {
    "CanUseToolResult": ("rag.agent.tooling.executor", "CanUseToolResult"),
    "DiscoveryPolicy": ("rag.agent.tooling.discovery", "DiscoveryPolicy"),
    "ToolDiscoveryState": ("rag.agent.tooling.discovery", "ToolDiscoveryState"),
    "ToolExecutor": ("rag.agent.tooling.executor", "ToolExecutor"),
    "ToolExecutorLoopAdapter": ("rag.agent.tooling.executor", "ToolExecutorLoopAdapter"),
    "canUseTool": ("rag.agent.tooling.executor", "canUseTool"),
    "ToolRegistry": ("rag.agent.tooling.registry", "ToolRegistry"),
    "install_minimal_workspace_tools": (
        "rag.agent.tooling.registry",
        "install_minimal_workspace_tools",
    ),
    "ModelRequest": ("rag.agent.tooling.request_builder", "ModelRequest"),
    "ModelRequestBuilder": ("rag.agent.tooling.request_builder", "ModelRequestBuilder"),
    "ToolCall": ("rag.agent.tooling.spec", "ToolCall"),
    "ToolDomain": ("rag.agent.tooling.spec", "ToolDomain"),
    "ToolExposure": ("rag.agent.tooling.spec", "ToolExposure"),
    "ToolResult": ("rag.agent.tooling.spec", "ToolResult"),
    "ToolRisk": ("rag.agent.tooling.spec", "ToolRisk"),
    "ToolSpec": ("rag.agent.tooling.spec", "ToolSpec"),
    "ProviderCapability": ("rag.agent.tooling.surface", "ProviderCapability"),
    "ToolSurfaceDecision": ("rag.agent.tooling.surface", "ToolSurfaceDecision"),
    "ToolSurfacePolicy": ("rag.agent.tooling.surface", "ToolSurfacePolicy"),
    "ToolSurfaceRequest": ("rag.agent.tooling.surface", "ToolSurfaceRequest"),
    "ModelRequestTrace": ("rag.agent.tooling.trace", "ModelRequestTrace"),
    "ToolExecutionTrace": ("rag.agent.tooling.trace", "ToolExecutionTrace"),
}


def __getattr__(name: str) -> object:
    export = _EXPORTS.get(name)
    if export is None:
        raise AttributeError(f"module 'rag.agent.tooling' has no attribute {name!r}")
    module_name, attr_name = export
    module = import_module(module_name)
    return getattr(module, attr_name)

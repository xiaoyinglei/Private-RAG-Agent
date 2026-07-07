"""Structured discovery state for the new tool surface path."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

from rag.agent.tooling.surface import ToolSurfaceRequest

if TYPE_CHECKING:
    from rag.agent.tooling.registry import ToolRegistry


class ToolDiscoveryState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    discovered_tool_names: list[str] = Field(default_factory=list)


class DiscoveryPolicy:
    """Apply structured discovery state without inspecting task text."""

    def apply(
        self,
        registry: ToolRegistry,
        request: ToolSurfaceRequest,
        state: ToolDiscoveryState | None = None,
    ) -> ToolSurfaceRequest:
        discovery_state = state or ToolDiscoveryState()
        if (
            request.force_empty
            or not request.allow_discovery_tools
            or not discovery_state.discovered_tool_names
        ):
            return request

        disabled = set(request.disabled_tool_names)
        requested: list[str] = []
        seen: set[str] = set()
        for name in [*request.requested_tool_names, *discovery_state.discovered_tool_names]:
            if name in seen or name in disabled:
                continue
            seen.add(name)
            if registry.get(name) is None:
                continue
            requested.append(name)

        return request.model_copy(update={"requested_tool_names": requested})

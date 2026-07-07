"""Per-request tool schema surface decisions."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field

from rag.agent.tooling.spec import ToolDomain, ToolExposure, ToolRisk, ToolSpec

if TYPE_CHECKING:
    from rag.agent.tooling.registry import ToolRegistry


class ToolSurfaceRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    requested_tool_names: list[str] = Field(default_factory=list)
    disabled_tool_names: list[str] = Field(default_factory=list)
    force_empty: bool = False
    allow_write_tools: bool = False
    allow_execute_tools: bool = False
    allow_discovery_tools: bool = False


class ProviderCapability(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: str | None = None
    model: str | None = None
    supports_tools: bool = True
    supports_tool_choice: bool = True


class ToolSurfaceDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    visible_tools: list[ToolSpec] = Field(default_factory=list)
    hidden_tools: list[str] = Field(default_factory=list)
    sent_schema_names: list[str] = Field(default_factory=list)
    tool_choice: str | dict[str, Any] | None = "auto"


class ToolSurfacePolicy:
    """Decide which installed tool schemas are sent for one model request."""

    def decide(
        self,
        registry: ToolRegistry,
        request: ToolSurfaceRequest,
        *,
        provider_capability: ProviderCapability | None = None,
    ) -> ToolSurfaceDecision:
        capability = provider_capability or ProviderCapability()
        if request.force_empty or not capability.supports_tools:
            return ToolSurfaceDecision(
                visible_tools=[],
                hidden_tools=sorted(spec.name for spec in registry.list_specs()),
                sent_schema_names=[],
                tool_choice="none" if capability.supports_tool_choice else None,
            )

        disabled = set(request.disabled_tool_names)
        visible: list[ToolSpec] = []
        seen: set[str] = set()

        for name in request.requested_tool_names:
            if name in seen or name in disabled:
                continue
            seen.add(name)
            spec = registry.get(name)
            if spec is None or not self._can_surface(spec, request):
                continue
            visible.append(spec)

        sent_schema_names = [spec.name for spec in visible]
        hidden_tools = sorted(
            spec.name for spec in registry.list_specs() if spec.name not in sent_schema_names
        )

        return ToolSurfaceDecision(
            visible_tools=visible,
            hidden_tools=hidden_tools,
            sent_schema_names=sent_schema_names,
            tool_choice=(
                ("auto" if visible else "none")
                if capability.supports_tool_choice
                else None
            ),
        )

    def _can_surface(self, spec: ToolSpec, request: ToolSurfaceRequest) -> bool:
        if spec.exposure == ToolExposure.INTERNAL:
            return False
        if spec.exposure == ToolExposure.DEFERRED and not request.allow_discovery_tools:
            return False
        if spec.domain == ToolDomain.DISCOVERY and not request.allow_discovery_tools:
            return False
        if spec.risk == ToolRisk.WRITE and not request.allow_write_tools:
            return False
        if spec.risk == ToolRisk.EXECUTE and not request.allow_execute_tools:
            return False
        if spec.risk == ToolRisk.NETWORK and not request.allow_discovery_tools:
            return False
        if spec.risk == ToolRisk.DESTRUCTIVE:
            return False
        return True

"""Provider request builder for the new tool schema surface."""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from rag.agent.tooling.surface import ProviderCapability, ToolSurfaceDecision
from rag.agent.tooling.trace import ModelRequestTrace


class ModelRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: str
    model: str
    payload: dict[str, Any]
    sent_schema_names: list[str] = Field(default_factory=list)
    trace: ModelRequestTrace


class ModelRequestBuilder:
    """Build OpenAI/Groq-compatible request payloads from a surface decision."""

    def __init__(
        self,
        *,
        provider: str,
        model: str,
        provider_capability: ProviderCapability | None = None,
    ) -> None:
        self._provider = provider
        self._model = model
        self._provider_capability = provider_capability or ProviderCapability()

    def build(
        self,
        *,
        messages: list[dict[str, Any]],
        surface: ToolSurfaceDecision,
    ) -> ModelRequest:
        tools = [
            {
                "type": "function",
                "function": {
                    "name": spec.name,
                    "description": spec.description,
                    "parameters": spec.input_schema,
                },
            }
            for spec in surface.visible_tools
        ]
        schema_bytes = _schema_bytes(tools)
        tool_choice = surface.tool_choice
        if not tools and tool_choice is None:
            tool_choice = "none"

        payload: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "tools": tools,
        }
        if self._provider_capability.supports_tool_choice:
            payload["tool_choice"] = tool_choice
        trace = ModelRequestTrace(
            provider=self._provider,
            model=self._model,
            visible_tools=[spec.name for spec in surface.visible_tools],
            hidden_tools=surface.hidden_tools,
            schema_bytes=schema_bytes,
            tool_choice=tool_choice
            if self._provider_capability.supports_tool_choice
            else None,
            latency_ms=0.0,
        )
        return ModelRequest(
            provider=self._provider,
            model=self._model,
            payload=payload,
            sent_schema_names=list(surface.sent_schema_names),
            trace=trace,
        )


def _schema_bytes(tools: list[dict[str, Any]]) -> int:
    if not tools:
        return 0
    encoded = json.dumps(
        tools,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return len(encoded)

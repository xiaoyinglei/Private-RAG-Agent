"""MCP Adapter — external tool source for the agent.

Converts MCP Server tools into ToolSpec + ToolCard + contextual runner,
using the same ACI infrastructure as builtin tools.

Design rules:
- MCP sessions/clients never enter LoopState (kept internal to the adapter).
- MCP tools are always category="deferred" (never visible by default).
- MCP tools pass through ToolExecutionService (approval, audit, error handling).
- First version: stdio transport only, flat JSON Schema only.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field, create_model

from rag.agent.tools.card import ToolCard
from rag.agent.tools.spec import (
    ExecutionCategory,
    RiskLevel,
    ToolError,
    ToolPermissions,
    ToolSpec,
)

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
# MCP Tool output wrapper
# ═══════════════════════════════════════════════════════════════════════════════


class MCPToolOutput(BaseModel):
    """Unified output wrapper for all MCP tool calls.

    MCP content blocks (text/image/resource) are serialized into this model.
    The MCPToolFormatter renders them for the LLM.

    ``ok`` follows the RunPythonOutput convention: when False,
    ToolExecutionService converts this to a ToolResult(status="failed").
    """

    text: str = ""
    images: list[str] = Field(default_factory=list)  # base64 strings
    resources: list[str] = Field(default_factory=list)  # resource URIs
    is_error: bool = False
    ok: bool = True
    raw: dict[str, Any] = Field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════════════════════
# Name normalization
# ═══════════════════════════════════════════════════════════════════════════════

_NAME_RE = re.compile(r"[^a-z0-9_]")


def normalize_name(name: str) -> str:
    """Normalize a name for use in canonical MCP tool names.

    Lowercase, replace non-alphanumeric with underscores, collapse repeats.
    """
    normalized = name.lower().replace("-", "_").replace(" ", "_")
    normalized = _NAME_RE.sub("_", normalized)
    return re.sub(r"_+", "_", normalized).strip("_")


def canonical_mcp_name(server_name: str, tool_name: str) -> str:
    """Build canonical name: mcp__{server}__{tool}."""
    return f"mcp__{normalize_name(server_name)}__{normalize_name(tool_name)}"


def server_from_canonical(canonical_name: str) -> str | None:
    """Extract server name from canonical name. Returns None if not MCP."""
    parts = canonical_name.split("__", 2)
    if len(parts) == 3 and parts[0] == "mcp":
        return parts[1]
    return None


def original_tool_name(canonical_name: str) -> str | None:
    """Extract original tool name from canonical name."""
    parts = canonical_name.split("__", 2)
    if len(parts) == 3 and parts[0] == "mcp":
        return parts[2]
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# JSON Schema → Pydantic model (limited scope)
# ═══════════════════════════════════════════════════════════════════════════════


class MCPUnsupportedSchemaError(ValueError):
    """Raised when a MCP tool inputSchema cannot be converted to Pydantic."""


def build_input_model(schema: dict[str, Any], tool_name: str) -> type[BaseModel]:
    """Convert a MCP JSON Schema to a Pydantic BaseModel.

    First version supports only:
      - type=object at top level
      - properties with type: string/number/integer/boolean
      - required (list of required field names)
      - description on fields
      - enum → Literal
      - nested objects up to 1 level deep

    Raises MCPUnsupportedSchemaError for anything beyond this scope.
    """
    if not schema or not isinstance(schema, dict):
        # No schema → empty model
        return type(f"{tool_name}_input", (BaseModel,), {})

    schema_type = schema.get("type", "object")
    if schema_type != "object":
        raise MCPUnsupportedSchemaError(
            f"top-level type must be 'object', got '{schema_type}'"
        )

    properties = schema.get("properties") or {}
    required: set[str] = set(schema.get("required") or [])

    if not properties:
        return type(f"{tool_name}_input", (BaseModel,), {})

    fields: dict[str, Any] = {}
    for prop_name, prop_schema in properties.items():
        if not isinstance(prop_schema, dict):
            continue
        field_name = normalize_name(prop_name)
        try:
            # Pass depth=0; nested objects increment the counter
            fields[field_name] = _map_field(
                field_name, prop_schema, prop_name in required, depth=0,
            )
        except MCPUnsupportedSchemaError:
            raise
        except Exception as e:
            raise MCPUnsupportedSchemaError(
                f"field '{prop_name}' in {tool_name}: {e}"
            ) from e

    model_name = re.sub(r"[^a-zA-Z0-9_]", "_", tool_name)
    return create_model(model_name, **fields)


def _map_field(
    name: str,
    schema: dict[str, Any],
    is_required: bool,
    depth: int = 0,
) -> Any:
    """Map a single JSON Schema property to a Pydantic field annotation."""
    # Reject $ref and $defs
    if "$ref" in schema:
        raise MCPUnsupportedSchemaError(f"$ref not supported (field '{name}')")
    if "$defs" in schema:
        raise MCPUnsupportedSchemaError(f"$defs not supported (field '{name}')")

    # Reject oneOf/anyOf/allOf
    if any(key in schema for key in ("oneOf", "anyOf", "allOf")):
        raise MCPUnsupportedSchemaError(
            f"oneOf/anyOf/allOf not supported (field '{name}')"
        )

    if depth > 1:
        raise MCPUnsupportedSchemaError(f"nested objects max depth 1, field '{name}'")

    field_type = schema.get("type", "string")
    description = schema.get("description")
    py_type: Any  # resolved below

    if field_type == "string":
        py_type = str
        if "enum" in schema:
            from typing import Literal as L

            py_type = L.__getitem__(tuple(schema["enum"]))
    elif field_type in ("number", "integer"):
        py_type = int if field_type == "integer" else float
    elif field_type == "boolean":
        py_type = bool
    elif field_type == "object":
        if depth >= 1:
            raise MCPUnsupportedSchemaError(f"nested object too deep: '{name}'")
        # Check if nested object has properties that would need deeper recursion
        nested_props = schema.get("properties")
        if nested_props and isinstance(nested_props, dict):
            # Has nested properties → build inner model (one level of nesting OK)
            inner_fields = {}
            for inner_name, inner_schema in nested_props.items():
                if not isinstance(inner_schema, dict):
                    continue
                inner_fields[normalize_name(inner_name)] = _map_field(
                    inner_name, inner_schema,
                    inner_name in schema.get("required", []),
                    depth=depth + 1,
                )
            if inner_fields:
                inner_model = create_model(
                    f"{name}_inner", **inner_fields,
                )
                if is_required:
                    default = ...
                else:
                    default = None
                    from typing import Optional
                    inner_model = Optional[inner_model]
                if description:
                    return (inner_model, Field(default, description=description))
                return (inner_model, default)
        # Simple nested object without properties → dict
        py_type = dict
    elif field_type == "array":
        # Array → list (items schema not processed in v1)
        py_type = list
    else:
        raise MCPUnsupportedSchemaError(
            f"unsupported type '{field_type}' for field '{name}'"
        )

    if is_required:
        default = ...
    else:
        default = None
        from typing import Optional

        py_type = Optional[py_type]

    if description:
        return (py_type, Field(default, description=description))
    return (py_type, default)


# ═══════════════════════════════════════════════════════════════════════════════
# MCP annotations → ToolSpec behaviour mapping
# ═══════════════════════════════════════════════════════════════════════════════


def map_mcp_annotations(
    read_only_hint: bool | None = None,
    destructive_hint: bool | None = None,
    idempotent_hint: bool | None = None,
) -> dict[str, Any]:
    """Map MCP ToolAnnotations hints to ToolSpec fields.

    Rules:
    - destructiveHint=True → MUTATE category, HIGH risk, requires confirmation
    - readOnlyHint=True → NETWORK category, LOW risk, idempotent, concurrency_safe
    - Otherwise → NETWORK category, MEDIUM risk
    - execution_category is NEVER READ (would conflict with external_network permission)
    """
    if destructive_hint:
        return dict(
            execution_category=ExecutionCategory.MUTATE,
            risk_level=RiskLevel.HIGH,
            requires_confirmation=True,
            audit_log=True,
            idempotent=False,
            concurrency_safe=False,
            interrupt_behavior="block",
        )

    if read_only_hint:
        # NETWORK has minimum risk MEDIUM (spec.py _minimum_risk_level).
        # readOnly only upgrades idempotent+concurrency_safe — risk stays MEDIUM.
        return dict(
            execution_category=ExecutionCategory.NETWORK,
            risk_level=RiskLevel.MEDIUM,
            idempotent=True,
            concurrency_safe=True,
        )

    # Default
    return dict(
        execution_category=ExecutionCategory.NETWORK,
        risk_level=RiskLevel.MEDIUM,
        idempotent=bool(idempotent_hint),
        concurrency_safe=False,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# MCP Tool → ToolSpec + ToolCard builder
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class MCPToolSpecResult:
    """A fully packaged MCP tool ready for registry registration."""

    spec: ToolSpec
    original_mcp_name: str  # name used in session.call_tool()


def build_mcp_tool_spec(
    mcp_tool: Any,  # mcp.types.Tool
    server_name: str,
) -> MCPToolSpecResult:
    """Convert a MCP Tool to a ToolSpec + ToolCard.

    Returns MCPToolSpecResult with:
    - spec: fully validated ToolSpec ready for ToolRegistry.register()
    - original_mcp_name: the original tool name for session.call_tool()
    """
    original_name = mcp_tool.name
    canonical = canonical_mcp_name(server_name, original_name)
    # getattr to handle both object and dict access patterns
    annotations = getattr(mcp_tool, "annotations", None)
    input_schema = getattr(mcp_tool, "inputSchema", {}) or {}
    description = getattr(mcp_tool, "description", None) or ""

    # Map annotation hints
    read_only = getattr(annotations, "readOnlyHint", None) if annotations else None
    destructive = getattr(annotations, "destructiveHint", None) if annotations else None
    idempotent = getattr(annotations, "idempotentHint", None) if annotations else None
    behavior = map_mcp_annotations(
        read_only_hint=read_only,
        destructive_hint=destructive,
        idempotent_hint=idempotent,
    )

    # Build input model
    try:
        input_model = build_input_model(input_schema, canonical)
    except MCPUnsupportedSchemaError:
        # Fallback: explicit dict-wrapper so arguments are preserved.
        # Without this, an empty BaseModel would silently drop all kwargs
        # because Pydantic ignores extra fields by default.
        input_model = create_model(
            f"{canonical}_input",
            arguments=(dict, Field(default_factory=dict, description="Raw tool arguments")),
        )

    spec = ToolSpec(
        name=canonical,
        description=description or f"MCP tool: {original_name}",
        input_model=input_model,
        output_model=MCPToolOutput,
        error_model=ToolError,
        permissions=ToolPermissions(external_network=True),
        timeout_seconds=30.0,
        max_retries=0,
        aci=ToolCard(
            when_to_use=description or f"External tool '{original_name}' from MCP server '{server_name}'",
            when_not_to_use="Verify server is connected before use",
            activation_group="mcp",
            selection_tags=("mcp", server_name),
            domains=("external", server_name),
        ),
        **behavior,
    )
    return MCPToolSpecResult(spec=spec, original_mcp_name=original_name)


# ═══════════════════════════════════════════════════════════════════════════════
# MCP Tool Config
# ═══════════════════════════════════════════════════════════════════════════════


class MCPToolConfig(BaseModel):
    """Configuration for a single MCP Server."""

    name: str
    transport: str = "stdio"
    command: str = ""
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
    tools_allowlist: list[str] = Field(default_factory=list)
    enabled: bool = False

    def resolve_env(self) -> dict[str, str]:
        """Resolve environment variables with ${VAR} syntax."""
        resolved: dict[str, str] = {}
        for key, value in self.env.items():
            if value.startswith("${") and value.endswith("}"):
                var_name = value[2:-1]
                resolved[key] = os.environ.get(var_name, "")
            else:
                resolved[key] = value
        return resolved


# ═══════════════════════════════════════════════════════════════════════════════
# MCP Tool Adapter (per-server session + tool management)
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class MCPToolAdapter:
    """Manages a single MCP Server connection and its tools.

    Holds:
    - config: server configuration
    - session: MCP ClientSession (None until connected)
    - tools: dict of canonical_name → MCPToolSpecResult
    - _original_names: dict of canonical_name → original MCP tool name
    """

    config: MCPToolConfig
    session: Any = None  # mcp.client.session.ClientSession
    tools: dict[str, MCPToolSpecResult] = field(default_factory=dict)
    _original_names: dict[str, str] = field(default_factory=dict)
    _connected: bool = False

    @property
    def is_connected(self) -> bool:
        return self._connected

    async def connect(self) -> None:
        """Establish MCP stdio connection and initialize session."""
        if self._connected:
            return
        from mcp.client.stdio import StdioServerParameters, stdio_client
        from mcp.client.session import ClientSession

        env = self.config.resolve_env()
        server_params = StdioServerParameters(
            command=self.config.command,
            args=self.config.args,
            env=env if env else None,
        )
        # stdio_client returns a context manager; enter it
        self._stdio_ctx = stdio_client(server_params)
        read_stream, write_stream = await self._stdio_ctx.__aenter__()
        self._session_ctx = ClientSession(read_stream, write_stream)
        self.session = await self._session_ctx.__aenter__()
        await self.session.initialize()
        self._connected = True
        logger.info(f"MCP server '{self.config.name}' connected")

    async def disconnect(self) -> None:
        """Close MCP session and transport."""
        if not self._connected:
            return
        try:
            if hasattr(self, "_session_ctx"):
                await self._session_ctx.__aexit__(None, None, None)
        except Exception:
            logger.warning(f"Error closing MCP session for '{self.config.name}'", exc_info=True)
        try:
            if hasattr(self, "_stdio_ctx"):
                await self._stdio_ctx.__aexit__(None, None, None)
        except Exception:
            logger.warning(f"Error closing stdio for '{self.config.name}'", exc_info=True)
        self._connected = False
        self.session = None
        logger.info(f"MCP server '{self.config.name}' disconnected")

    async def list_tools(self) -> list[MCPToolSpecResult]:
        """Discover tools from the MCP server and build ToolSpecs.

        Only tools in config.tools_allowlist are included.
        """
        if not self._connected:
            raise RuntimeError(f"MCP server '{self.config.name}' is not connected")

        result = await self.session.list_tools()
        mcp_tools = getattr(result, "tools", []) or []

        allowlist = set(self.config.tools_allowlist) if self.config.tools_allowlist else set()
        self.tools.clear()
        self._original_names.clear()

        for mcp_tool in mcp_tools:
            original_name = getattr(mcp_tool, "name", "")
            if allowlist and original_name not in allowlist:
                continue
            try:
                built = build_mcp_tool_spec(mcp_tool, self.config.name)
            except Exception as e:
                logger.warning(
                    f"Skipping MCP tool '{original_name}' from '{self.config.name}': {e}"
                )
                continue
            self.tools[built.spec.name] = built
            self._original_names[built.spec.name] = original_name

        return list(self.tools.values())

    def get_runner(self, canonical_name: str) -> Any:
        """Return a contextual runner for the given MCP tool.

        The runner calls session.call_tool() with the original tool name.
        It returns MCPToolOutput (BaseModel), not ToolResult.
        ToolExecutionService wraps it into ToolResult.
        """
        adapter = self
        original_name = self._original_names.get(canonical_name, "")

        async def _run(input_payload: BaseModel, context: Any) -> MCPToolOutput:
            if not adapter._connected:
                return MCPToolOutput(
                    ok=False,
                    is_error=True,
                    raw={"error": f"MCP server '{adapter.config.name}' disconnected"},
                )
            # Handle dict-wrapper fallback for complex schemas
            raw_args = input_payload.model_dump(exclude_none=True)
            if "arguments" in raw_args and isinstance(raw_args["arguments"], dict):
                args = raw_args["arguments"]
            else:
                args = raw_args
            try:
                result = await adapter.session.call_tool(original_name, arguments=args)
            except Exception as e:
                return MCPToolOutput(
                    ok=False,
                    is_error=True,
                    raw={"error": str(e)},
                )
            # Extract content blocks
            content_blocks = getattr(result, "content", []) or []
            text_parts: list[str] = []
            images: list[str] = []
            resources: list[str] = []
            is_error = getattr(result, "isError", False)

            for block in content_blocks:
                block_type = getattr(block, "type", "")
                if block_type == "text":
                    text_parts.append(getattr(block, "text", ""))
                elif block_type == "image":
                    data = getattr(block, "data", "")
                    if data:
                        images.append(data)
                elif block_type == "resource":
                    uri = getattr(block, "resource", {})
                    if isinstance(uri, dict):
                        resources.append(uri.get("uri", str(uri)))
                    else:
                        resources.append(str(uri))

            return MCPToolOutput(
                ok=not is_error,  # propagate MCP-level errors to ToolExecutionService
                text="\n".join(text_parts) if text_parts else json.dumps(args),
                images=images,
                resources=resources,
                is_error=is_error,
                raw={"mcp_tool": original_name, "server": adapter.config.name},
            )

        return _run


# ═══════════════════════════════════════════════════════════════════════════════
# MCP Tool Registry (multi-server manager)
# ═══════════════════════════════════════════════════════════════════════════════

import json  # noqa: E402


@dataclass
class MCPToolRegistry:
    """Manages all MCP Server adapters and their tools.

    Lifecycle:
    1. load_configs(configs) → creates adapters (doesn't connect yet)
    2. connect_all() → connects enabled servers and discovers tools
    3. list_all_tools() → returns ToolSpecs for all discovered tools
    4. register_to(registry, agent_allowed_tools) → registers specs + runners
    5. disconnect_all() → cleanup
    """

    adapters: dict[str, MCPToolAdapter] = field(default_factory=dict)
    _all_tools: list[ToolSpec] = field(default_factory=list)
    _adapter_by_tool: dict[str, MCPToolAdapter] = field(default_factory=dict)

    def load_configs(self, configs: list[MCPToolConfig]) -> None:
        """Load MCP server configs; create adapters for enabled ones."""
        for cfg in configs:
            if not cfg.enabled:
                continue
            if not cfg.tools_allowlist:
                logger.warning(
                    f"MCP server '{cfg.name}' is enabled but has no tools_allowlist — skipping"
                )
                continue
            if cfg.name in self.adapters:
                logger.warning(f"MCP server '{cfg.name}' already registered — skipping")
                continue
            self.adapters[cfg.name] = MCPToolAdapter(config=cfg)

    async def connect_all(self) -> list[ToolSpec]:
        """Connect all enabled adapters and discover their tools."""
        self._all_tools.clear()
        self._adapter_by_tool.clear()

        for name, adapter in self.adapters.items():
            try:
                await adapter.connect()
            except Exception as e:
                logger.error(f"Failed to connect MCP server '{name}': {e}")
                continue

            try:
                tool_results = await adapter.list_tools()
            except Exception as e:
                logger.error(f"Failed to list tools from '{name}': {e}")
                await adapter.disconnect()
                continue

            for tr in tool_results:
                self._all_tools.append(tr.spec)
                self._adapter_by_tool[tr.spec.name] = adapter

        return self._all_tools

    def list_all_tools(self) -> list[ToolSpec]:
        """Return ToolSpecs for all discovered MCP tools."""
        return list(self._all_tools)

    def get_adapter(self, canonical_name: str) -> MCPToolAdapter | None:
        """Get the adapter responsible for a given canonical tool name."""
        return self._adapter_by_tool.get(canonical_name)

    def get_runner(self, canonical_name: str) -> Any:
        """Get a contextual runner for a specific MCP tool."""
        adapter = self._adapter_by_tool.get(canonical_name)
        if adapter is None:
            raise KeyError(f"No MCP adapter for '{canonical_name}'")
        return adapter.get_runner(canonical_name)

    async def disconnect_all(self) -> None:
        """Disconnect all adapters."""
        for name, adapter in self.adapters.items():
            try:
                await adapter.disconnect()
            except Exception:
                logger.warning(f"Error disconnecting '{name}'", exc_info=True)
        self._all_tools.clear()
        self._adapter_by_tool.clear()

    @property
    def server_names(self) -> list[str]:
        """List of connected server names."""
        return [name for name, a in self.adapters.items() if a.is_connected]

"""B2a: MCP Adapter unit tests — naming, permissions, schema, spec building, runner type."""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from rag.agent.tools.mcp_adapter import (
    MCPToolConfig,
    MCPToolOutput,
    MCPToolRegistry,
    MCPUnsupportedSchemaError,
    build_input_model,
    canonical_mcp_name,
    map_mcp_annotations,
    normalize_name,
    server_from_canonical,
)
from rag.agent.tools.spec import ExecutionCategory, RiskLevel


class TestMCPNaming:
    def test_canonical_name_format(self) -> None:
        assert canonical_mcp_name("github", "search_repos") == "mcp__github__search_repos"

    def test_canonical_name_normalizes_case(self) -> None:
        assert canonical_mcp_name("GitHub", "Search_Repos") == "mcp__github__search_repos"

    def test_canonical_name_strips_special_chars(self) -> None:
        name = canonical_mcp_name("my-server", "get file!")
        assert "__" in name
        assert "!" not in name

    def test_normalize_name_handles_empty(self) -> None:
        assert normalize_name("") == ""

    def test_server_from_canonical(self) -> None:
        assert server_from_canonical("mcp__github__search") == "github"
        assert server_from_canonical("builtin_tool") is None
        assert server_from_canonical("mcp__") is None


class TestMCPAnnotations:
    def test_readonly_hint_maps_to_network_with_idempotent(self) -> None:
        """readOnlyHint → NETWORK (not READ!). risk=MEDIUM (NETWORK floor), idempotent, concurrency_safe."""
        behavior = map_mcp_annotations(read_only_hint=True)
        assert behavior["execution_category"] == ExecutionCategory.NETWORK
        # NETWORK has minimum risk MEDIUM (spec.py _minimum_risk_level).
        # readOnly upgrades idempotent+concurrency_safe, not risk level.
        assert behavior["risk_level"] == RiskLevel.MEDIUM
        assert behavior["idempotent"] is True
        assert behavior["concurrency_safe"] is True

    def test_destructive_hint_maps_to_mutate(self) -> None:
        """destructiveHint → MUTATE, HIGH risk, requires confirmation."""
        behavior = map_mcp_annotations(destructive_hint=True)
        assert behavior["execution_category"] == ExecutionCategory.MUTATE
        assert behavior["risk_level"] == RiskLevel.HIGH
        assert behavior["requires_confirmation"] is True
        assert behavior["audit_log"] is True

    def test_default_maps_to_network_medium(self) -> None:
        """No hints → NETWORK, MEDIUM risk."""
        behavior = map_mcp_annotations()
        assert behavior["execution_category"] == ExecutionCategory.NETWORK
        assert behavior["risk_level"] == RiskLevel.MEDIUM

    def test_destructive_overrides_readonly(self) -> None:
        """destructiveHint=True takes priority over readOnlyHint=True."""
        behavior = map_mcp_annotations(read_only_hint=True, destructive_hint=True)
        assert behavior["execution_category"] == ExecutionCategory.MUTATE


class TestJSONSchemaToPydantic:
    def test_flat_schema(self) -> None:
        model = build_input_model(
            {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "search term"},
                    "limit": {"type": "integer", "description": "max results"},
                },
                "required": ["query"],
            },
            "test_tool",
        )
        instance = model(query="hello")
        assert instance.query == "hello"
        assert instance.limit is None  # not required
        assert "limit" in model.model_fields

    def test_empty_schema(self) -> None:
        model = build_input_model({}, "no_args")
        assert issubclass(model, BaseModel)

    def test_no_properties(self) -> None:
        model = build_input_model({"type": "object"}, "no_props")
        assert issubclass(model, BaseModel)

    def test_enum_field(self) -> None:
        model = build_input_model(
            {
                "type": "object",
                "properties": {
                    "status": {"type": "string", "enum": ["open", "closed"]},
                },
            },
            "enum_test",
        )
        instance = model(status="open")
        assert instance.status == "open"

    def test_complex_schema_raises(self) -> None:
        """Schema with $ref raises MCPUnsupportedSchemaError."""
        with pytest.raises(MCPUnsupportedSchemaError):
            build_input_model(
                {
                    "type": "object",
                    "properties": {
                        "item": {"$ref": "#/definitions/Item"},
                    },
                },
                "ref_test",
            )

    def test_deep_nesting_raises(self) -> None:
        """Nested objects > 1 level deep raise MCPUnsupportedSchemaError."""
        with pytest.raises(MCPUnsupportedSchemaError):
            build_input_model(
                {
                    "type": "object",
                    "properties": {
                        "outer": {
                            "type": "object",
                            "properties": {
                                "inner": {
                                    "type": "object",
                                    "properties": {"deep": {"type": "string"}},
                                },
                            },
                        },
                    },
                },
                "deep_test",
            )


class TestMCPToolConfig:
    def test_minimal_config(self) -> None:
        cfg = MCPToolConfig(
            name="test",
            command="echo",
            tools_allowlist=["hello"],
        )
        assert cfg.enabled is False
        assert cfg.transport == "stdio"

    def test_resolve_env(self) -> None:
        import os

        os.environ["_TEST_MCP_TOKEN"] = "secret123"
        cfg = MCPToolConfig(
            name="test",
            command="npx",
            env={"API_KEY": "${_TEST_MCP_TOKEN}"},
            tools_allowlist=["x"],
        )
        resolved = cfg.resolve_env()
        assert resolved["API_KEY"] == "secret123"
        del os.environ["_TEST_MCP_TOKEN"]


class TestMCPToolOutput:
    def test_ok_output(self) -> None:
        out = MCPToolOutput(text="result text")
        assert out.text == "result text"
        assert out.is_error is False

    def test_error_output(self) -> None:
        out = MCPToolOutput(is_error=True, raw={"error": "timeout"})
        assert out.is_error is True

    def test_images_counted(self) -> None:
        out = MCPToolOutput(images=["base64data"])
        assert len(out.images) == 1


class TestMCPToolRegistry:
    def test_load_configs_only_enabled(self) -> None:
        registry = MCPToolRegistry()
        registry.load_configs([
            MCPToolConfig(name="a", enabled=False, tools_allowlist=["x"]),
            MCPToolConfig(name="b", enabled=True, tools_allowlist=["y"]),
            MCPToolConfig(name="c", enabled=True, tools_allowlist=[]),  # no allowlist → skipped
        ])
        assert "a" not in registry.adapters
        assert "b" in registry.adapters
        assert "c" not in registry.adapters  # skipped due to empty allowlist


class TestMCPErrorPropagation:
    """P2-4: MCP errors produce ok=False in MCPToolOutput."""

    def test_error_output_has_ok_false(self) -> None:
        out = MCPToolOutput(ok=False, is_error=True, raw={"error": "timeout"})
        assert out.ok is False
        assert out.is_error is True

    def test_ok_output_has_ok_true(self) -> None:
        out = MCPToolOutput(text="success")
        assert out.ok is True


class TestMCPFallbackInputModel:
    """P1-3: Fallback input model preserves arguments."""

    def test_fallback_model_preserves_dict_args(self) -> None:
        """Fallback model has explicit 'arguments' field, not empty."""
        from rag.agent.tools.mcp_adapter import build_mcp_tool_spec

        # Simulate a MCP tool with a complex schema
        class MockMCPTool:
            name = "complex_tool"
            description = "A tool with complex schema"
            annotations = None
            inputSchema = {"type": "object", "properties": {"item": {"$ref": "#/defs/X"}}}

        result = build_mcp_tool_spec(MockMCPTool(), "test_server")
        spec = result.spec
        # Should have an 'arguments' field, not be empty
        assert spec.input_model is not None
        # Fallback model should accept arbitrary dict
        instance = spec.input_model(arguments={"x": 1, "y": "hello"})
        assert instance.arguments == {"x": 1, "y": "hello"}

    def test_server_names_empty_by_default(self) -> None:
        registry = MCPToolRegistry()
        assert registry.server_names == []

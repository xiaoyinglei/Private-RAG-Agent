from __future__ import annotations

import pytest
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from pydantic import BaseModel

from rag.agent.tools.spec import ToolError, ToolPermissions, ToolResult, ToolSpec


class SearchInput(BaseModel):
    query: str
    limit: int = 10


class SearchOutput(BaseModel):
    items: list[str]


class TestToolPermissions:
    def test_default_all_false(self) -> None:
        p = ToolPermissions()
        assert p.read_db is False
        assert p.write_db is False
        assert p.kg_mutation is False
        assert p.external_network is False

    def test_kg_mutation_flags_write(self) -> None:
        p = ToolPermissions(write_db=True, kg_mutation=True)
        assert p.kg_mutation is True
        assert p.write_db is True


class TestToolSpec:
    def test_minimal_spec(self) -> None:
        spec = ToolSpec(
            name="test_search",
            description="Search for documents",
            input_model=SearchInput,
            output_model=SearchOutput,
            error_model=ToolError,
            permissions=ToolPermissions(read_db=True),
            timeout_seconds=5.0,
        )
        assert spec.name == "test_search"
        assert spec.timeout_seconds == 5.0
        assert spec.max_retries == 0
        assert spec.idempotent is False
        assert spec.requires_confirmation is False
        assert spec.audit_log is False

    def test_kg_tool_spec_enforces_confirmation(self) -> None:
        spec = ToolSpec(
            name="kg_write",
            description="Write to knowledge graph",
            input_model=SearchInput,
            output_model=SearchOutput,
            error_model=ToolError,
            permissions=ToolPermissions(kg_mutation=True, write_db=True),
            timeout_seconds=10.0,
            requires_confirmation=True,
            audit_log=True,
            idempotent=True,
            max_retries=2,
        )
        assert spec.requires_confirmation is True
        assert spec.audit_log is True
        assert spec.idempotent is True
        assert spec.permissions.kg_mutation is True


class TestToolResult:
    def test_ok_result(self) -> None:
        result = ToolResult(
            tool_call_id="tc_001",
            tool_name="search",
            status="ok",
            output=SearchOutput(items=["a", "b"]),
            latency_ms=100.0,
        )
        assert result.status == "ok"
        assert result.output is not None

    def test_error_result(self) -> None:
        result = ToolResult(
            tool_call_id="tc_002",
            tool_name="search",
            status="error",
            error=ToolError(code="timeout", message="timed out after 5s", retryable=True),
            latency_ms=5000.0,
        )
        assert result.status == "error"
        assert result.error is not None

    def test_ok_rejects_missing_output(self) -> None:
        with pytest.raises(ValueError, match="output is required"):
            ToolResult(tool_call_id="tc_003", tool_name="x", status="ok", output=None, latency_ms=0)

    def test_ok_rejects_error_present(self) -> None:
        with pytest.raises(ValueError, match="error must be None"):
            ToolResult(
                tool_call_id="tc_004",
                tool_name="x",
                status="ok",
                output=SearchOutput(items=[]),
                error=ToolError(code="internal", message="x", retryable=True),
                latency_ms=0,
            )

    def test_error_rejects_missing_error(self) -> None:
        with pytest.raises(ValueError, match="error is required"):
            ToolResult(tool_call_id="tc_005", tool_name="x", status="error", error=None, latency_ms=0)

    def test_error_rejects_output_present(self) -> None:
        with pytest.raises(ValueError, match="output must be None"):
            ToolResult(
                tool_call_id="tc_006",
                tool_name="x",
                status="error",
                output=SearchOutput(items=[]),
                error=ToolError(code="internal", message="x", retryable=True),
                latency_ms=0,
            )

    def test_msgpack_round_trip_preserves_output_model_type(self) -> None:
        result = ToolResult(
            tool_call_id="tc_007",
            tool_name="search",
            status="ok",
            output=SearchOutput(items=["a", "b"]),
            latency_ms=100.0,
        )
        serde = JsonPlusSerializer(allowed_msgpack_modules=[ToolResult])

        restored = serde.loads_typed(serde.dumps_typed(result))

        assert isinstance(restored, ToolResult)
        assert restored.output == SearchOutput(items=["a", "b"])


class TestToolError:
    def test_timeout_error_is_retryable(self) -> None:
        e = ToolError(code="timeout", message="timed out", retryable=True)
        assert e.retryable is True
        assert e.code == "timeout"

    def test_tool_denied_is_not_retryable(self) -> None:
        e = ToolError(code="tool_denied", message="not allowed", retryable=False)
        assert e.retryable is False
        assert e.code == "tool_denied"

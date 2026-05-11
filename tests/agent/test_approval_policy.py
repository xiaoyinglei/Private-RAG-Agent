from __future__ import annotations

from rag.agent.core.approval_policy import (
    ApprovalAction,
    ApprovalDecision,
    ApprovalPolicy,
    merge_approval_requests,
)
from rag.agent.tools.spec import ToolError, ToolPermissions, ToolSpec
from pydantic import BaseModel


class _DummyInput(BaseModel):
    query: str
    top_k: int = 8


class _DummyOutput(BaseModel):
    items: list[str]


def _make_spec(
    name: str = "test_tool",
    **permissions: bool,
) -> ToolSpec:
    return ToolSpec(
        name=name,
        description="test",
        input_model=_DummyInput,
        output_model=_DummyOutput,
        error_model=ToolError,
        permissions=ToolPermissions(**permissions),
        timeout_seconds=5.0,
    )


class TestApprovalPolicy:
    def test_unregistered_tool_denied(self) -> None:
        policy = ApprovalPolicy()
        result = policy.decide(
            tool_name="unknown_tool",
            arguments={},
            spec=None,
        )
        assert result.action == ApprovalAction.DENY
        assert result.risk_level == "high"

    def test_read_only_tool_allowed(self) -> None:
        policy = ApprovalPolicy()
        spec = _make_spec(name="vector_search", read_db=True, embed=True)
        result = policy.decide(
            tool_name="vector_search",
            arguments={"query": "test", "top_k": 8},
            spec=spec,
        )
        assert result.action == ApprovalAction.ALLOW
        assert result.risk_level == "low"

    def test_write_tool_asks_for_approval(self) -> None:
        policy = ApprovalPolicy()
        spec = _make_spec(name="kg_upsert", write_db=True, kg_mutation=True)
        result = policy.decide(
            tool_name="kg_upsert",
            arguments={"node": "test", "data": "x"},
            spec=spec,
        )
        assert result.action == ApprovalAction.ASK
        assert result.risk_level == "medium"
        assert result.request is not None
        assert result.request.kind == "tool_approval"
        assert len(result.request.tool_calls) == 1

    def test_external_network_tool_asks(self) -> None:
        policy = ApprovalPolicy()
        spec = _make_spec(name="web_search", external_network=True)
        result = policy.decide(
            tool_name="web_search",
            arguments={"query": "test"},
            spec=spec,
        )
        assert result.action == ApprovalAction.ASK

    def test_delete_file_tool_denied(self) -> None:
        policy = ApprovalPolicy()
        spec = _make_spec(name="delete_file", write_db=True)
        result = policy.decide(
            tool_name="delete_file",
            arguments={"path": "/tmp/test"},
            spec=spec,
        )
        assert result.action == ApprovalAction.DENY

    def test_user_data_tool_asks(self) -> None:
        policy = ApprovalPolicy()
        spec = _make_spec(name="export_report", user_data=True)
        result = policy.decide(
            tool_name="export_report",
            arguments={"format": "pdf"},
            spec=spec,
        )
        assert result.action == ApprovalAction.ASK


class TestMergeApprovalRequests:
    def test_merges_multiple_ask_decisions(self) -> None:
        policy = ApprovalPolicy()
        spec1 = _make_spec(name="kg_upsert", write_db=True, kg_mutation=True)
        spec2 = _make_spec(name="export_report", user_data=True)

        d1 = policy.decide(tool_name="kg_upsert", arguments={"node": "x"}, spec=spec1)
        d2 = policy.decide(tool_name="export_report", arguments={"format": "pdf"}, spec=spec2)

        merged = merge_approval_requests([d1, d2])
        assert merged.kind == "tool_approval"
        assert len(merged.tool_calls) == 2
        assert "kg_upsert" in merged.question
        assert "export_report" in merged.question

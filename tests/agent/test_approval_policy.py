from __future__ import annotations

from pydantic import BaseModel

from rag.agent.core.approval_policy import (
    ApprovalAction,
    ApprovalPolicy,
    merge_approval_requests,
)
from rag.agent.tools.spec import ExecutionCategory, RiskLevel, ToolError, ToolPermissions, ToolSpec


class _DummyInput(BaseModel):
    query: str
    top_k: int = 8


class _DummyOutput(BaseModel):
    items: list[str]


def _make_spec(
    name: str = "test_tool",
    *,
    requires_confirmation: bool = False,
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
        requires_confirmation=requires_confirmation,
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
        assert result.risk_level == "high"  # kg_mutation → MUTATE → high
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

    def test_tool_contract_can_require_approval(self) -> None:
        policy = ApprovalPolicy()
        spec = _make_spec(
            name="sensitive_read",
            read_db=True,
            requires_confirmation=True,
        )
        result = policy.decide(
            tool_name="sensitive_read",
            arguments={"query": "test"},
            spec=spec,
        )
        assert result.action == ApprovalAction.ASK
        assert result.request is not None
        assert result.request.tool_calls[0].reason == "工具契约要求执行前确认"

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


def test_write_fs_requires_ask() -> None:
    policy = ApprovalPolicy()
    spec = _make_spec(name="write_file", write_fs=True)
    result = policy.decide(
        tool_name="write_file",
        arguments={"path": "reports/a.md"},
        spec=spec,
    )
    assert result.action == ApprovalAction.ASK


def test_execute_code_requires_ask() -> None:
    policy = ApprovalPolicy()
    spec = _make_spec(name="run_python", execute_code=True)
    result = policy.decide(
        tool_name="run_python",
        arguments={"script_path": "scratch/main.py"},
        spec=spec,
    )
    assert result.action == ApprovalAction.ASK


def test_auto_approve_sandboxed_does_not_bypass_network_permission() -> None:
    policy = ApprovalPolicy()
    spec = _make_spec(name="network_python", execute_code=True, external_network=True)
    result = policy.decide(
        tool_name="network_python",
        arguments={"script_path": "scratch/fetch.py"},
        spec=spec,
        auto_approve_sandboxed=True,
    )
    assert result.action == ApprovalAction.ASK


def test_read_fs_allows() -> None:
    policy = ApprovalPolicy()
    spec = _make_spec(name="list_files", read_fs=True)
    result = policy.decide(
        tool_name="list_files",
        arguments={},
        spec=spec,
    )
    assert result.action == ApprovalAction.ALLOW


def test_permission_backstop_prevents_runtime_category_downgrade() -> None:
    policy = ApprovalPolicy()
    spec = _make_spec(name="write_file", write_fs=True)
    object.__setattr__(spec, "execution_category", ExecutionCategory.READ)

    result = policy.decide(
        tool_name="write_file",
        arguments={"path": "reports/a.md"},
        spec=spec,
    )

    assert result.action == ApprovalAction.ASK
    assert result.risk_level == "medium"


def test_spec_risk_level_controls_approval_summary() -> None:
    policy = ApprovalPolicy()
    spec = _make_spec(name="write_file", write_fs=True)
    object.__setattr__(spec, "risk_level", RiskLevel.HIGH)

    result = policy.decide(
        tool_name="write_file",
        arguments={"path": "reports/a.md"},
        spec=spec,
    )

    assert result.action == ApprovalAction.ASK
    assert result.risk_level == "high"
    assert result.request is not None
    assert result.request.tool_calls[0].risk_level == "high"


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

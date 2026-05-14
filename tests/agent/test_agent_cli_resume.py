from __future__ import annotations

import pytest

from rag.agent.cli import _build_resume_response
from rag.agent.core.human_input import HumanInputRequest, ToolCallSummary


def test_build_resume_response_approves_pending_tool_ids() -> None:
    request = HumanInputRequest(
        request_id="hir_test",
        kind="tool_approval",
        question="approve?",
        tool_calls=[
            ToolCallSummary(
                tool_call_id="tc_one",
                tool_name="write_tool",
                args_preview="data='one'",
            ),
            ToolCallSummary(
                tool_call_id="tc_two",
                tool_name="write_tool",
                args_preview="data='two'",
            ),
        ],
    )

    response = _build_resume_response(request, "allow_once")

    assert response.request_id == "hir_test"
    assert response.decision == "allow_once"
    assert response.approved_tool_call_ids == ["tc_one", "tc_two"]
    assert response.denied_tool_call_ids == []


def test_build_resume_response_denies_pending_tool_ids() -> None:
    request = HumanInputRequest(
        request_id="hir_test",
        kind="tool_approval",
        question="approve?",
        tool_calls=[
            ToolCallSummary(
                tool_call_id="tc_one",
                tool_name="write_tool",
                args_preview="data='one'",
            )
        ],
    )

    response = _build_resume_response(request, "deny")

    assert response.approved_tool_call_ids == []
    assert response.denied_tool_call_ids == ["tc_one"]


def test_build_resume_response_rejects_unknown_decision() -> None:
    request = HumanInputRequest(
        request_id="hir_test",
        kind="tool_approval",
        question="approve?",
    )

    with pytest.raises(ValueError):
        _build_resume_response(request, "unknown")

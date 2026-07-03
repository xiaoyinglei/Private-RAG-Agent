from __future__ import annotations

import pytest

from rag.agent import cli
from rag.agent.cli import _build_resume_response
from rag.agent.core.human_input import HumanInputRequest, ToolCallSummary
from rag.agent.service import AgentRunResult


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


def test_agent_resume_closes_service_after_result(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    closed: list[bool] = []

    class _Service:
        async def apending_human_input_request(self, *, run_id: str) -> HumanInputRequest:
            return HumanInputRequest(
                request_id=f"hir_{run_id}",
                kind="tool_approval",
                question="approve?",
                tool_calls=[
                    ToolCallSummary(
                        tool_call_id="tc_one",
                        tool_name="write_file",
                        args_preview="path='scratch/out.txt'",
                    )
                ],
            )

        async def resume(
            self,
            *,
            run_id: str,
            response: object,
            workspace_path: str | None,
        ) -> AgentRunResult:
            del response, workspace_path
            return AgentRunResult(
                run_id=run_id,
                thread_id=run_id,
                status="done",
                final_answer="resumed",
            )

        async def aclose(self) -> None:
            closed.append(True)

    monkeypatch.setattr(cli, "_build_optional_rag_runtime", lambda **_kwargs: (None, ()))
    monkeypatch.setattr(cli, "_build_model_control_plane", lambda **_kwargs: None)
    monkeypatch.setattr(cli, "_build_agent_service", lambda *_args, **_kwargs: _Service())

    cli.agent_resume(
        run_id="resume-close",
        checkpoint_db=tmp_path / "agent.sqlite",
    )

    assert closed == [True]

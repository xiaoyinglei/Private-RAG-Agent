from __future__ import annotations

from contextlib import asynccontextmanager

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
    facade_options: list[dict[str, object]] = []

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

    class _Facade:
        workspace_path = tmp_path

        @asynccontextmanager
        async def _open_product_runtime(self, **_kwargs: object):
            service = _Service()
            try:
                yield service
            finally:
                await service.aclose()

    def create_facade(**kwargs: object) -> _Facade:
        facade_options.append(kwargs)
        return _Facade()

    monkeypatch.setattr(cli, "_create_agent_facade", create_facade)

    cli.agent_resume(
        run_id="resume-close",
        checkpoint_db=tmp_path / "agent.sqlite",
        knowledge=["company-docs"],
    )

    assert closed == [True]
    assert facade_options[0]["knowledge"] == ("company-docs",)


def test_interactive_terminal_fails_closed_for_ci_or_non_tty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Stream:
        def __init__(self, value: bool) -> None:
            self.value = value

        def isatty(self) -> bool:
            return self.value

    monkeypatch.delenv("CI", raising=False)
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    monkeypatch.setattr(cli.sys, "stdin", _Stream(True))
    monkeypatch.setattr(cli.sys, "stdout", _Stream(True))
    assert cli._is_interactive_terminal() is True

    monkeypatch.setattr(cli.sys, "stdin", _Stream(False))
    assert cli._is_interactive_terminal() is False

    monkeypatch.setattr(cli.sys, "stdin", _Stream(True))
    monkeypatch.setenv("CI", "true")
    assert cli._is_interactive_terminal() is False


@pytest.mark.anyio
async def test_inline_approval_resumes_on_the_same_runtime(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = HumanInputRequest(
        request_id="hir_inline",
        kind="tool_approval",
        question="approve?",
        tool_calls=[
            ToolCallSummary(
                tool_call_id="tc_inline",
                tool_name="run_command",
                args_preview="command='pytest'",
            )
        ],
    )
    calls: list[str] = []

    class _Service:
        async def run(self, run_request: object) -> AgentRunResult:
            calls.append("run")
            return AgentRunResult(
                run_id="inline-run",
                thread_id="inline-run",
                status="paused",
                human_input_request=request,
                needs_user_input="approve?",
            )

        async def resume(self, *, run_id: str, response: object) -> AgentRunResult:
            del run_id, response
            calls.append("resume")
            return AgentRunResult(
                run_id="inline-run",
                thread_id="inline-run",
                status="done",
                final_answer="continued",
            )

    class _Facade:
        workspace_path = tmp_path

        @asynccontextmanager
        async def _open_product_runtime(self, **_kwargs: object):
            calls.append("open")
            try:
                yield _Service()
            finally:
                calls.append("close")

    monkeypatch.setattr(
        cli,
        "_handle_pause",
        lambda _result, _run_id: _build_resume_response(
            request,
            "allow_once",
        ),
    )

    result = await cli._run_facade_command(
        _Facade(),
        task="run tests",
        files=(),
        run_id="inline-run",
        max_tokens_total=None,
        interactive_approval=True,
    )

    assert result.answer == "continued"
    assert calls == ["open", "run", "resume", "close"]

from __future__ import annotations

from contextlib import asynccontextmanager
from uuid import uuid4

import pytest

from agent_runtime import AgentResult
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

    class _Facade:
        async def aresume(
            self,
            turn_id: str,
            action: str,
            *,
            user_input: str | None,
        ) -> AgentResult:
            assert action == "allow_once"
            assert user_input is None
            closed.append(True)
            return AgentResult.from_internal(
                AgentRunResult(
                    run_id=turn_id,
                    thread_id=turn_id,
                    session_id=str(uuid4()),
                    status="done",
                    final_answer="resumed",
                )
            )

    def create_facade(**kwargs: object) -> _Facade:
        facade_options.append(kwargs)
        return _Facade()

    monkeypatch.setattr(cli, "_create_agent_facade", create_facade)

    turn_id = str(uuid4())
    cli.agent_resume(
        turn_id=turn_id,
        checkpoint_db=tmp_path / "agent.sqlite",
        action="allow_once",
    )

    assert closed == [True]
    assert facade_options == [
        {
            "checkpoint_db": tmp_path / "agent.sqlite",
            "vector_dsn": None,
        }
    ]


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
        async def chat(self, run_request: object) -> AgentRunResult:
            calls.append("chat")
            return AgentRunResult(
                run_id=run_request.run_id,
                thread_id=run_request.run_id,
                session_id=str(uuid4()),
                status="paused",
                human_input_request=request,
                needs_user_input="approve?",
            )

        async def resume_turn(
            self,
            *,
            turn_id: str,
            action: str,
            user_input: str | None,
        ) -> AgentRunResult:
            del action, user_input
            calls.append("resume")
            return AgentRunResult(
                run_id=turn_id,
                thread_id=turn_id,
                session_id=str(uuid4()),
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
        turn_id=str(uuid4()),
        max_tokens_total=None,
        interactive_approval=True,
    )

    assert result.answer == "continued"
    assert calls == ["open", "chat", "resume", "close"]

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from uuid import uuid4

import pytest
from pytest import MonkeyPatch

from rag.agent import cli
from rag.agent.service import AgentRunRequest, AgentRunResult
from rag.agent.sessions import RuntimeBinding, SessionStore


@pytest.mark.anyio
async def test_chat_slash_commands_are_local_cli_projections(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    database = tmp_path / "agent.sqlite"
    store = SessionStore(database)
    session = store.create_session(
        RuntimeBinding(
            model_alias="fake-model",
            workspace_path=str(workspace.resolve()),
        )
    )
    store.close()
    model_calls: list[object] = []

    class _ModelControlPlane:
        default_model = "fake-model"

    class _Service:
        _model_registry = _ModelControlPlane()

        async def chat(self, request: object) -> object:
            model_calls.append(request)
            raise AssertionError("slash commands must not reach the model")

    class _Facade:
        checkpoint_db = database
        workspace_path = workspace

        def _agent_for_session(self, session_id: str) -> _Facade:
            assert session_id == session.session_id
            return self

        @asynccontextmanager
        async def _open_product_runtime(self, **_kwargs: object):
            yield _Service()

    commands = iter(
        [
            "/status",
            "/sessions",
            "/new",
            "/status",
            "/help",
            "/unknown",
            "/exit",
        ]
    )
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(commands))

    await cli._chat_facade_session(
        _Facade(),
        agent_type="generic",
        requested_model=None,
        budget=None,
        session_id=session.session_id,
    )

    output = capsys.readouterr().out
    assert f"Session: {session.session_id}" in output
    assert f"* {session.session_id}" in output
    assert "已开始新的 Session" in output
    assert "Session: (new)" in output
    assert "/sessions" in output
    assert "未知命令: /unknown" in output
    assert model_calls == []


@pytest.mark.anyio
async def test_chat_projects_max_turns_to_each_model_turn(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    requests: list[AgentRunRequest] = []

    class _ModelControlPlane:
        default_model = "fake-model"

    class _Service:
        _model_registry = _ModelControlPlane()

        async def chat(self, request: AgentRunRequest) -> AgentRunResult:
            requests.append(request)
            run_id = str(request.run_id)
            return AgentRunResult(
                run_id=run_id,
                thread_id=run_id,
                session_id=str(uuid4()),
                status="done",
                final_answer="bounded",
            )

    class _Facade:
        checkpoint_db = tmp_path / "agent.sqlite"
        workspace_path = workspace

        @asynccontextmanager
        async def _open_product_runtime(self, **_kwargs: object):
            yield _Service()

    commands = iter(["hello", "/exit"])
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(commands))

    await cli._chat_facade_session(
        _Facade(),
        agent_type="generic",
        requested_model=None,
        budget=None,
        max_turns=3,
    )

    assert len(requests) == 1
    assert requests[0].max_turns == 3

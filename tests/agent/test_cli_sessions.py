from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest
from langgraph.checkpoint.base import empty_checkpoint
from pytest import MonkeyPatch
from typer.testing import CliRunner

import rag.agent.cli as agent_cli
from rag.agent.core.checkpointing import (
    aclose_agent_checkpointer,
    create_agent_checkpointer,
)
from rag.agent.sessions import (
    RuntimeBinding,
    SessionNotFoundError,
    SessionStore,
    TurnStatus,
)

runner = CliRunner()


def _binding(workspace: Path) -> RuntimeBinding:
    return RuntimeBinding(
        model_alias="qwen3_5_9b_mlx_4bit",
        workspace_path=str(workspace.resolve()),
    )


def test_session_commands_list_show_archive_and_unarchive(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.chdir(workspace)
    database = tmp_path / "agent.sqlite"
    store = SessionStore(database)
    session = store.create_session(_binding(workspace))
    turn = store.begin_turn(session.session_id, "remember alpha")
    store.mark_terminal(turn.turn_id, TurnStatus.COMPLETED)
    store.close()

    listed = runner.invoke(
        agent_cli.agent_app,
        ["session", "list", "--checkpoint-db", str(database)],
    )
    shown = runner.invoke(
        agent_cli.agent_app,
        ["session", "show", session.session_id, "--checkpoint-db", str(database)],
    )
    archived = runner.invoke(
        agent_cli.agent_app,
        ["session", "archive", session.session_id, "--checkpoint-db", str(database)],
    )
    hidden = runner.invoke(
        agent_cli.agent_app,
        ["session", "list", "--checkpoint-db", str(database)],
    )
    archived_list = runner.invoke(
        agent_cli.agent_app,
        ["session", "list", "--archived", "--checkpoint-db", str(database)],
    )
    restored = runner.invoke(
        agent_cli.agent_app,
        ["session", "unarchive", session.session_id, "--checkpoint-db", str(database)],
    )

    assert listed.exit_code == 0
    assert session.session_id in listed.stdout
    assert shown.exit_code == 0
    assert turn.turn_id in shown.stdout
    assert "remember alpha" in shown.stdout
    assert "agent chat --session-id" in shown.stdout
    assert archived.exit_code == 0
    assert session.session_id not in hidden.stdout
    assert session.session_id in archived_list.stdout
    assert restored.exit_code == 0


def test_session_delete_requires_confirmation_and_cleans_checkpoints(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    database = tmp_path / "agent.sqlite"
    store = SessionStore(database)
    session = store.create_session(_binding(tmp_path))
    turn = store.begin_turn(session.session_id, "done")
    store.mark_terminal(turn.turn_id, TurnStatus.COMPLETED)
    store.close()

    async def seed_checkpoint() -> dict[str, object]:
        checkpointer = create_agent_checkpointer(database)
        saved = await checkpointer.aput(
            {
                "configurable": {
                    "thread_id": turn.turn_id,
                    "checkpoint_ns": "",
                }
            },
            empty_checkpoint(),
            {},
            {},
        )
        await aclose_agent_checkpointer(checkpointer)
        return saved

    saved = asyncio.run(seed_checkpoint())

    refused = runner.invoke(
        agent_cli.agent_app,
        ["session", "delete", session.session_id, "--checkpoint-db", str(database)],
    )
    deleted = runner.invoke(
        agent_cli.agent_app,
        [
            "session",
            "delete",
            session.session_id,
            "--yes",
            "--checkpoint-db",
            str(database),
        ],
    )

    assert refused.exit_code == 2
    assert "--yes" in refused.stdout
    assert deleted.exit_code == 0
    with pytest.raises(SessionNotFoundError):
        SessionStore(database).get_session(session.session_id)

    async def checkpoint_is_gone() -> bool:
        checkpointer = create_agent_checkpointer(database)
        missing = await checkpointer.aget_tuple(saved) is None
        await aclose_agent_checkpointer(checkpointer)
        return missing

    assert asyncio.run(checkpoint_is_gone())


def test_chat_last_resolves_latest_workspace_session(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    database = tmp_path / "agent.sqlite"
    store = SessionStore(database)
    session = store.create_session(_binding(tmp_path))
    store.close()
    captured: dict[str, object] = {}
    facade = object()

    async def fake_chat_session(passed_facade: object, **kwargs: object) -> None:
        captured["facade"] = passed_facade
        captured.update(kwargs)

    monkeypatch.setattr(agent_cli, "_create_agent_facade", lambda **_kwargs: facade)
    monkeypatch.setattr(agent_cli, "_chat_facade_session", fake_chat_session)

    result = runner.invoke(
        agent_cli.agent_app,
        ["chat", "--last", "--checkpoint-db", str(database)],
    )

    assert result.exit_code == 0
    assert captured["facade"] is facade
    assert captured["session_id"] == session.session_id


def test_resume_last_resolves_latest_recoverable_workspace_turn(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    database = tmp_path / "agent.sqlite"
    store = SessionStore(database)
    session = store.create_session(_binding(tmp_path))
    turn = store.begin_turn(session.session_id, "approve")
    store.mark_paused(turn.turn_id)
    store.close()
    captured: dict[str, object] = {}
    facade = object()

    async def fake_resume(passed_facade: object, **kwargs: object) -> object:
        captured["facade"] = passed_facade
        captured.update(kwargs)
        return SimpleNamespace(status="completed")

    monkeypatch.setattr(agent_cli, "_create_agent_facade", lambda **_kwargs: facade)
    monkeypatch.setattr(agent_cli, "_resume_facade_command", fake_resume)
    monkeypatch.setattr(agent_cli, "_display_agent_result", lambda *_args, **_kwargs: None)

    result = runner.invoke(
        agent_cli.agent_app,
        ["resume", "--last", "--checkpoint-db", str(database)],
    )

    assert result.exit_code == 0
    assert captured["facade"] is facade
    assert captured["turn_id"] == turn.turn_id


def test_resume_requires_turn_id_or_last() -> None:
    result = runner.invoke(agent_cli.agent_app, ["resume"])

    assert result.exit_code == 2
    assert "Turn ID" in result.stdout
    assert "--last" in result.stdout

from __future__ import annotations

from pathlib import Path
from uuid import UUID

import pytest

from rag.agent.core.messages import ModelMessage
from rag.agent.sessions import (
    RuntimeBinding,
    SessionBusyError,
    SessionStore,
    TurnStatus,
)


def _binding(workspace: Path) -> RuntimeBinding:
    return RuntimeBinding(
        agent_type="generic",
        model_alias="qwen3_5_9b_mlx_4bit",
        workspace_path=str(workspace),
        knowledge=("team-docs",),
        rag_storage_root=str(workspace / ".rag"),
    )


def test_session_store_uses_distinct_uuid_session_and_turn_ids(
    tmp_path: Path,
) -> None:
    store = SessionStore(tmp_path / "agent.sqlite")

    session = store.create_session(_binding(tmp_path))
    turn = store.begin_turn(session.session_id, "remember alpha")

    assert str(UUID(session.session_id)) == session.session_id
    assert str(UUID(turn.turn_id)) == turn.turn_id
    assert turn.turn_id != session.session_id
    assert turn.session_id == session.session_id
    assert turn.status is TurnStatus.RUNNING
    assert store.history(session.session_id) == (
        ModelMessage(role="user", content="remember alpha"),
    )


def test_session_store_allows_only_one_nonterminal_turn(
    tmp_path: Path,
) -> None:
    store = SessionStore(tmp_path / "agent.sqlite")
    session = store.create_session(_binding(tmp_path))
    first = store.begin_turn(session.session_id, "first")

    with pytest.raises(SessionBusyError, match=first.turn_id):
        store.begin_turn(session.session_id, "overlap")

    store.mark_terminal(first.turn_id, TurnStatus.COMPLETED)
    second = store.begin_turn(session.session_id, "second")

    assert second.turn_id != first.turn_id
    assert second.ordinal == 2


def test_session_store_reloads_runtime_metadata_and_canonical_history(
    tmp_path: Path,
) -> None:
    database = tmp_path / "agent.sqlite"
    first_store = SessionStore(database)
    session = first_store.create_session(_binding(tmp_path))
    turn = first_store.begin_turn(session.session_id, "remember alpha")
    transcript = (
        ModelMessage(role="user", content="remember alpha"),
        ModelMessage(role="assistant", content="remembered"),
    )
    first_store.sync_turn_messages(turn.turn_id, transcript)
    first_store.mark_terminal(turn.turn_id, TurnStatus.COMPLETED)
    first_store.close()

    restored_store = SessionStore(database)
    restored_session = restored_store.get_session(session.session_id)
    restored_turn = restored_store.get_turn(turn.turn_id)

    assert restored_session.runtime == _binding(tmp_path)
    assert restored_turn.session_id == session.session_id
    assert restored_turn.status is TurnStatus.COMPLETED
    assert restored_store.history(session.session_id) == transcript


def test_sync_turn_messages_is_append_only_and_idempotent(
    tmp_path: Path,
) -> None:
    store = SessionStore(tmp_path / "agent.sqlite")
    session = store.create_session(_binding(tmp_path))
    turn = store.begin_turn(session.session_id, "hello")
    messages = (
        ModelMessage(role="user", content="hello"),
        ModelMessage(role="assistant", content="hi"),
    )

    store.sync_turn_messages(turn.turn_id, messages)
    revision = store.get_session(session.session_id).history_revision
    store.sync_turn_messages(turn.turn_id, messages)

    assert store.get_session(session.session_id).history_revision == revision
    with pytest.raises(RuntimeError, match="canonical history conflict"):
        store.sync_turn_messages(
            turn.turn_id,
            (
                ModelMessage(role="user", content="changed"),
                ModelMessage(role="assistant", content="hi"),
            ),
        )

from __future__ import annotations

from pathlib import Path
from uuid import UUID

import pytest

from rag.agent.core.messages import ModelMessage
from rag.agent.sessions import (
    RuntimeBinding,
    SessionBusyError,
    SessionStore,
    TurnStateError,
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


def test_runtime_binding_never_persists_connection_secrets() -> None:
    binding = RuntimeBinding.model_validate(
        {
            "model_alias": "qwen3_5_9b_mlx_4bit",
            "vector_dsn": "postgresql://user:secret@example.invalid/db",
        }
    )

    assert "secret" not in binding.model_dump_json()
    assert "vector_dsn" not in binding.model_dump()


def test_session_runtime_is_frozen_after_first_turn(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "agent.sqlite")
    session = store.create_session(RuntimeBinding())
    initialized = store.initialize_session_runtime(
        session.session_id,
        RuntimeBinding(workspace_path=str(tmp_path)),
    )
    store.begin_turn(initialized.session_id, "hello")

    with pytest.raises(TurnStateError, match="frozen after its first Turn"):
        store.initialize_session_runtime(
            initialized.session_id,
            RuntimeBinding(workspace_path=str(tmp_path / "other")),
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


def test_paused_turn_blocks_chat_and_can_be_claimed_for_resume(
    tmp_path: Path,
) -> None:
    store = SessionStore(tmp_path / "agent.sqlite")
    session = store.create_session(_binding(tmp_path))
    turn = store.begin_turn(
        session.session_id,
        "first",
        lease_owner="worker-a",
    )

    paused = store.mark_paused(turn.turn_id)

    assert paused.status is TurnStatus.PAUSED
    assert paused.lease_owner is None
    with pytest.raises(SessionBusyError, match="active Turn"):
        store.begin_turn(session.session_id, "must fail")

    claimed = store.claim_for_resume(
        turn.turn_id,
        lease_owner="worker-b",
    )

    assert claimed.status is TurnStatus.RUNNING
    assert claimed.lease_owner == "worker-b"


def test_active_or_terminal_turn_cannot_be_resumed(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "agent.sqlite")
    session = store.create_session(_binding(tmp_path))
    active = store.begin_turn(
        session.session_id,
        "active",
        lease_owner="worker-a",
    )

    with pytest.raises(TurnStateError, match="still running"):
        store.claim_for_resume(active.turn_id, lease_owner="worker-b")

    store.mark_terminal(active.turn_id, TurnStatus.COMPLETED)
    with pytest.raises(TurnStateError, match="completed"):
        store.claim_for_resume(active.turn_id, lease_owner="worker-b")


def test_expired_running_turn_is_claimable_after_process_loss(
    tmp_path: Path,
) -> None:
    store = SessionStore(tmp_path / "agent.sqlite")
    session = store.create_session(_binding(tmp_path))
    turn = store.begin_turn(
        session.session_id,
        "crash me",
        lease_owner="dead-worker",
        lease_seconds=0.001,
    )

    claimed = store.claim_for_resume(
        turn.turn_id,
        lease_owner="replacement",
        now=turn.lease_expires_at,
    )

    assert claimed.status is TurnStatus.RUNNING
    assert claimed.lease_owner == "replacement"


def test_running_turn_lease_can_be_renewed_by_its_owner(
    tmp_path: Path,
) -> None:
    store = SessionStore(tmp_path / "agent.sqlite")
    session = store.create_session(_binding(tmp_path))
    turn = store.begin_turn(
        session.session_id,
        "long operation",
        lease_owner="worker-a",
        lease_seconds=1.0,
    )
    assert turn.lease_expires_at is not None

    renewed = store.renew_lease(
        turn.turn_id,
        lease_owner="worker-a",
        lease_seconds=1.0,
        now=turn.lease_expires_at - 0.25,
    )

    assert renewed.lease_expires_at == turn.lease_expires_at + 0.75
    with pytest.raises(TurnStateError, match="owned by another worker"):
        store.renew_lease(
            turn.turn_id,
            lease_owner="worker-b",
        )

from __future__ import annotations

from pathlib import Path
from uuid import UUID

import pytest

from rag.agent.core.messages import ModelMessage
from rag.agent.turns import (
    RuntimeBinding,
    TurnStateError,
    TurnStatus,
    TurnStore,
)


def _runtime(workspace: Path, *, model: str = "test-model") -> RuntimeBinding:
    return RuntimeBinding(
        model_alias=model,
        workspace_path=str(workspace.resolve()),
    )


def test_turn_store_links_followups_without_session(tmp_path: Path) -> None:
    store = TurnStore(tmp_path / "agent.sqlite")
    runtime = _runtime(tmp_path)
    first = store.begin_turn("remember alpha", runtime)
    store.sync_turn_messages(
        first.turn_id,
        [
            ModelMessage(role="user", content="remember alpha"),
            ModelMessage(role="assistant", content="alpha"),
        ],
    )
    store.mark_terminal(first.turn_id, TurnStatus.COMPLETED)
    second = store.begin_turn(
        "what did I say?",
        runtime,
        previous_turn_id=first.turn_id,
    )

    assert str(UUID(first.turn_id)) == first.turn_id
    assert first.previous_turn_id is None
    assert second.previous_turn_id == first.turn_id
    assert store.history_before_turn(second.turn_id) == (
        ModelMessage(role="user", content="remember alpha"),
        ModelMessage(role="assistant", content="alpha"),
    )
    assert store.turn_history(second.turn_id) == (
        ModelMessage(role="user", content="what did I say?"),
    )
    store.close()


def test_followup_requires_terminal_predecessor_and_same_runtime(tmp_path: Path) -> None:
    store = TurnStore(tmp_path / "agent.sqlite")
    runtime = _runtime(tmp_path)
    first = store.begin_turn("first", runtime)

    with pytest.raises(TurnStateError, match="only terminal Turns"):
        store.begin_turn("too early", runtime, previous_turn_id=first.turn_id)

    store.mark_terminal(first.turn_id, TurnStatus.COMPLETED)
    with pytest.raises(TurnStateError, match="runtime does not match"):
        store.begin_turn(
            "different runtime",
            _runtime(tmp_path, model="other-model"),
            previous_turn_id=first.turn_id,
        )
    store.close()


def test_turn_transcript_sync_is_append_only(tmp_path: Path) -> None:
    store = TurnStore(tmp_path / "agent.sqlite")
    turn = store.begin_turn("hello", _runtime(tmp_path))
    transcript = [
        ModelMessage(role="user", content="hello"),
        ModelMessage(role="assistant", content="hi"),
    ]
    store.sync_turn_messages(turn.turn_id, transcript)
    store.sync_turn_messages(turn.turn_id, transcript)

    with pytest.raises(RuntimeError, match="canonical history conflict"):
        store.sync_turn_messages(
            turn.turn_id,
            [
                ModelMessage(role="user", content="changed"),
                ModelMessage(role="assistant", content="hi"),
            ],
        )
    assert store.turn_history(turn.turn_id) == tuple(transcript)
    store.close()


def test_turn_store_persists_runtime_lineage_and_messages(tmp_path: Path) -> None:
    database = tmp_path / "agent.sqlite"
    runtime = _runtime(tmp_path)
    first_store = TurnStore(database)
    first = first_store.begin_turn("first", runtime)
    first_store.sync_turn_messages(
        first.turn_id,
        [
            ModelMessage(role="user", content="first"),
            ModelMessage(role="assistant", content="answer"),
        ],
    )
    first_store.mark_terminal(first.turn_id, TurnStatus.COMPLETED)
    second = first_store.begin_turn("second", runtime, previous_turn_id=first.turn_id)
    first_store.close()

    restored = TurnStore(database)
    assert restored.get_turn(second.turn_id).previous_turn_id == first.turn_id
    assert restored.get_turn(first.turn_id).runtime == runtime
    assert restored.history_before_turn(second.turn_id)[-1].content == "answer"
    restored.close()


def test_turn_status_and_resume_lease_lifecycle(tmp_path: Path) -> None:
    store = TurnStore(tmp_path / "agent.sqlite")
    turn = store.begin_turn(
        "approve",
        _runtime(tmp_path),
        lease_owner="owner-a",
        lease_seconds=1,
    )
    paused = store.mark_paused(turn.turn_id)
    assert paused.status is TurnStatus.PAUSED
    claimed = store.claim_for_resume(
        turn.turn_id,
        lease_owner="owner-b",
        lease_seconds=10,
    )
    assert claimed.status is TurnStatus.RUNNING
    assert claimed.lease_owner == "owner-b"
    renewed = store.renew_lease(
        turn.turn_id,
        lease_owner="owner-b",
        lease_seconds=20,
    )
    assert renewed.lease_expires_at is not None
    completed = store.mark_terminal(turn.turn_id, TurnStatus.COMPLETED)
    assert completed.status is TurnStatus.COMPLETED
    assert completed.lease_owner is None
    store.close()


def test_prepare_resume_normalizes_expired_running_turn(tmp_path: Path) -> None:
    store = TurnStore(tmp_path / "agent.sqlite")
    turn = store.begin_turn(
        "interrupted",
        _runtime(tmp_path),
        lease_owner="dead-worker",
        lease_seconds=1,
    )
    prepared = store.prepare_turn_for_resume(
        turn.turn_id,
        now=(turn.lease_expires_at or 0) + 1,
    )
    assert prepared.status is TurnStatus.INTERRUPTED
    assert prepared.lease_owner is None
    store.close()


def test_latest_turn_queries_use_turn_runtime_not_session_join(tmp_path: Path) -> None:
    store = TurnStore(tmp_path / "agent.sqlite")
    runtime = _runtime(tmp_path)
    completed = store.begin_turn("done", runtime)
    store.mark_terminal(completed.turn_id, TurnStatus.COMPLETED)
    paused = store.begin_turn("paused", runtime)
    store.mark_paused(paused.turn_id)

    assert store.latest_turn(workspace_path=tmp_path).turn_id == completed.turn_id
    assert store.latest_resumable_turn(workspace_path=tmp_path).turn_id == paused.turn_id
    assert [item.turn_id for item in store.list_turns(workspace_path=tmp_path)] == [
        paused.turn_id,
        completed.turn_id,
    ]
    store.close()

from __future__ import annotations

import json
import sqlite3
import threading
import time
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, cast
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict

from rag.agent.core.messages import (
    ModelMessage,
    ToolCall,
    canonical_json_text,
    model_message_payload,
    snapshot_model_message,
)


class TurnStatus(StrEnum):
    RUNNING = "running"
    PAUSED = "paused"
    INTERRUPTED = "interrupted"
    COMPLETED = "completed"
    FAILED = "failed"


class SessionNotFoundError(LookupError):
    pass


class TurnNotFoundError(LookupError):
    pass


class SessionBusyError(RuntimeError):
    pass


class TurnStateError(RuntimeError):
    pass


class RuntimeBinding(BaseModel):
    """Persisted, secret-free inputs needed to rebuild one product runtime."""

    model_config = ConfigDict(frozen=True)

    agent_type: str = "generic"
    model_alias: str | None = None
    workspace_path: str | None = None
    knowledge: tuple[str, ...] = ()
    rag_storage_root: str = ".rag"
    embedding_model_alias: str | None = None
    reranker_model_alias: str | None = None
    vector_backend: str = "milvus"
    vector_dsn: str | None = None
    vector_namespace: str | None = None
    vector_collection_prefix: str | None = None


@dataclass(frozen=True, slots=True)
class SessionRecord:
    session_id: str
    runtime: RuntimeBinding
    history_revision: int
    active_turn_id: str | None


@dataclass(frozen=True, slots=True)
class TurnRecord:
    turn_id: str
    session_id: str
    ordinal: int
    status: TurnStatus
    user_message: str
    runtime: RuntimeBinding
    checkpoint_id: str
    lease_owner: str | None
    lease_expires_at: float | None


class SessionStore:
    """Concrete SQLite store for Session, Turn, and canonical history."""

    def __init__(self, path: Path | str | None = None) -> None:
        self._path = None if path is None else Path(path)
        if self._path is not None:
            self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._connection = sqlite3.connect(
            ":memory:" if self._path is None else str(self._path),
            timeout=30.0,
            check_same_thread=False,
        )
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA foreign_keys = ON")
        if self._path is not None:
            self._connection.execute("PRAGMA journal_mode = WAL")
        self._create_schema()

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def create_session(
        self,
        runtime: RuntimeBinding,
        *,
        session_id: str | None = None,
    ) -> SessionRecord:
        effective_id = _uuid_text(session_id)
        now = time.time()
        with self._lock, self._connection:
            self._connection.execute(
                """
                INSERT INTO agent_sessions (
                    session_id, runtime_json, history_revision,
                    active_turn_id, created_at, updated_at
                ) VALUES (?, ?, 0, NULL, ?, ?)
                """,
                (effective_id, runtime.model_dump_json(), now, now),
            )
        return self.get_session(effective_id)

    def get_session(self, session_id: str) -> SessionRecord:
        with self._lock:
            row = self._connection.execute(
                """
                SELECT session_id, runtime_json, history_revision,
                       active_turn_id
                FROM agent_sessions
                WHERE session_id = ?
                """,
                (session_id,),
            ).fetchone()
        if row is None:
            raise SessionNotFoundError(f"Session not found: {session_id}")
        return _session_record(row)

    def get_turn(self, turn_id: str) -> TurnRecord:
        with self._lock:
            row = self._connection.execute(
                """
                SELECT turn_id, session_id, ordinal, status, user_message,
                       runtime_json, checkpoint_id, lease_owner,
                       lease_expires_at
                FROM agent_turns
                WHERE turn_id = ?
                """,
                (turn_id,),
            ).fetchone()
        if row is None:
            raise TurnNotFoundError(f"Turn not found: {turn_id}")
        return _turn_record(row)

    def begin_turn(
        self,
        session_id: str,
        message: str,
        *,
        turn_id: str | None = None,
        lease_owner: str | None = None,
        lease_seconds: float = 300.0,
    ) -> TurnRecord:
        if not isinstance(message, str) or not message.strip():
            raise ValueError("Turn message must be a non-empty string")
        if lease_seconds <= 0:
            raise ValueError("lease_seconds must be positive")
        effective_turn_id = _uuid_text(turn_id)
        now = time.time()
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                session = self._connection.execute(
                    """
                    SELECT runtime_json, history_revision, active_turn_id
                    FROM agent_sessions
                    WHERE session_id = ?
                    """,
                    (session_id,),
                ).fetchone()
                if session is None:
                    raise SessionNotFoundError(
                        f"Session not found: {session_id}"
                    )
                active_turn_id = cast(str | None, session["active_turn_id"])
                if active_turn_id is not None:
                    raise SessionBusyError(
                        f"Session {session_id} already has active Turn "
                        f"{active_turn_id}"
                    )
                ordinal = int(
                    self._connection.execute(
                        """
                        SELECT COALESCE(MAX(ordinal), 0) + 1
                        FROM agent_turns
                        WHERE session_id = ?
                        """,
                        (session_id,),
                    ).fetchone()[0]
                )
                lease_expires_at = (
                    None if lease_owner is None else now + lease_seconds
                )
                self._connection.execute(
                    """
                    INSERT INTO agent_turns (
                        turn_id, session_id, ordinal, status, user_message,
                        runtime_json, checkpoint_id, lease_owner,
                        lease_expires_at, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        effective_turn_id,
                        session_id,
                        ordinal,
                        TurnStatus.RUNNING.value,
                        message,
                        session["runtime_json"],
                        effective_turn_id,
                        lease_owner,
                        lease_expires_at,
                        now,
                        now,
                    ),
                )
                revision = int(session["history_revision"]) + 1
                self._connection.execute(
                    """
                    INSERT INTO agent_session_events (
                        session_id, turn_id, event_index,
                        session_sequence, payload_json
                    ) VALUES (?, ?, 0, ?, ?)
                    """,
                    (
                        session_id,
                        effective_turn_id,
                        revision,
                        _message_json(
                            ModelMessage(role="user", content=message)
                        ),
                    ),
                )
                self._connection.execute(
                    """
                    UPDATE agent_sessions
                    SET active_turn_id = ?, history_revision = ?,
                        updated_at = ?
                    WHERE session_id = ?
                    """,
                    (effective_turn_id, revision, now, session_id),
                )
                self._connection.commit()
            except Exception:
                self._connection.rollback()
                raise
        return self.get_turn(effective_turn_id)

    def sync_turn_messages(
        self,
        turn_id: str,
        messages: tuple[ModelMessage, ...] | list[ModelMessage],
    ) -> None:
        payloads = tuple(_message_json(message) for message in messages)
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                turn = self._connection.execute(
                    """
                    SELECT session_id
                    FROM agent_turns
                    WHERE turn_id = ?
                    """,
                    (turn_id,),
                ).fetchone()
                if turn is None:
                    raise TurnNotFoundError(f"Turn not found: {turn_id}")
                existing_rows = self._connection.execute(
                    """
                    SELECT event_index, payload_json
                    FROM agent_session_events
                    WHERE turn_id = ?
                    ORDER BY event_index
                    """,
                    (turn_id,),
                ).fetchall()
                existing = tuple(str(row["payload_json"]) for row in existing_rows)
                if len(payloads) < len(existing) or payloads[: len(existing)] != existing:
                    raise RuntimeError(
                        f"canonical history conflict for Turn {turn_id}"
                    )
                if len(payloads) == len(existing):
                    self._connection.commit()
                    return
                session_id = str(turn["session_id"])
                session = self._connection.execute(
                    """
                    SELECT history_revision
                    FROM agent_sessions
                    WHERE session_id = ?
                    """,
                    (session_id,),
                ).fetchone()
                if session is None:
                    raise SessionNotFoundError(
                        f"Session not found: {session_id}"
                    )
                revision = int(session["history_revision"])
                for event_index in range(len(existing), len(payloads)):
                    revision += 1
                    self._connection.execute(
                        """
                        INSERT INTO agent_session_events (
                            session_id, turn_id, event_index,
                            session_sequence, payload_json
                        ) VALUES (?, ?, ?, ?, ?)
                        """,
                        (
                            session_id,
                            turn_id,
                            event_index,
                            revision,
                            payloads[event_index],
                        ),
                    )
                now = time.time()
                self._connection.execute(
                    """
                    UPDATE agent_sessions
                    SET history_revision = ?, updated_at = ?
                    WHERE session_id = ?
                    """,
                    (revision, now, session_id),
                )
                self._connection.execute(
                    """
                    UPDATE agent_turns SET updated_at = ? WHERE turn_id = ?
                    """,
                    (now, turn_id),
                )
                self._connection.commit()
            except Exception:
                self._connection.rollback()
                raise

    def history(self, session_id: str) -> tuple[ModelMessage, ...]:
        self.get_session(session_id)
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT event.payload_json
                FROM agent_session_events AS event
                JOIN agent_turns AS turn ON turn.turn_id = event.turn_id
                WHERE event.session_id = ?
                ORDER BY turn.ordinal, event.event_index
                """,
                (session_id,),
            ).fetchall()
        return tuple(_message_from_json(str(row["payload_json"])) for row in rows)

    def mark_terminal(self, turn_id: str, status: TurnStatus) -> TurnRecord:
        if status not in {TurnStatus.COMPLETED, TurnStatus.FAILED}:
            raise ValueError("terminal status must be completed or failed")
        now = time.time()
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                row = self._connection.execute(
                    """
                    SELECT session_id, status
                    FROM agent_turns
                    WHERE turn_id = ?
                    """,
                    (turn_id,),
                ).fetchone()
                if row is None:
                    raise TurnNotFoundError(f"Turn not found: {turn_id}")
                current = TurnStatus(str(row["status"]))
                if current in {TurnStatus.COMPLETED, TurnStatus.FAILED}:
                    raise TurnStateError(
                        f"Turn {turn_id} is already {current.value}"
                    )
                session_id = str(row["session_id"])
                self._connection.execute(
                    """
                    UPDATE agent_turns
                    SET status = ?, lease_owner = NULL,
                        lease_expires_at = NULL, updated_at = ?
                    WHERE turn_id = ?
                    """,
                    (status.value, now, turn_id),
                )
                self._connection.execute(
                    """
                    UPDATE agent_sessions
                    SET active_turn_id = NULL, updated_at = ?
                    WHERE session_id = ? AND active_turn_id = ?
                    """,
                    (now, session_id, turn_id),
                )
                self._connection.commit()
            except Exception:
                self._connection.rollback()
                raise
        return self.get_turn(turn_id)

    def _create_schema(self) -> None:
        with self._lock, self._connection:
            self._connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS agent_sessions (
                    session_id TEXT PRIMARY KEY,
                    runtime_json TEXT NOT NULL,
                    history_revision INTEGER NOT NULL DEFAULT 0,
                    active_turn_id TEXT,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS agent_turns (
                    turn_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    ordinal INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    user_message TEXT NOT NULL,
                    runtime_json TEXT NOT NULL,
                    checkpoint_id TEXT NOT NULL,
                    lease_owner TEXT,
                    lease_expires_at REAL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    UNIQUE(session_id, ordinal),
                    FOREIGN KEY(session_id) REFERENCES agent_sessions(session_id)
                );

                CREATE TABLE IF NOT EXISTS agent_session_events (
                    session_id TEXT NOT NULL,
                    turn_id TEXT NOT NULL,
                    event_index INTEGER NOT NULL,
                    session_sequence INTEGER NOT NULL,
                    payload_json TEXT NOT NULL,
                    PRIMARY KEY(turn_id, event_index),
                    UNIQUE(session_id, session_sequence),
                    FOREIGN KEY(session_id) REFERENCES agent_sessions(session_id),
                    FOREIGN KEY(turn_id) REFERENCES agent_turns(turn_id)
                );
                """
            )


def _uuid_text(value: str | None) -> str:
    if value is None:
        return str(uuid4())
    return str(UUID(value))


def _session_record(row: sqlite3.Row) -> SessionRecord:
    return SessionRecord(
        session_id=str(row["session_id"]),
        runtime=RuntimeBinding.model_validate_json(str(row["runtime_json"])),
        history_revision=int(row["history_revision"]),
        active_turn_id=cast(str | None, row["active_turn_id"]),
    )


def _turn_record(row: sqlite3.Row) -> TurnRecord:
    return TurnRecord(
        turn_id=str(row["turn_id"]),
        session_id=str(row["session_id"]),
        ordinal=int(row["ordinal"]),
        status=TurnStatus(str(row["status"])),
        user_message=str(row["user_message"]),
        runtime=RuntimeBinding.model_validate_json(str(row["runtime_json"])),
        checkpoint_id=str(row["checkpoint_id"]),
        lease_owner=cast(str | None, row["lease_owner"]),
        lease_expires_at=cast(float | None, row["lease_expires_at"]),
    )


def _message_json(message: ModelMessage) -> str:
    return canonical_json_text(
        cast(Any, model_message_payload(snapshot_model_message(message)))
    )


def _message_from_json(raw: str) -> ModelMessage:
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise RuntimeError("canonical history message must be an object")
    tool_calls_raw = payload.get("tool_calls", [])
    if not isinstance(tool_calls_raw, list):
        raise RuntimeError("canonical history tool_calls must be an array")
    tool_calls = tuple(
        ToolCall(
            id=str(item["id"]),
            name=str(item["name"]),
            input=cast(dict[str, Any], item["arguments"]),
        )
        for item in tool_calls_raw
        if isinstance(item, dict)
    )
    return snapshot_model_message(
        ModelMessage(
            role=cast(Any, payload.get("role")),
            content=str(payload.get("content", "")),
            tool_calls=tool_calls,
            tool_call_id=cast(str | None, payload.get("tool_call_id")),
        )
    )


__all__ = [
    "RuntimeBinding",
    "SessionBusyError",
    "SessionNotFoundError",
    "SessionRecord",
    "SessionStore",
    "TurnNotFoundError",
    "TurnRecord",
    "TurnStateError",
    "TurnStatus",
]

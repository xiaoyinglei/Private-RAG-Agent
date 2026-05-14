from __future__ import annotations

from pathlib import Path

import aiosqlite
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver


def create_agent_checkpointer(checkpoint_db: Path | str | None) -> BaseCheckpointSaver:
    if checkpoint_db is None:
        return MemorySaver()

    path = Path(checkpoint_db)
    path.parent.mkdir(parents=True, exist_ok=True)
    return AsyncSqliteSaver(aiosqlite.connect(str(path)))


async def aclose_agent_checkpointer(checkpointer: BaseCheckpointSaver) -> None:
    connection = getattr(checkpointer, "conn", None)
    if connection is not None and hasattr(connection, "close"):
        await connection.close()

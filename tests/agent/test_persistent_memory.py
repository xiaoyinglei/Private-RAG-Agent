from __future__ import annotations

from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from rag.agent.memory.persistent.extractor import MemoryExtractor
from rag.agent.memory.persistent.models import MemoryFile
from rag.agent.memory.persistent.selector import MemorySelector


class _StaticGateway:
    def __init__(self, response: str) -> None:
        self.response = response
        self.calls = 0

    async def agenerate_text(self, **_: object) -> SimpleNamespace:
        self.calls += 1
        return SimpleNamespace(value=self.response)


def _memory(name: str, memory_type: str = "project") -> MemoryFile:
    return MemoryFile(
        name=name,
        description=f"{name} description",
        memory_type=memory_type,
        content=f"{name} content",
    )


@pytest.mark.anyio
async def test_extractor_does_not_report_failed_writes() -> None:
    gateway = _StaticGateway(
        """---MEMORY---
name: project-alpha
description: Project alpha preferences
type: project
content:
  Use the direct agent path for spreadsheet smoke tests.
---END---"""
    )

    class Store:
        is_available = True

        def read_index(self) -> str:
            return ""

        def read_all_memories(self) -> list[MemoryFile]:
            return []

        def write_memory(self, memory: MemoryFile) -> bool:
            return False

    state = {
        "messages": [
            HumanMessage(content="start"),
            AIMessage(content="ok"),
            HumanMessage(content="remember this"),
            AIMessage(content="done"),
        ],
        "run_config": SimpleNamespace(run_id="run_test"),
    }

    written = await MemoryExtractor(llm_gateway=gateway).extract(  # type: ignore[arg-type]
        state=state,  # type: ignore[arg-type]
        store=Store(),  # type: ignore[arg-type]
    )

    assert written == []


@pytest.mark.anyio
async def test_selector_skips_llm_when_user_memories_fill_selection() -> None:
    memories = [_memory(f"user-{i}", memory_type="user") for i in range(6)]
    memories.append(_memory("project-extra"))
    index_content = "\n".join(memory.index_line() for memory in memories)
    gateway = _StaticGateway("project-extra")

    class Store:
        def read_all_memories(self) -> list[MemoryFile]:
            return memories

    selected = await MemorySelector(
        llm_gateway=gateway, max_selected=5, llm_threshold=0
    ).select(
        task="continue work",
        index_content=index_content,
        store=Store(),  # type: ignore[arg-type]
    )

    assert gateway.calls == 0
    assert [memory.name for memory in selected] == [
        "user-0",
        "user-1",
        "user-2",
        "user-3",
        "user-4",
    ]

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING
from uuid import uuid4

from rag.agent.core.definition import AgentRuntimePolicy, ToolPolicy
from rag.agent.memory.models import MemoryPolicy
from rag.schema.runtime import AccessPolicy

if TYPE_CHECKING:
    from rag.agent.memory.store import WorkspaceMemoryStore


@dataclass(frozen=True)
class AgentRunConfig:
    run_id: str
    thread_id: str
    max_depth: int
    access_policy: AccessPolicy
    max_turns: int | None = None
    agent_type: str | None = None
    max_context_tokens: int | None = None
    parent_run_id: str | None = None
    source_scope: tuple[str, ...] = ()
    deadline_iso: str | None = None
    trace_parent_id: str | None = None
    tool_policy: ToolPolicy = field(default_factory=ToolPolicy)
    memory_policy: MemoryPolicy = field(default_factory=MemoryPolicy)


def derive_child_config(parent: AgentRunConfig, child_def: AgentRuntimePolicy) -> AgentRunConfig:
    if parent.max_depth <= 0:
        raise RuntimeError(f"Agent nesting depth exceeded for {child_def.agent_type}")
    child_id = str(uuid4())
    return AgentRunConfig(
        run_id=child_id,
        thread_id=child_id,
        parent_run_id=parent.run_id,
        access_policy=parent.access_policy,
        source_scope=parent.source_scope,
        max_depth=parent.max_depth - 1,
        agent_type=child_def.agent_type,
        max_context_tokens=parent.max_context_tokens,
        tool_policy=child_def.tool_policy,
        memory_policy=parent.memory_policy,
    )



@dataclass
class AgentRuntimeHandles:
    cancellation: asyncio.Event
    memory_store: WorkspaceMemoryStore | None = None


class RunRegistry:
    _handles: dict[str, AgentRuntimeHandles] = {}

    @classmethod
    def get_or_create(cls, run_config: AgentRunConfig) -> AgentRuntimeHandles:
        if run_config.run_id not in cls._handles:
            cls._handles[run_config.run_id] = AgentRuntimeHandles(
                cancellation=asyncio.Event(),
            )
        return cls._handles[run_config.run_id]

    @classmethod
    def get(cls, run_id: str) -> AgentRuntimeHandles:
        return cls._handles[run_id]

    @classmethod
    def remove(cls, run_id: str) -> None:
        cls._handles.pop(run_id, None)


__all__ = [
    "AgentRunConfig",
    "AgentRuntimeHandles",
    "BudgetLedger",
    "RunRegistry",
    "derive_child_config",
]

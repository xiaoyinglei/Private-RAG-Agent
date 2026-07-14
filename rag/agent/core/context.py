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


class LLMBudgetLedger:
    """Async ledger for per-run LLM token budget accounting."""

    def __init__(self, *, total: int) -> None:
        self._total = total
        self._committed = 0
        self._reserved: dict[str, int] = {}
        self._lock = asyncio.Lock()

    async def reserve(self, lease_id: str, amount: int) -> bool:
        async with self._lock:
            if amount < 0:
                raise ValueError("amount must be non-negative")
            if self._committed + sum(self._reserved.values()) + amount > self._total:
                return False
            self._reserved[lease_id] = amount
            return True

    async def commit(self, lease_id: str, actual: int) -> int:
        async with self._lock:
            if actual < 0:
                raise ValueError("actual must be non-negative")
            reserved = self._reserved.pop(lease_id, 0)
            self._committed += actual
            return max(actual - reserved, 0)

    async def refund(self, lease_id: str) -> int:
        async with self._lock:
            return self._reserved.pop(lease_id, 0)

    async def remaining(self) -> int:
        async with self._lock:
            return max(self._total - self._committed - sum(self._reserved.values()), 0)

    async def committed(self) -> int:
        async with self._lock:
            return self._committed

    async def reserved(self) -> int:
        async with self._lock:
            return sum(self._reserved.values())


@dataclass(frozen=True)
class AgentRunConfig:
    run_id: str
    thread_id: str
    max_depth: int
    access_policy: AccessPolicy
    llm_budget_total: int | None = None
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
        llm_budget_total=parent.llm_budget_total,
        tool_policy=child_def.tool_policy,
        memory_policy=parent.memory_policy,
    )



@dataclass
class AgentRuntimeHandles:
    cancellation: asyncio.Event
    llm_budget_ledger: LLMBudgetLedger | None = None
    memory_store: WorkspaceMemoryStore | None = None


class RunRegistry:
    _handles: dict[str, AgentRuntimeHandles] = {}

    @classmethod
    def get_or_create(cls, run_config: AgentRunConfig) -> AgentRuntimeHandles:
        if run_config.run_id not in cls._handles:
            llm_budget_ledger = None
            if run_config.parent_run_id is not None:
                parent = cls._handles.get(run_config.parent_run_id)
                if parent is not None:
                    llm_budget_ledger = parent.llm_budget_ledger
            if llm_budget_ledger is None and run_config.llm_budget_total is not None:
                llm_budget_ledger = LLMBudgetLedger(total=run_config.llm_budget_total)
            cls._handles[run_config.run_id] = AgentRuntimeHandles(
                cancellation=asyncio.Event(),
                llm_budget_ledger=llm_budget_ledger,
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
    "LLMBudgetLedger",
    "RunRegistry",
    "derive_child_config",
]

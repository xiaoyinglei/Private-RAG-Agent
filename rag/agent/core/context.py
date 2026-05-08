from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from rag.agent.core.definition import ToolPolicy
from rag.schema.runtime import AccessPolicy, ExecutionLocationPreference


@dataclass(frozen=True)
class AgentRunConfig:
    run_id: str
    thread_id: str
    budget_total: int
    max_depth: int
    access_policy: AccessPolicy
    execution_location_preference: ExecutionLocationPreference
    parent_run_id: str | None = None
    source_scope: tuple[str, ...] = ()
    deadline_iso: str | None = None
    trace_parent_id: str | None = None
    budget_committed: int = 0
    budget_reserved: dict[str, int] = field(default_factory=dict)
    tool_policy: ToolPolicy = field(default_factory=ToolPolicy)


class BudgetLedger:
    def __init__(self, total: int) -> None:
        self._total = total
        self._lock = asyncio.Lock()
        self._reserved: dict[str, int] = {}
        self._committed = 0

    async def remaining(self) -> int:
        async with self._lock:
            return max(0, self._total - self._committed - sum(self._reserved.values()))

    async def reserve(self, lease_id: str, amount: int) -> bool:
        async with self._lock:
            current = max(0, self._total - self._committed - sum(self._reserved.values()))
            if amount > current:
                return False
            self._reserved[lease_id] = amount
            return True

    async def commit(self, lease_id: str, actual: int) -> int:
        async with self._lock:
            reserved = self._reserved.pop(lease_id, 0)
            overrun = max(0, actual - reserved)
            self._committed += actual
            return overrun

    async def refund(self, lease_id: str) -> int:
        async with self._lock:
            return self._reserved.pop(lease_id, 0)


@dataclass
class AgentRuntimeHandles:
    budget_ledger: BudgetLedger
    cancellation: asyncio.Event


class RuntimeRegistry:
    _handles: dict[str, AgentRuntimeHandles] = {}

    @classmethod
    def get_or_create(cls, run_config: AgentRunConfig) -> AgentRuntimeHandles:
        if run_config.run_id not in cls._handles:
            cls._handles[run_config.run_id] = AgentRuntimeHandles(
                budget_ledger=BudgetLedger(total=run_config.budget_total),
                cancellation=asyncio.Event(),
            )
        return cls._handles[run_config.run_id]

    @classmethod
    def get(cls, run_id: str) -> AgentRuntimeHandles:
        return cls._handles[run_id]

    @classmethod
    def remove(cls, run_id: str) -> None:
        cls._handles.pop(run_id, None)

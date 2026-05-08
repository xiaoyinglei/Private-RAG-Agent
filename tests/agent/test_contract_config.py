from __future__ import annotations

import asyncio

import pytest

from rag.agent.core.context import (
    AgentRunConfig,
    AgentRuntimeHandles,
    BudgetLedger,
    RuntimeRegistry,
)
from rag.agent.core.definition import ToolPolicy
from rag.schema.runtime import AccessPolicy, ExecutionLocationPreference


class TestBudgetLedger:
    @pytest.mark.anyio
    async def test_reserve_commit(self) -> None:
        ledger = BudgetLedger(total=1000)
        ok = await ledger.reserve("lease-1", 300)
        assert ok is True
        assert await ledger.remaining() == 700
        overrun = await ledger.commit("lease-1", 250)
        assert overrun == 0
        assert await ledger.remaining() == 750

    @pytest.mark.anyio
    async def test_reserve_rejects_over_budget(self) -> None:
        ledger = BudgetLedger(total=500)
        ok = await ledger.reserve("lease-1", 600)
        assert ok is False

    @pytest.mark.anyio
    async def test_refund_returns_tokens(self) -> None:
        ledger = BudgetLedger(total=1000)
        await ledger.reserve("lease-1", 300)
        refunded = await ledger.refund("lease-1")
        assert refunded == 300
        assert await ledger.remaining() == 1000

    @pytest.mark.anyio
    async def test_commit_records_overrun(self) -> None:
        ledger = BudgetLedger(total=1000)
        await ledger.reserve("lease-1", 200)
        overrun = await ledger.commit("lease-1", 350)
        assert overrun == 150
        assert await ledger.remaining() == 650

    @pytest.mark.anyio
    async def test_concurrent_reserve(self) -> None:
        ledger = BudgetLedger(total=500)

        async def reserve_300() -> bool:
            return await ledger.reserve("a", 300)

        async def reserve_300_b() -> bool:
            return await ledger.reserve("b", 300)

        results = await asyncio.gather(reserve_300(), reserve_300_b())
        assert sum(1 for result in results if result) == 1


class TestAgentRunConfig:
    def test_minimal_config(self) -> None:
        cfg = AgentRunConfig(
            run_id="r1",
            thread_id="t1",
            budget_total=10000,
            max_depth=2,
            access_policy=AccessPolicy.default(),
            execution_location_preference=ExecutionLocationPreference.LOCAL_FIRST,
        )
        assert cfg.run_id == "r1"
        assert cfg.max_depth == 2
        assert cfg.parent_run_id is None
        assert cfg.source_scope == ()

    def test_config_defaults(self) -> None:
        cfg = AgentRunConfig(
            run_id="r1",
            thread_id="t2",
            budget_total=5000,
            max_depth=1,
            access_policy=AccessPolicy.default(),
            execution_location_preference=ExecutionLocationPreference.LOCAL_FIRST,
        )
        assert cfg.deadline_iso is None
        assert cfg.budget_committed == 0
        assert isinstance(cfg.tool_policy, ToolPolicy)


class TestRuntimeRegistry:
    def test_get_or_create_initializes_handles(self) -> None:
        cfg = AgentRunConfig(
            run_id="reg-test",
            thread_id="t",
            budget_total=8000,
            max_depth=2,
            access_policy=AccessPolicy.default(),
            execution_location_preference=ExecutionLocationPreference.LOCAL_FIRST,
        )
        RuntimeRegistry.remove(cfg.run_id)
        handles = RuntimeRegistry.get_or_create(cfg)
        assert isinstance(handles, AgentRuntimeHandles)
        assert isinstance(handles.budget_ledger, BudgetLedger)
        assert isinstance(handles.cancellation, asyncio.Event)

    def test_get_or_create_returns_same_handles(self) -> None:
        cfg = AgentRunConfig(
            run_id="reg-test-2",
            thread_id="t",
            budget_total=8000,
            max_depth=2,
            access_policy=AccessPolicy.default(),
            execution_location_preference=ExecutionLocationPreference.LOCAL_FIRST,
        )
        RuntimeRegistry.remove(cfg.run_id)
        h1 = RuntimeRegistry.get_or_create(cfg)
        h2 = RuntimeRegistry.get_or_create(cfg)
        assert h1 is h2

    def test_remove_cleans_up(self) -> None:
        cfg = AgentRunConfig(
            run_id="reg-test-3",
            thread_id="t",
            budget_total=8000,
            max_depth=2,
            access_policy=AccessPolicy.default(),
            execution_location_preference=ExecutionLocationPreference.LOCAL_FIRST,
        )
        RuntimeRegistry.remove(cfg.run_id)
        RuntimeRegistry.get_or_create(cfg)
        RuntimeRegistry.remove("reg-test-3")
        h_new = RuntimeRegistry.get_or_create(cfg)
        assert h_new is not None

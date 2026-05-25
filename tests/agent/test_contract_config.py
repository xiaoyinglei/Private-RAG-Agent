from __future__ import annotations

import asyncio

import pytest

from rag.agent.core.context import (
    AgentRunConfig,
    AgentRuntimeHandles,
    BudgetLedger,
    RuntimeRegistry,
    derive_child_config,
)
from rag.agent.core.definition import AgentDefinition, ModelSelectionPolicy, ToolPolicy
from rag.agent.core.registry import AgentRegistry
from rag.schema.runtime import AccessPolicy, RuntimeMode


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
        )
        assert cfg.deadline_iso is None
        assert cfg.budget_committed == 0
        assert isinstance(cfg.tool_policy, ToolPolicy)


class TestDeriveChildConfig:
    def test_derive_child_config_inherits_parent_runtime_scope(self) -> None:
        parent = AgentRunConfig(
            run_id="parent",
            thread_id="parent-thread",
            budget_total=20000,
            max_depth=2,
            access_policy=AccessPolicy.default(),
            source_scope=("doc-a", "doc-b"),
        )
        child_def = AgentDefinition(
            agent_type="research",
            description="Research",
            system_prompt="Research",
            allowed_tools=["vector_search"],
            estimated_token_budget=3500,
            tool_policy=ToolPolicy(max_parallel_calls=1),
        )

        child = derive_child_config(parent, child_def)

        assert child.run_id
        assert child.run_id != parent.run_id
        assert child.thread_id == child.run_id
        assert child.parent_run_id == parent.run_id
        assert child.source_scope == parent.source_scope
        assert child.access_policy == parent.access_policy
        assert child.access_policy.allowed_runtimes == parent.access_policy.allowed_runtimes
        assert child.max_depth == 1
        assert child.budget_total == 3500
        assert child.budget_committed == 0
        assert child.budget_reserved == {}
        assert child.tool_policy.max_parallel_calls == 1

    def test_derive_child_config_inherits_parent_access_policy(self) -> None:
        parent = AgentRunConfig(
            run_id="parent-policy",
            thread_id="parent-policy-thread",
            budget_total=10000,
            max_depth=2,
            access_policy=AccessPolicy.default(),
        )
        child_def = AgentDefinition(
            agent_type="local_research",
            description="Local only",
            system_prompt="Local only",
            allowed_tools=[],
            access_policy=AccessPolicy(allowed_runtimes=frozenset({RuntimeMode.DEEP})),
        )

        child = derive_child_config(parent, child_def)

        assert child.access_policy.allowed_runtimes == frozenset({RuntimeMode.FAST, RuntimeMode.DEEP})

    def test_derive_child_config_rejects_exhausted_depth(self) -> None:
        parent = AgentRunConfig(
            run_id="parent-depth",
            thread_id="parent-depth-thread",
            budget_total=10000,
            max_depth=0,
            access_policy=AccessPolicy.default(),
        )
        child_def = AgentDefinition(
            agent_type="research",
            description="Research",
            system_prompt="Research",
            allowed_tools=[],
        )

        with pytest.raises(RuntimeError, match="Agent nesting depth exceeded"):
            derive_child_config(parent, child_def)


class TestRuntimeRegistry:
    def test_get_or_create_initializes_handles(self) -> None:
        cfg = AgentRunConfig(
            run_id="reg-test",
            thread_id="t",
            budget_total=8000,
            max_depth=2,
            access_policy=AccessPolicy.default(),
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
        )
        RuntimeRegistry.remove(cfg.run_id)
        RuntimeRegistry.get_or_create(cfg)
        RuntimeRegistry.remove("reg-test-3")
        h_new = RuntimeRegistry.get_or_create(cfg)
        assert h_new is not None


class TestAgentDefinition:
    def test_minimal_definition(self) -> None:
        ad = AgentDefinition(
            agent_type="research",
            description="Deep research agent",
            system_prompt="You are a research agent.",
            allowed_tools=["vector_search", "grounding"],
        )
        assert ad.agent_type == "research"
        assert ad.allowed_tools == ["vector_search", "grounding"]
        assert ad.model_selection.retrieval_hint_model is None  # 默认不绑定特定模型
        assert ad.max_iterations == 10
        assert ad.max_depth == 2
        assert ad.estimated_token_budget == 8000

    def test_definition_with_access_policy(self) -> None:
        policy = AccessPolicy.default()
        ad = AgentDefinition(
            agent_type="compare",
            description="Comparison agent",
            system_prompt="You compare documents.",
            allowed_tools=["vector_search", "llm_compare"],
            access_policy=policy,
            estimated_token_budget=12000,
        )
        assert ad.access_policy is policy
        assert ad.estimated_token_budget == 12000

    def test_tool_policy_defaults(self) -> None:
        tp = ToolPolicy()
        assert tp.max_parallel_calls == 4
        assert len(tp.require_confirmation_for) == 0
        assert len(tp.deny_tools) == 0

    def test_tool_policy_custom(self) -> None:
        tp = ToolPolicy(
            max_parallel_calls=2,
            require_confirmation_for=frozenset({"kg_upsert"}),
            deny_tools=frozenset({"web_search"}),
        )
        assert "kg_upsert" in tp.require_confirmation_for
        assert "web_search" in tp.deny_tools
        assert tp.max_parallel_calls == 2

    def test_model_selection_defaults(self) -> None:
        ms = ModelSelectionPolicy()
        assert ms.retrieval_hint_model is None
        assert ms.tool_decision_model is None
        assert ms.thinking is True
        assert ms.retrieval_hint_temperature == 0.0
        assert ms.tool_decision_temperature == 0.0
        assert ms.retrieval_hint_max_tokens is None
        assert ms.tool_decision_max_tokens is None


class TestAgentRegistry:
    def test_register_and_get(self) -> None:
        registry = AgentRegistry()
        ad = AgentDefinition(
            agent_type="test_research",
            description="Test agent",
            system_prompt="You are a test agent.",
            allowed_tools=["search"],
        )
        registry.register(ad)
        retrieved = registry.get("test_research")
        assert retrieved is ad

    def test_get_missing_raises(self) -> None:
        registry = AgentRegistry()
        with pytest.raises(KeyError, match="not found"):
            registry.get("nonexistent_agent_type")

    def test_list_all(self) -> None:
        registry = AgentRegistry()
        ad1 = AgentDefinition(
            agent_type="agent_a",
            description="A",
            system_prompt="A",
            allowed_tools=[],
        )
        registry.register(ad1)
        all_agents = registry.list_all()
        assert ad1 in all_agents

    def test_instances_are_isolated(self) -> None:
        first = AgentRegistry()
        second = AgentRegistry()
        ad = AgentDefinition(
            agent_type="isolated_agent",
            description="A",
            system_prompt="A",
            allowed_tools=[],
        )

        first.register(ad)

        assert first.get("isolated_agent") is ad
        with pytest.raises(KeyError, match="not found"):
            second.get("isolated_agent")

    def test_duplicate_registration_fails_unless_replace_is_explicit(self) -> None:
        registry = AgentRegistry()
        first = AgentDefinition(
            agent_type="dup_agent",
            description="A",
            system_prompt="A",
            allowed_tools=[],
        )
        replacement = AgentDefinition(
            agent_type="dup_agent",
            description="B",
            system_prompt="B",
            allowed_tools=[],
        )

        registry.register(first)
        with pytest.raises(ValueError, match="already registered"):
            registry.register(replacement)

        registry.register(replacement, replace=True)
        assert registry.get("dup_agent") is replacement

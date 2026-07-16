from __future__ import annotations

import asyncio

import pytest

from rag.agent.core.context import (
    AgentRunConfig,
    AgentRuntimeHandles,
    LLMBudgetLedger,
    RunRegistry,
    derive_child_config,
)
from rag.agent.core.definition import AgentRuntimePolicy, ModelSelectionPolicy, ToolPolicy
from rag.agent.core.registry import AgentRegistry
from rag.schema.runtime import AccessPolicy, RuntimeMode


class TestLLMBudgetLedger:
    @pytest.mark.anyio
    async def test_reserve_commit(self) -> None:
        ledger = LLMBudgetLedger(total=1000)
        ok = await ledger.reserve("lease-1", 300)
        assert ok is True
        assert await ledger.remaining() == 700
        overrun = await ledger.commit("lease-1", 250)
        assert overrun == 0
        assert await ledger.remaining() == 750

    @pytest.mark.anyio
    async def test_reserve_rejects_over_budget(self) -> None:
        ledger = LLMBudgetLedger(total=500)
        ok = await ledger.reserve("lease-1", 600)
        assert ok is False

    @pytest.mark.anyio
    async def test_refund_returns_tokens(self) -> None:
        ledger = LLMBudgetLedger(total=1000)
        await ledger.reserve("lease-1", 300)
        refunded = await ledger.refund("lease-1")
        assert refunded == 300
        assert await ledger.remaining() == 1000

    @pytest.mark.anyio
    async def test_commit_records_overrun(self) -> None:
        ledger = LLMBudgetLedger(total=1000)
        await ledger.reserve("lease-1", 200)
        overrun = await ledger.commit("lease-1", 350)
        assert overrun == 150
        assert await ledger.remaining() == 650

    @pytest.mark.anyio
    async def test_concurrent_reserve(self) -> None:
        ledger = LLMBudgetLedger(total=500)

        async def reserve_300() -> bool:
            return await ledger.reserve("a", 300)

        async def reserve_300_b() -> bool:
            return await ledger.reserve("b", 300)

        results = await asyncio.gather(reserve_300(), reserve_300_b())
        assert sum(1 for result in results if result) == 1

    @pytest.mark.anyio
    async def test_exposes_committed_and_reserved_totals(self) -> None:
        ledger = LLMBudgetLedger(total=500)
        assert await ledger.reserve("call-1", 200) is True
        assert await ledger.commit("call-1", 125) == 0
        assert await ledger.reserve("call-2", 50) is True

        assert await ledger.committed() == 125
        assert await ledger.reserved() == 50


class TestAgentRunConfig:
    def test_minimal_config(self) -> None:
        cfg = AgentRunConfig(
            run_id="r1",
            thread_id="t1",
            llm_budget_total=10000,
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
            llm_budget_total=5000,
            max_depth=1,
            access_policy=AccessPolicy.default(),
        )
        assert cfg.deadline_iso is None
        assert cfg.llm_budget_total == 5000
        assert isinstance(cfg.tool_policy, ToolPolicy)

    @pytest.mark.parametrize("max_turns", [0, -1, True])
    def test_config_rejects_invalid_max_turns(self, max_turns: object) -> None:
        with pytest.raises((TypeError, ValueError)):
            AgentRunConfig(
                run_id="invalid-turn-limit",
                thread_id="invalid-turn-limit",
                max_depth=1,
                access_policy=AccessPolicy.default(),
                max_turns=max_turns,  # type: ignore[arg-type]
            )


class TestDeriveChildConfig:
    def test_derive_child_config_inherits_parent_runtime_scope(self) -> None:
        parent = AgentRunConfig(
            run_id="parent",
            thread_id="parent-thread",
            llm_budget_total=20000,
            max_depth=2,
            access_policy=AccessPolicy.default(),
            source_scope=("doc-a", "doc-b"),
        )
        child_def = AgentRuntimePolicy.test_factory(
            agent_type="research",
            description="Research",
            system_prompt="Research",
            allowed_tools=["vector_search"],
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
        assert child.llm_budget_total == parent.llm_budget_total
        assert child.tool_policy.max_parallel_calls == 1

    @pytest.mark.anyio
    async def test_child_runtime_shares_parent_llm_token_ledger(self) -> None:
        parent = AgentRunConfig(
            run_id="parent-shared-budget",
            thread_id="parent-shared-budget",
            llm_budget_total=20_000,
            max_depth=2,
            access_policy=AccessPolicy.default(),
        )
        child = derive_child_config(
            parent,
            AgentRuntimePolicy.test_factory(
                agent_type="research",
                description="Research",
                system_prompt="Research",
                allowed_tools=[],
            ),
        )
        RunRegistry.remove(parent.run_id)
        RunRegistry.remove(child.run_id)
        parent_handles = RunRegistry.get_or_create(parent)
        child_handles = RunRegistry.get_or_create(child)

        assert child_handles.llm_budget_ledger is parent_handles.llm_budget_ledger
        assert await child_handles.llm_budget_ledger.reserve("child-call", 1_000)
        assert await parent_handles.llm_budget_ledger.remaining() == 19_000
        RunRegistry.remove(child.run_id)
        RunRegistry.remove(parent.run_id)

    def test_derive_child_config_inherits_parent_access_policy(self) -> None:
        parent = AgentRunConfig(
            run_id="parent-policy",
            thread_id="parent-policy-thread",
            llm_budget_total=10000,
            max_depth=2,
            access_policy=AccessPolicy.default(),
        )
        child_def = AgentRuntimePolicy.test_factory(
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
            llm_budget_total=10000,
            max_depth=0,
            access_policy=AccessPolicy.default(),
        )
        child_def = AgentRuntimePolicy.test_factory(
            agent_type="research",
            description="Research",
            system_prompt="Research",
            allowed_tools=[],
        )

        with pytest.raises(RuntimeError, match="Agent nesting depth exceeded"):
            derive_child_config(parent, child_def)


class TestRunRegistry:
    def test_get_or_create_initializes_handles(self) -> None:
        cfg = AgentRunConfig(
            run_id="reg-test",
            thread_id="t",
            llm_budget_total=8000,
            max_depth=2,
            access_policy=AccessPolicy.default(),
        )
        RunRegistry.remove(cfg.run_id)
        handles = RunRegistry.get_or_create(cfg)
        assert isinstance(handles, AgentRuntimeHandles)
        assert isinstance(handles.llm_budget_ledger, LLMBudgetLedger)
        assert isinstance(handles.cancellation, asyncio.Event)

    def test_get_or_create_returns_same_handles(self) -> None:
        cfg = AgentRunConfig(
            run_id="reg-test-2",
            thread_id="t",
            llm_budget_total=8000,
            max_depth=2,
            access_policy=AccessPolicy.default(),
        )
        RunRegistry.remove(cfg.run_id)
        h1 = RunRegistry.get_or_create(cfg)
        h2 = RunRegistry.get_or_create(cfg)
        assert h1 is h2

    def test_remove_cleans_up(self) -> None:
        cfg = AgentRunConfig(
            run_id="reg-test-3",
            thread_id="t",
            llm_budget_total=8000,
            max_depth=2,
            access_policy=AccessPolicy.default(),
        )
        RunRegistry.remove(cfg.run_id)
        RunRegistry.get_or_create(cfg)
        RunRegistry.remove("reg-test-3")
        h_new = RunRegistry.get_or_create(cfg)
        assert h_new is not None


class TestAgentRuntimePolicy:
    def test_minimal_definition(self) -> None:
        ad = AgentRuntimePolicy.test_factory(
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
        assert not hasattr(ad, "token_budget")
        assert ad.output_validation_max_retries == 2

    def test_definition_rejects_negative_output_validation_retries(self) -> None:
        with pytest.raises(
            ValueError,
            match="output_validation_max_retries must be non-negative",
        ):
            AgentRuntimePolicy.test_factory(
                agent_type="invalid_output_retry",
                description="Invalid",
                system_prompt="Invalid",
                allowed_tools=[],
                output_validation_max_retries=-1,
            )

    def test_definition_with_access_policy(self) -> None:
        policy = AccessPolicy.default()
        ad = AgentRuntimePolicy.test_factory(
            agent_type="compare",
            description="Comparison agent",
            system_prompt="You compare documents.",
            allowed_tools=["vector_search", "llm_compare"],
            access_policy=policy,
        )
        assert ad.access_policy_ceiling is policy

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

    @pytest.mark.parametrize("value", [0, -1, True])
    def test_tool_policy_rejects_invalid_parallel_limit(
        self,
        value: object,
    ) -> None:
        with pytest.raises((TypeError, ValueError)):
            ToolPolicy(max_parallel_calls=value)  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        ("field_name", "value"),
        [
            ("require_confirmation_for", frozenset({""})),
            ("deny_tools", frozenset({1})),
        ],
    )
    def test_tool_policy_rejects_invalid_tool_names(
        self,
        field_name: str,
        value: object,
    ) -> None:
        with pytest.raises((TypeError, ValueError)):
            ToolPolicy(**{field_name: value})

    def test_tool_policy_rejects_non_boolean_sandbox_approval(self) -> None:
        with pytest.raises(TypeError, match="auto_approve_sandboxed"):
            ToolPolicy(auto_approve_sandboxed=1)  # type: ignore[arg-type]

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
        ad = AgentRuntimePolicy.test_factory(
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
        ad1 = AgentRuntimePolicy.test_factory(
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
        ad = AgentRuntimePolicy.test_factory(
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
        first = AgentRuntimePolicy.test_factory(
            agent_type="dup_agent",
            description="A",
            system_prompt="A",
            allowed_tools=[],
        )
        replacement = AgentRuntimePolicy.test_factory(
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

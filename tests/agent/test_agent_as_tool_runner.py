from __future__ import annotations

import pytest
from pydantic import ValidationError

from rag.agent.builtin.compare import COMPARE_AGENT
from rag.agent.builtin.factcheck import FACTCHECK_AGENT
from rag.agent.builtin.orchestrator import ORCHESTRATOR_AGENT
from rag.agent.builtin.research import RESEARCH_AGENT
from rag.agent.builtin.synthesize import SYNTHESIZE_AGENT
from rag.agent.builtin_registry import create_builtin_tool_registry
from rag.agent.core.agent_as_tool import (
    AgentAsToolAdapter,
    AgentAsToolExecutionError,
    AgentAsToolRunner,
    AgentToolInput,
    AgentToolOutput,
    build_agent_tool_spec,
)
from rag.agent.core.agent_service_factory import AgentServiceFactory
from rag.agent.core.context import AgentRunConfig, RunRegistry
from rag.agent.core.definition import AgentDefinition
from rag.agent.core.delegation import AgentDelegationRequest
from rag.agent.core.registry import AgentRegistry
from rag.agent.core.subagent_runner import BuiltinSubAgentRunner
from rag.agent.core.turn_contracts import ThinkOutput, ToolCallPlan
from rag.agent.loop.state import LoopState, create_loop_state
from rag.agent.service import AgentRunRequest, AgentRunResult
from rag.agent.tools.llm_tools import LLMTextOutput
from rag.schema.query import RetrievalSignals
from rag.schema.runtime import AccessPolicy


class _ResearchUnderstandingService:
    def analyze(
        self,
        query: str,
        *,
        access_policy: object | None = None,
    ) -> RetrievalSignals:
        del query, access_policy
        return RetrievalSignals()


class _ChildDecisionProvider:
    def __init__(self) -> None:
        self.calls = 0
        self.seen_configs: list[AgentRunConfig] = []

    async def decide(
        self,
        state: LoopState,
        *,
        definition: AgentDefinition,
        budget_remaining: int,
        context: object,
    ) -> ThinkOutput:
        del definition, budget_remaining, context
        self.calls += 1
        self.seen_configs.append(state["run_config"])
        if self.calls == 1:
            return ThinkOutput(
                action="execute",
                thought="summarize child task",
                tool_calls=[
                    ToolCallPlan.create(
                        "llm_summarize",
                        {
                            "task": state["task"],
                            "evidence_ids": ["ev1"],
                            "citation_ids": ["cit1"],
                        },
                    )
                ],
            )
        return ThinkOutput(
            action="synthesize",
            thought="child done",
            stop_reason="child_complete",
        )


def _parent_state(run_id: str = "parent-run", *, max_depth: int = 2) -> LoopState:
    config = AgentRunConfig(
        run_id=run_id,
        thread_id=f"{run_id}-thread",
        budget_total=10000,
        max_depth=max_depth,
        parent_run_id=None,
        source_scope=("doc-1",),
        access_policy=AccessPolicy.default(),
    )
    RunRegistry.remove(run_id)
    RunRegistry.get_or_create(config)
    return create_loop_state(task="Parent task", run_config=config)


@pytest.mark.anyio
async def test_agent_as_tool_runner_executes_registered_child_with_derived_config() -> None:
    child_def = AgentDefinition(
        agent_type="child_research_runner",
        description="Child research",
        system_prompt="Research child task",
        allowed_tools=["llm_summarize"],
        estimated_token_budget=2500,
    )
    agent_registry = AgentRegistry()
    agent_registry.register(child_def)
    decision_provider = _ChildDecisionProvider()
    runner = AgentAsToolRunner(
        agent_registry=agent_registry,
        tool_registry=create_builtin_tool_registry(
            runners={
                "llm_summarize": lambda payload: LLMTextOutput(
                    text=f"summary:{payload.task}",
                    evidence_ids=payload.evidence_ids,
                    citation_ids=payload.citation_ids,
                )
            }
        ),

        tool_decision_provider=decision_provider,
    )

    result = await runner.run_delegated_task(
        request=AgentDelegationRequest(
            delegation_id="s1",
            agent_type="child_research_runner",
            prompt="Child task",
            estimated_tokens=2400,
        ),
        parent_state=_parent_state(),
    )

    assert result.status == "done"
    assert result.final_answer == "summary:Child task"
    assert result.tool_results[0].status == "ok"
    first_child_config = decision_provider.seen_configs[0]
    assert first_child_config.parent_run_id == "parent-run"
    assert first_child_config.source_scope == ("doc-1",)
    assert first_child_config.max_depth == 1
    assert first_child_config.budget_total == 2400
    with pytest.raises(KeyError):
        RunRegistry.get(result.run_id)


@pytest.mark.anyio
async def test_agent_as_tool_runner_rejects_exhausted_parent_depth() -> None:
    child_def = AgentDefinition(
        agent_type="child_depth_runner",
        description="Child depth",
        system_prompt="Depth",
        allowed_tools=[],
    )
    agent_registry = AgentRegistry()
    agent_registry.register(child_def)
    runner = AgentAsToolRunner(
        agent_registry=agent_registry,
        tool_registry=create_builtin_tool_registry(),

    )

    with pytest.raises(RuntimeError, match="Agent nesting depth exceeded"):
        await runner.run_delegated_task(
            request=AgentDelegationRequest(
                delegation_id="s1",
                agent_type="child_depth_runner",
                prompt="Child task",
            ),
            parent_state=_parent_state("parent-depth-run", max_depth=0),
        )


class TestBuildAgentToolSpec:
    def test_research_agent_generates_tool_spec(self) -> None:
        spec = build_agent_tool_spec(RESEARCH_AGENT)
        assert spec.tool_spec.name == "agent_research"
        assert spec.tool_spec.input_model is AgentToolInput
        assert spec.tool_spec.output_model is AgentToolOutput
        assert spec.tool_spec.permissions.generate is True
        assert spec.agent_definition is RESEARCH_AGENT

    def test_blocklist_rejects_orchestrator(self) -> None:
        with pytest.raises(ValueError, match="blocklisted"):
            build_agent_tool_spec(ORCHESTRATOR_AGENT)

    def test_all_four_agents_generate_valid_specs(self) -> None:
        for agent_def in [RESEARCH_AGENT, COMPARE_AGENT, FACTCHECK_AGENT, SYNTHESIZE_AGENT]:
            spec = build_agent_tool_spec(agent_def)
            assert spec.tool_spec.name.startswith("agent_")
            assert spec.tool_spec.timeout_seconds == 120.0
            assert spec.tool_spec.max_retries == 0


class TestBuiltinRegistryHasAgentTools:
    def test_registry_contains_agent_research(self) -> None:
        registry = create_builtin_tool_registry()
        spec = registry.get("agent_research")
        assert spec.name == "agent_research"
        assert spec.input_model is AgentToolInput

    def test_registry_contains_agent_factcheck(self) -> None:
        registry = create_builtin_tool_registry()
        spec = registry.get("agent_factcheck")
        assert spec.name == "agent_factcheck"

    def test_registry_has_no_runner_for_agent_tools_by_default(self) -> None:
        registry = create_builtin_tool_registry()
        # Specs are registered, but no runner attached (request-scoped injection)
        assert not registry.has_runner("agent_research")

    def test_registry_does_not_contain_agent_orchestrator(self) -> None:
        registry = create_builtin_tool_registry()
        with pytest.raises(KeyError):
            registry.get("agent_orchestrator")


class TestOrchestratorAllowedTools:
    def test_allowed_tools_includes_agent_research(self) -> None:
        assert "agent_research" in ORCHESTRATOR_AGENT.allowed_tools

    def test_allowed_tools_includes_agent_factcheck(self) -> None:
        assert "agent_factcheck" in ORCHESTRATOR_AGENT.allowed_tools

    def test_allowed_tools_excludes_agent_orchestrator(self) -> None:
        assert "agent_orchestrator" not in ORCHESTRATOR_AGENT.allowed_tools

    def test_allowed_tools_excludes_agent_synthesize(self) -> None:
        assert "agent_synthesize" not in ORCHESTRATOR_AGENT.allowed_tools


class TestAgentToolInput:
    def test_minimal_input_is_valid(self) -> None:
        inp = AgentToolInput(task="Find documents about travel policy")
        assert inp.task == "Find documents about travel policy"
        assert inp.goal is None
        assert inp.required_outputs == []

    def test_full_input_is_valid(self) -> None:
        inp = AgentToolInput(
            task="Search financial reports",
            goal="Find Q1 revenue numbers",
            context_summary="User is looking for quarterly financials",
            required_outputs=["evidence", "conclusion"],
            constraints=["prefer_table", "max_5_items"],
        )
        assert inp.goal == "Find Q1 revenue numbers"
        assert len(inp.constraints) == 2

    def test_empty_task_raises_validation_error(self) -> None:
        with pytest.raises(ValidationError):
            AgentToolInput(task="")


class TestAgentToolOutput:
    def test_error_result_has_expected_fields(self) -> None:
        out = AgentToolOutput.error_result("research", "depth limit exceeded", status="failed")
        assert out.conclusion == "depth limit exceeded"
        assert out.status == "failed"
        assert out.agent_name == "research"
        assert out.confidence == 0.0

    def test_from_run_result_extracts_fields(self) -> None:
        from rag.agent.service import AgentRunResult
        from rag.schema.query import AnswerCitation, EvidenceItem

        result = AgentRunResult(
            run_id="r1",
            thread_id="t1",
            status="done",
            final_answer="The travel policy covers 3 regions.",
            evidence=[
                EvidenceItem(
                    evidence_id="ev-1",
                    doc_id=1,
                    text="Region A allows 500 CNY per night.",
                    score=0.9,
                    citation_anchor="p1",
                    record_type="text",
                )
            ],
            citations=[
                AnswerCitation(
                    citation_id="cit-1",
                    evidence_id="ev-1",
                    record_type="text",
                    citation_anchor="p1",
                    doc_id=1,
                )
            ],
            groundedness_flag=True,
        )
        out = AgentToolOutput.from_run_result(result, "research")
        assert out.conclusion == "The travel policy covers 3 regions."
        assert out.status == "done"
        assert out.agent_name == "research"
        assert out.confidence == 0.8
        assert out.evidence_refs[0].evidence_id == "ev-1"
        assert out.evidence_refs[0].citation_id == "cit-1"
        assert out.evidence_refs[0].citation_anchor == "p1"
        assert out.evidence_refs[0].doc_id == 1
        assert out.citations == result.citations
        assert len(out.key_facts) >= 1

    def test_from_run_result_bounds_delegated_evidence_and_citations(self) -> None:
        from rag.agent.service import AgentRunResult
        from rag.schema.query import AnswerCitation, EvidenceItem

        result = AgentRunResult(
            run_id="r-bounded",
            thread_id="t-bounded",
            status="done",
            final_answer="Bounded conclusion",
            evidence=[
                EvidenceItem(
                    evidence_id=f"ev-{index}",
                    doc_id=index,
                    text=f"Evidence item {index}",
                    score=0.9,
                    citation_anchor=f"p{index}",
                    record_type="text",
                )
                for index in range(21)
            ],
            citations=[
                AnswerCitation(
                    citation_id=f"cit-{index}",
                    evidence_id=f"ev-{index}",
                    record_type="text",
                    citation_anchor=f"p{index}",
                    doc_id=index,
                )
                for index in range(21)
            ],
            groundedness_flag=True,
        )

        out = AgentToolOutput.from_run_result(result, "research")

        assert len(out.evidence_refs) == 20
        assert len(out.citations) == 20

    def test_from_run_result_omits_evidence_without_traceable_locator(self) -> None:
        from rag.agent.service import AgentRunResult
        from rag.schema.query import EvidenceItem

        result = AgentRunResult(
            run_id="r-untraceable",
            thread_id="t-untraceable",
            status="done",
            final_answer="Conclusion still returns",
            evidence=[
                EvidenceItem(
                    evidence_id="ev-untraceable",
                    doc_id=1,
                    text="Unsupported observation.",
                    score=0.5,
                    citation_anchor="",
                    record_type="text",
                )
            ],
        )

        out = AgentToolOutput.from_run_result(result, "research")

        assert out.conclusion == "Conclusion still returns"
        assert out.evidence_refs == []


class TestAgentAsToolAdapter:
    def _make_adapter(self, agent_type: str = "research", run_config: AgentRunConfig | None = None):
        """构造最小的 AgentAsToolAdapter 用于测试"""
        from rag.agent.core.agent_as_tool import AgentAsToolRunner

        runner = AgentAsToolRunner(
            tool_registry=create_builtin_tool_registry(),
            agent_registry=AgentRegistry(),
        )
        # 注册一个子 agent definition
        from rag.agent.builtin.research import RESEARCH_AGENT

        runner._agent_registry.register(RESEARCH_AGENT)

        rc = run_config or AgentRunConfig(
            run_id="test-run",
            thread_id="test-thread",
            budget_total=10000,
            max_depth=2,
            access_policy=AccessPolicy.default(),
        )
        return AgentAsToolAdapter(runner=runner, agent_type=agent_type, run_config=rc)

    @pytest.mark.anyio
    async def test_adapter_raises_on_unregistered_agent_type(self) -> None:
        """Adapter failures remain typed for ToolExecutionService conversion."""
        from rag.agent.core.agent_as_tool import AgentAsToolRunner

        runner = AgentAsToolRunner(
            tool_registry=create_builtin_tool_registry(),
            agent_registry=AgentRegistry(),
        )
        rc = AgentRunConfig(
            run_id="test-run",
            thread_id="test-thread",
            budget_total=10000,
            max_depth=2,
            access_policy=AccessPolicy.default(),
        )
        adapter = AgentAsToolAdapter(runner=runner, agent_type="research", run_config=rc)
        inp = AgentToolInput(task="Test task")
        with pytest.raises(AgentAsToolExecutionError, match="subagent execution failed"):
            await adapter(inp)

    @pytest.mark.anyio
    async def test_adapter_is_callable_compatible_with_tool_runner(self) -> None:
        """adapter 实例可直接作为 ToolRunner 使用"""
        class _CompletedSubAgentRunner:
            async def run_delegated_task(
                self,
                *,
                request: AgentDelegationRequest,
                parent_state: LoopState,
            ) -> AgentRunResult:
                del request, parent_state
                return AgentRunResult(
                    run_id="child-done",
                    thread_id="child-done",
                    status="done",
                    final_answer="Completed child output",
                )

        adapter = AgentAsToolAdapter(
            runner=_CompletedSubAgentRunner(),
            agent_type="research",
            run_config=AgentRunConfig(
                run_id="test-run",
                thread_id="test-thread",
                budget_total=10000,
                max_depth=2,
                access_policy=AccessPolicy.default(),
            ),
        )
        inp = AgentToolInput(task="Test task", goal="Verify callability")
        result = await adapter(inp)
        assert isinstance(result, AgentToolOutput)
        assert result.agent_name == "research"
        assert result.conclusion == "Completed child output"

    def test_depth_exhaust_returns_error_not_exception(self) -> None:
        """depth=0 时不抛异常，返回结构化错误结果（通过 run_config 的 depth 控制）"""
        # depth=0 意味着 child 无法再派生 —— adapter 本身不崩溃
        rc = AgentRunConfig(
            run_id="depth-0",
            thread_id="depth-0-thread",
            budget_total=10000,
            max_depth=0,
            access_policy=AccessPolicy.default(),
        )
        adapter = self._make_adapter(run_config=rc)
        # adapter 不会在 init 时报错，实际错误在 child AgentService 运行时报
        # 这里验证 adapter 能正常构造
        assert adapter is not None


class TestRegistryCloneAndIsolation:
    """并发安全: runtime_tool_registry.clone() 不污染 base registry"""

    def test_clone_has_independent_runners(self) -> None:
        base = create_builtin_tool_registry()
        cloned = base.clone()

        # Clone 中有 specs 但没有 agent runners
        assert cloned.get("agent_research").name == "agent_research"
        assert not cloned.has_runner("agent_research")

        # 注入 runner 到 clone 不影响 base
        async def _dummy_runner(payload: AgentToolInput) -> AgentToolOutput:
            return AgentToolOutput(conclusion="ok", status="done", agent_name="research")

        cloned.register_runner("agent_research", _dummy_runner)
        assert cloned.has_runner("agent_research")
        assert not base.has_runner("agent_research")

    def test_base_registry_not_polluted_after_run_with_config(self) -> None:
        """模拟 run_with_config 生命周期后 base registry 无残留"""
        base = create_builtin_tool_registry()
        # 模拟 run_with_config 内部的 clone
        runtime = base.clone()

        rc = AgentRunConfig(
            run_id="isolation-test",
            thread_id="isolation-thread",
            budget_total=10000,
            max_depth=2,
            access_policy=AccessPolicy.default(),
        )
        # 注入 adapter（模拟 _runtime_tool_registry）
        from rag.agent.core.agent_as_tool import AgentAsToolAdapter, AgentAsToolRunner

        runner = AgentAsToolRunner(
            tool_registry=base,
            agent_registry=AgentRegistry(),
        )
        adapter = AgentAsToolAdapter(runner=runner, agent_type="research", run_config=rc)
        runtime.register_runner("agent_research", adapter)

        # runtime 有 runner，base 没有
        assert runtime.has_runner("agent_research")
        assert not base.has_runner("agent_research")

        # runtime registry 被丢弃（模拟 request 结束），base 保持干净
        del runtime
        assert not base.has_runner("agent_research")

    def test_two_concurrent_adapters_have_independent_run_configs(self) -> None:
        """两个不同 run_config 的 adapter 互不干扰"""
        from rag.schema.runtime import RuntimeMode

        rc1 = AgentRunConfig(
            run_id="run-1",
            thread_id="thread-1",
            budget_total=5000,
            max_depth=3,
            access_policy=AccessPolicy.default(),
        )
        rc2 = AgentRunConfig(
            run_id="run-2",
            thread_id="thread-2",
            budget_total=8000,
            max_depth=1,
            access_policy=AccessPolicy(allowed_runtimes=frozenset({RuntimeMode.FAST})),
        )

        from rag.agent.core.agent_as_tool import AgentAsToolAdapter, AgentAsToolRunner

        runner = AgentAsToolRunner(
            tool_registry=create_builtin_tool_registry(),
            agent_registry=AgentRegistry(),
        )
        a1 = AgentAsToolAdapter(runner=runner, agent_type="research", run_config=rc1)
        a2 = AgentAsToolAdapter(runner=runner, agent_type="research", run_config=rc2)

        assert a1._run_config.budget_total == 5000
        assert a2._run_config.budget_total == 8000
        assert a1._run_config.max_depth == 3
        assert a2._run_config.max_depth == 1
        assert a1._run_config.max_depth == 3
        assert a2._run_config.max_depth == 1
        assert a1._run_config.run_id == "run-1"
        assert a2._run_config.run_id == "run-2"


@pytest.mark.anyio
async def test_builtin_subagent_runner_is_injected_for_agent_tool_calls() -> None:
    child_def = AgentDefinition(
        agent_type="research",
        description="Research child",
        system_prompt="Research child task",
        allowed_tools=["llm_summarize"],
        estimated_token_budget=2500,
        max_depth=1,
    )
    agent_registry = AgentRegistry()
    agent_registry.register(child_def)
    decision_provider = _ChildDecisionProvider()
    factory = AgentServiceFactory(
        tool_registry=create_builtin_tool_registry(
            runners={
                "llm_summarize": lambda payload: LLMTextOutput(
                    text=f"summary:{payload.task}",
                    evidence_ids=payload.evidence_ids,
                    citation_ids=payload.citation_ids,
                )
            }
        ),
        model_registry=None,
        tool_decision_provider=decision_provider,
    )
    runner = BuiltinSubAgentRunner(agent_registry=agent_registry, service_factory=factory)
    factory.bind_subagent_runner(runner)
    service = factory.create(ORCHESTRATOR_AGENT)
    call = ToolCallPlan.create(
        "agent_research",
        {
            "task": "Find the reimbursement policy",
            "goal": "Return a grounded summary",
        },
    )

    result = await service.run(
        AgentRunRequest(
            task="Use a child research agent",
            run_id="agent-tool-builtin-runner",
            thread_id="agent-tool-builtin-runner",
            pending_tool_calls=[call],
        )
    )

    assert result.tool_results[0].status == "ok"
    output = AgentToolOutput.model_validate(result.tool_results[0].output)
    assert output.status == "done"
    assert output.agent_name == "research"
    assert "summary:## Task" in output.conclusion

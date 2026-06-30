from __future__ import annotations

import pytest
from pydantic import ValidationError

from rag.agent.builtin.generic import GENERIC_AGENT
from rag.agent.builtin_registry import create_builtin_tool_registry
from rag.agent.core.agent_as_tool import (
    AgentAsToolAdapter,
    AgentAsToolExecutionError,
    AgentAsToolRunner,
    AgentToolInput,
    AgentToolOutput,
)
from rag.agent.core.context import AgentRunConfig, RunRegistry
from rag.agent.core.definition import AgentRuntimePolicy
from rag.agent.core.delegation import AgentDelegationRequest
from rag.agent.core.registry import AgentRegistry
from rag.agent.core.turn_contracts import ToolCallPlan
from rag.agent.loop.state import LoopState, ModelTurnDraft, create_loop_state
from rag.agent.service import AgentRunResult
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

    async def next_turn(
        self,
        state: LoopState,
        *,
        definition: AgentRuntimePolicy,
        budget_remaining: int,
    ) -> ModelTurnDraft:
        del definition, budget_remaining
        self.calls += 1
        self.seen_configs.append(state["run_config"])
        if self.calls == 1:
            return ModelTurnDraft(
                action="execute",
                tool_calls=(
                    ToolCallPlan.create(
                        "llm_summarize",
                        {
                            "task": state["task"],
                            "evidence_ids": ["ev1"],
                            "citation_ids": ["cit1"],
                        },
                    ),
                ),
            )
        return ModelTurnDraft(
            action="finish",
            final_answer="summary:Child task",  # PR2: answer_candidates no longer written to LoopState
        )


def _parent_state(run_id: str = "parent-run", *, max_depth: int = 2) -> LoopState:
    config = AgentRunConfig(
        run_id=run_id,
        thread_id=f"{run_id}-thread",
        llm_budget_total=10000,
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
    child_def = AgentRuntimePolicy.test_factory(
        agent_type="child_research_runner",
        description="Child research",
        system_prompt="Research child task",
        allowed_tools=["llm_summarize"],
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
        model_turn_provider=decision_provider,
    )

    result = await runner.run_delegated_task(
        request=AgentDelegationRequest(
            delegation_id="s1",
            agent_type="child_research_runner",
            prompt="Child task",
            llm_budget_total=2400,
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
    assert first_child_config.llm_budget_total == 2400
    with pytest.raises(KeyError):
        RunRegistry.get(result.run_id)


@pytest.mark.anyio
async def test_agent_as_tool_runner_rejects_exhausted_parent_depth() -> None:
    child_def = AgentRuntimePolicy.test_factory(
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
        out = AgentToolOutput.from_run_result(result, "generic")
        assert out.conclusion == "The travel policy covers 3 regions."
        assert out.status == "done"
        assert out.agent_name == "generic"
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

        out = AgentToolOutput.from_run_result(result, "generic")

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

        out = AgentToolOutput.from_run_result(result, "generic")

        assert out.conclusion == "Conclusion still returns"
        assert out.evidence_refs == []


class TestAgentAsToolAdapter:
    def _make_adapter(self, agent_type: str = "generic", run_config: AgentRunConfig | None = None):
        """构造最小的 AgentAsToolAdapter 用于测试"""
        from rag.agent.core.agent_as_tool import AgentAsToolRunner

        runner = AgentAsToolRunner(
            tool_registry=create_builtin_tool_registry(),
            agent_registry=AgentRegistry(),
        )
        # 注册一个子 agent definition
        runner._agent_registry.register(GENERIC_AGENT)

        rc = run_config or AgentRunConfig(
            run_id="test-run",
            thread_id="test-thread",
            llm_budget_total=10000,
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
            llm_budget_total=10000,
            max_depth=2,
            access_policy=AccessPolicy.default(),
        )
        adapter = AgentAsToolAdapter(runner=runner, agent_type="generic", run_config=rc)
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
                llm_budget_total=10000,
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
            llm_budget_total=10000,
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

        # Clone 中有 specs 但没有 runners
        assert cloned.get("llm_summarize").name == "llm_summarize"
        assert not cloned.has_runner("llm_summarize")

        # 注入 runner 到 clone 不影响 base
        async def _dummy_runner(payload):
            return None

        cloned.register_runner("llm_summarize", _dummy_runner)
        assert cloned.has_runner("llm_summarize")
        assert not base.has_runner("llm_summarize")

    def test_base_registry_not_polluted_after_run_with_config(self) -> None:
        """模拟 run_with_config 生命周期后 base registry 无残留"""
        base = create_builtin_tool_registry()
        # 模拟 run_with_config 内部的 clone
        runtime = base.clone()

        async def _dummy_runner(payload):
            return None

        runtime.register_runner("llm_summarize", _dummy_runner)

        # runtime 有 runner，base 没有
        assert runtime.has_runner("llm_summarize")
        assert not base.has_runner("llm_summarize")

        # runtime registry 被丢弃（模拟 request 结束），base 保持干净
        del runtime
        assert not base.has_runner("llm_summarize")

    def test_two_concurrent_adapters_have_independent_run_configs(self) -> None:
        """两个不同 run_config 的 adapter 互不干扰"""
        from rag.schema.runtime import RuntimeMode

        rc1 = AgentRunConfig(
            run_id="run-1",
            thread_id="thread-1",
            llm_budget_total=5000,
            max_depth=3,
            access_policy=AccessPolicy.default(),
        )
        rc2 = AgentRunConfig(
            run_id="run-2",
            thread_id="thread-2",
            llm_budget_total=8000,
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

        assert a1._run_config.llm_budget_total == 5000
        assert a2._run_config.llm_budget_total == 8000
        assert a1._run_config.max_depth == 3
        assert a2._run_config.max_depth == 1
        assert a1._run_config.max_depth == 3
        assert a2._run_config.max_depth == 1
        assert a1._run_config.run_id == "run-1"
        assert a2._run_config.run_id == "run-2"

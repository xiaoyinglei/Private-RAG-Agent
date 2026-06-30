from __future__ import annotations

import pytest

from rag.agent.builtin_registry import create_builtin_tool_registry
from rag.agent.core.agent_service_factory import AgentServiceFactory
from rag.agent.core.context import AgentRunConfig, RunRegistry
from rag.agent.core.definition import AgentRuntimePolicy
from rag.agent.core.delegation import AgentDelegationRequest
from rag.agent.core.registry import AgentRegistry
from rag.agent.core.subagent_runner import BuiltinSubAgentRunner
from rag.agent.core.turn_contracts import ToolCallPlan
from rag.agent.loop.state import LoopState, ModelTurnDraft, create_loop_state
from rag.agent.service import AgentRunResult
from rag.agent.tools.llm_tools import LLMTextOutput
from rag.schema.runtime import AccessPolicy


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
        # PR2: answer_candidates no longer written to LoopState
        if state["tool_results"]:
            latest = state["tool_results"][-1]
            if latest.output is not None:
                text = getattr(latest.output, "text", None)
                if text:
                    return ModelTurnDraft(action="finish", final_answer=text)
        return ModelTurnDraft(
            action="finish",
            final_answer="summary:Child task",
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
async def test_builtin_subagent_runner_returns_agent_run_result_with_derived_config() -> None:
    child_def = AgentRuntimePolicy.test_factory(
        agent_type="child_research_runner",
        description="Child research",
        system_prompt="Research child task",
        allowed_tools=["llm_summarize"],
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
        model_turn_provider=decision_provider,
    )
    runner = BuiltinSubAgentRunner(agent_registry=agent_registry, service_factory=factory)
    factory.bind_subagent_runner(runner)

    result = await runner.run_delegated_task(
        request=AgentDelegationRequest(
            delegation_id="s1",
            agent_type="child_research_runner",
            prompt="Child task",
            llm_budget_total=2400,
        ),
        parent_state=_parent_state(),
    )

    assert isinstance(result, AgentRunResult)
    assert result.status == "done"
    assert result.final_answer == "summary:Child task"
    first_child_config = decision_provider.seen_configs[0]
    assert first_child_config.parent_run_id == "parent-run"
    assert first_child_config.source_scope == ("doc-1",)
    assert first_child_config.max_depth == 1
    assert first_child_config.llm_budget_total == 2400
    with pytest.raises(KeyError):
        RunRegistry.get(result.run_id)


@pytest.mark.anyio
async def test_builtin_subagent_runner_rejects_exhausted_parent_depth() -> None:
    child_def = AgentRuntimePolicy.test_factory(
        agent_type="child_depth_runner",
        description="Child depth",
        system_prompt="Depth",
        allowed_tools=[],
    )
    agent_registry = AgentRegistry()
    agent_registry.register(child_def)
    factory = AgentServiceFactory(
        tool_registry=create_builtin_tool_registry(),
        model_registry=None,
    )
    runner = BuiltinSubAgentRunner(agent_registry=agent_registry, service_factory=factory)
    factory.bind_subagent_runner(runner)

    with pytest.raises(RuntimeError, match="Agent nesting depth exceeded"):
        await runner.run_delegated_task(
            request=AgentDelegationRequest(
                delegation_id="s1",
                agent_type="child_depth_runner",
                prompt="Child task",
            ),
            parent_state=_parent_state("parent-depth-run", max_depth=0),
        )

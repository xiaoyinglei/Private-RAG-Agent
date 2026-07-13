from __future__ import annotations

import pytest

from rag.agent.core.agent_service_factory import AgentServiceFactory
from rag.agent.core.context import AgentRunConfig, RunRegistry
from rag.agent.core.definition import AgentRuntimePolicy
from rag.agent.core.delegation import AgentDelegationRequest
from rag.agent.core.registry import AgentRegistry
from rag.agent.core.subagent_runner import BuiltinSubAgentRunner
from rag.agent.loop.state import LoopState, ModelTurnDraft
from rag.agent.service import AgentRunResult
from rag.agent.tools.registry import ToolRegistry
from rag.schema.runtime import AccessPolicy


class _ChildDecisionProvider:
    def __init__(self) -> None:
        self.seen_configs: list[AgentRunConfig] = []

    async def next_turn(
        self,
        state: LoopState,
        *,
        definition: AgentRuntimePolicy,
        budget_remaining: int,
    ) -> ModelTurnDraft:
        del definition, budget_remaining
        self.seen_configs.append(state["run_config"])
        return ModelTurnDraft(
            action="finish",
            final_answer=f"summary:{state['task']}",
        )


def _parent_config(
    run_id: str = "parent-run",
    *,
    max_depth: int = 2,
) -> AgentRunConfig:
    return AgentRunConfig(
        run_id=run_id,
        thread_id=f"{run_id}-thread",
        llm_budget_total=10_000,
        max_depth=max_depth,
        source_scope=("doc-1",),
        access_policy=AccessPolicy.default(),
    )


@pytest.mark.anyio
async def test_builtin_subagent_runner_uses_derived_child_config() -> None:
    child = AgentRuntimePolicy.test_factory(
        agent_type="child_research_runner",
        description="Child research",
        system_prompt="Research the bounded child task.",
        allowed_tools=[],
    )
    agents = AgentRegistry()
    agents.register(child)
    provider = _ChildDecisionProvider()
    factory = AgentServiceFactory(
        tool_registry=ToolRegistry(),
        model_turn_provider=provider,
    )
    runner = BuiltinSubAgentRunner(
        agent_registry=agents,
        service_factory=factory,
    )
    parent = _parent_config()

    result = await runner.run_delegated_task(
        request=AgentDelegationRequest(
            delegation_id="delegation-1",
            agent_type=child.agent_type,
            prompt="Child task",
            llm_budget_total=2400,
        ),
        parent_state={"run_config": parent},
    )

    assert isinstance(result, AgentRunResult)
    assert result.status == "done"
    assert result.final_answer == "summary:Child task"
    child_config = provider.seen_configs[0]
    assert child_config.parent_run_id == parent.run_id
    assert child_config.source_scope == ("doc-1",)
    assert child_config.max_depth == 1
    assert child_config.llm_budget_total == 2400
    with pytest.raises(KeyError):
        RunRegistry.get(result.run_id)


@pytest.mark.anyio
async def test_builtin_subagent_runner_rejects_exhausted_parent_depth() -> None:
    child = AgentRuntimePolicy.test_factory(
        agent_type="child_depth_runner",
        description="Child depth",
        system_prompt="Depth test.",
        allowed_tools=[],
    )
    agents = AgentRegistry()
    agents.register(child)
    runner = BuiltinSubAgentRunner(
        agent_registry=agents,
        service_factory=AgentServiceFactory(tool_registry=ToolRegistry()),
    )

    with pytest.raises(RuntimeError, match="Agent nesting depth exceeded"):
        await runner.run_delegated_task(
            request=AgentDelegationRequest(
                delegation_id="delegation-depth",
                agent_type=child.agent_type,
                prompt="Child task",
            ),
            parent_state={
                "run_config": _parent_config(
                    "parent-depth-run",
                    max_depth=0,
                )
            },
        )

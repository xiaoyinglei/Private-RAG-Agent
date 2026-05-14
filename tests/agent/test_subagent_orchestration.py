from __future__ import annotations

import pytest

from rag.agent.core.agent_service_factory import AgentServiceFactory
from rag.agent.core.context import AgentRunConfig, RuntimeRegistry
from rag.agent.core.definition import AgentDefinition
from rag.agent.core.registry import AgentRegistry
from rag.agent.core.subagent_runner import BuiltinSubAgentRunner
from rag.agent.core.task import SubTaskNode, TaskDAG
from rag.agent.service import AgentRunRequest
from rag.agent.state import AgentState, ThinkOutput, ToolCallPlan
from rag.agent.tools.builtin_registry import create_builtin_tool_registry
from rag.agent.tools.llm_tools import LLMTextOutput


class _ParentRouteProvider:
    def route(self, state: AgentState) -> dict[str, object]:
        if state["run_config"].parent_run_id is None:
            return {
                "status": "decompose",
                "execution_mode": "decompose",
                "route_reason": "integration_test",
            }
        return {"status": "direct", "execution_mode": "direct", "route_reason": "child"}


class _PlanProvider:
    def __init__(self, plan: TaskDAG) -> None:
        self.plan = plan
        self.calls = 0

    async def create_plan(self, state: AgentState, *, definition: AgentDefinition) -> TaskDAG:
        del state, definition
        self.calls += 1
        return self.plan


class _ChildDecisionProvider:
    def __init__(self) -> None:
        self.seen_configs: list[AgentRunConfig] = []

    async def decide(
        self,
        state: AgentState,
        *,
        definition: AgentDefinition,
        budget_remaining: int,
        context: object,
    ) -> ThinkOutput:
        del definition, budget_remaining, context
        self.seen_configs.append(state["run_config"])
        if not state.get("tool_results"):
            return ThinkOutput(
                action="execute",
                thought="summarize child",
                tool_calls=[ToolCallPlan.create("llm_summarize", {"task": state["task"]})],
            )
        return ThinkOutput(
            action="synthesize",
            thought="child complete",
            stop_reason="child_complete",
        )


@pytest.mark.anyio
async def test_task_dag_send_runs_real_builtin_subagent_runner_end_to_end() -> None:
    parent_definition = AgentDefinition(
        agent_type="orchestrator_test",
        description="Test orchestrator",
        system_prompt="Plan child tasks.",
        allowed_tools=[],
        estimated_token_budget=20000,
    )
    child_definition = AgentDefinition(
        agent_type="child_research_test",
        description="Test child",
        system_prompt="Summarize child task.",
        allowed_tools=["llm_summarize"],
        estimated_token_budget=12000,
    )
    subtask = SubTaskNode(
        subtask_id="s1",
        agent_type="child_research_test",
        prompt="Child research task",
        priority=1,
        estimated_tokens=10000,
    )
    agent_registry = AgentRegistry()
    agent_registry.register(child_definition)
    plan_provider = _PlanProvider(TaskDAG(subtasks=[subtask]))
    decision_provider = _ChildDecisionProvider()
    service_factory = AgentServiceFactory(
        tool_registry=create_builtin_tool_registry(
            runners={
                "llm_summarize": lambda payload: LLMTextOutput(
                    text=f"child-summary:{payload.task}",
                )
            }
        ),
        model_registry=None,
        route_provider=_ParentRouteProvider(),
        evaluate_decision_provider=decision_provider,
        plan_provider=plan_provider,
    )
    subagent_runner = BuiltinSubAgentRunner(
        agent_registry=agent_registry,
        service_factory=service_factory,
    )
    service_factory.bind_subagent_runner(subagent_runner)
    service = service_factory.create(parent_definition)

    result = await service.run(
        AgentRunRequest(
            task="Coordinate child task",
            run_id="subagent-orchestration",
            thread_id="subagent-orchestration",
        )
    )

    assert result.status == "done"
    assert result.final_answer == "child-summary:Child research task"
    assert plan_provider.calls == 1
    child_config = decision_provider.seen_configs[0]
    assert child_config.parent_run_id == "subagent-orchestration"
    assert child_config.max_depth == 1
    assert child_config.budget_total == 10000
    with pytest.raises(KeyError):
        RuntimeRegistry.get("subagent-orchestration")

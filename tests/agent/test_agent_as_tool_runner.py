from __future__ import annotations

import pytest

from rag.agent.core.agent_as_tool import AgentAsToolRunner
from rag.agent.core.context import AgentRunConfig, RuntimeRegistry
from rag.agent.core.definition import AgentDefinition
from rag.agent.core.registry import AgentRegistry
from rag.agent.core.task import SubTaskNode
from rag.agent.state import AgentState, ThinkOutput, ToolCallPlan
from rag.agent.tools.builtin_registry import create_builtin_tool_registry
from rag.agent.tools.llm_tools import LLMTextOutput
from rag.schema.query import RetrievalSignals
from rag.schema.runtime import AccessPolicy, ExecutionLocationPreference


class _ResearchUnderstandingService:
    def analyze(
        self,
        query: str,
        *,
        access_policy: object | None = None,
        execution_location_preference: object | None = None,
    ) -> RetrievalSignals:
        del query, access_policy, execution_location_preference
        return RetrievalSignals()


class _ChildDecisionProvider:
    def __init__(self) -> None:
        self.calls = 0
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


def _parent_state(run_id: str = "parent-run", *, max_depth: int = 2) -> AgentState:
    config = AgentRunConfig(
        run_id=run_id,
        thread_id=f"{run_id}-thread",
        budget_total=10000,
        max_depth=max_depth,
        parent_run_id=None,
        source_scope=("doc-1",),
        access_policy=AccessPolicy.default(),
        execution_location_preference=ExecutionLocationPreference.LOCAL_FIRST,
    )
    RuntimeRegistry.remove(run_id)
    RuntimeRegistry.get_or_create(config)
    return {
        "messages": [],
        "evidence": [],
        "citations": [],
        "tool_results": [],
        "task": "Parent task",
        "run_config": config,
        "plan": None,
        "iteration": 0,
        "status": "running",
        "route_reason": None,
        "stop_reason": None,
        "needs_user_input": None,
        "pending_tool_calls": [],
        "confirmed_tool_call_ids": set(),
        "user_decision": None,
        "next_subtasks": None,
        "working_summary": None,
        "extracted_facts": [],
        "context_budget": None,
        "subtask_results": {},
        "terminal_subtasks": set(),
        "successful_subtasks": set(),
        "final_answer": None,
        "groundedness_flag": False,
        "insufficient_evidence_flag": False,
    }


@pytest.mark.anyio
async def test_agent_as_tool_runner_executes_registered_child_with_derived_config() -> None:
    child_def = AgentDefinition(
        agent_type="child_research_runner",
        description="Child research",
        system_prompt="Research child task",
        allowed_tools=["llm_summarize"],
        estimated_token_budget=2500,
    )
    AgentRegistry.register(child_def)
    decision_provider = _ChildDecisionProvider()
    runner = AgentAsToolRunner(
        tool_registry=create_builtin_tool_registry(
            runners={
                "llm_summarize": lambda payload: LLMTextOutput(
                    text=f"summary:{payload.task}",
                    evidence_ids=payload.evidence_ids,
                    citation_ids=payload.citation_ids,
                )
            }
        ),

        evaluate_decision_provider=decision_provider,
    )

    result = await runner.run_subtask(
        subtask=SubTaskNode(
            subtask_id="s1",
            agent_type="child_research_runner",
            prompt="Child task",
            priority=1,
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
        RuntimeRegistry.get(result.run_id)


@pytest.mark.anyio
async def test_agent_as_tool_runner_rejects_exhausted_parent_depth() -> None:
    child_def = AgentDefinition(
        agent_type="child_depth_runner",
        description="Child depth",
        system_prompt="Depth",
        allowed_tools=[],
    )
    AgentRegistry.register(child_def)
    runner = AgentAsToolRunner(
        tool_registry=create_builtin_tool_registry(),

    )

    with pytest.raises(RuntimeError, match="Agent nesting depth exceeded"):
        await runner.run_subtask(
            subtask=SubTaskNode(
                subtask_id="s1",
                agent_type="child_depth_runner",
                prompt="Child task",
                priority=1,
            ),
            parent_state=_parent_state("parent-depth-run", max_depth=0),
        )

from __future__ import annotations

import pytest

from rag.agent.builtin.synthesize import SYNTHESIZE_AGENT
from rag.agent.builtin_registry import create_builtin_tool_registry
from rag.agent.core.agent_service_factory import AgentServiceFactory
from rag.agent.core.context import AgentRunConfig, RunRegistry
from rag.agent.core.definition import AgentDefinition
from rag.agent.core.delegation import AgentDelegationRequest
from rag.agent.core.observations import AnswerCandidate, EvidenceRef
from rag.agent.core.registry import AgentRegistry
from rag.agent.core.subagent_runner import BuiltinSubAgentRunner, BuiltinSynthesisRunner
from rag.agent.core.turn_contracts import ThinkOutput, ToolCallPlan
from rag.agent.loop.state import LoopState, create_loop_state
from rag.agent.service import AgentRunResult
from rag.agent.tools.llm_tools import LLMTextOutput
from rag.schema.query import AnswerCitation, EvidenceItem
from rag.schema.runtime import AccessPolicy


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
async def test_builtin_subagent_runner_returns_agent_run_result_with_derived_config() -> None:
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

    result = await runner.run_delegated_task(
        request=AgentDelegationRequest(
            delegation_id="s1",
            agent_type="child_research_runner",
            prompt="Child task",
            estimated_tokens=2400,
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
    assert first_child_config.budget_total == 2400
    with pytest.raises(KeyError):
        RunRegistry.get(result.run_id)


@pytest.mark.anyio
async def test_builtin_synthesis_runner_passes_bounded_grounding_to_child() -> None:
    class _SynthesisService:
        def __init__(self) -> None:
            self.task = ""
            self.config: AgentRunConfig | None = None
            self.pending_tool_calls: list[ToolCallPlan] = []

        async def run_with_config(
            self,
            *,
            task: str,
            run_config: AgentRunConfig,
            pending_tool_calls: list[ToolCallPlan],
        ) -> AgentRunResult:
            self.task = task
            self.config = run_config
            self.pending_tool_calls = pending_tool_calls
            return AgentRunResult(
                run_id=run_config.run_id,
                thread_id=run_config.thread_id,
                status="done",
                final_answer="grounded synthesis",
            )

    class _SynthesisFactory:
        def __init__(self, service: _SynthesisService) -> None:
            self.service = service

        def create(self, definition: AgentDefinition) -> _SynthesisService:
            assert definition == SYNTHESIZE_AGENT
            return self.service

    registry = AgentRegistry()
    registry.register(SYNTHESIZE_AGENT)
    service = _SynthesisService()
    runner = BuiltinSynthesisRunner(
        agent_registry=registry,
        service_factory=_SynthesisFactory(service),  # type: ignore[arg-type]
    )
    state = _parent_state(run_id="synthesis-parent")
    state["evidence"] = [
        EvidenceItem(
            evidence_id="ev-1",
            doc_id=7,
            citation_anchor="policy#3",
            text="E" * 2_000,
            score=0.9,
        )
    ]
    state["citations"] = [
        AnswerCitation(
            citation_id="cit-1",
            evidence_id="ev-1",
            record_type="section",
            citation_anchor="policy#3",
            doc_id=7,
        )
    ]
    state["answer_candidates"] = [
        AnswerCandidate(
            text="Candidate answer",
            evidence_refs=[
                EvidenceRef(
                    evidence_id="ev-1",
                    citation_id="cit-1",
                    citation_anchor="policy#3",
                )
            ],
        )
    ]
    state["evidence_refs"] = list(state["answer_candidates"][0].evidence_refs)

    result = await runner.run_synthesis(parent_state=state)

    assert result.final_answer == "grounded synthesis"
    assert service.config is not None
    assert service.config.parent_run_id == "synthesis-parent"
    [call] = service.pending_tool_calls
    assert call.tool_name == "llm_generate"
    assert call.arguments["evidence_ids"] == ["ev-1"]
    assert call.arguments["citation_ids"] == ["cit-1"]
    assert all(
        len(section) <= 1_600
        for section in call.arguments["context_sections"]
    )


@pytest.mark.anyio
async def test_builtin_subagent_runner_rejects_exhausted_parent_depth() -> None:
    child_def = AgentDefinition(
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

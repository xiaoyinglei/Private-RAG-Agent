from __future__ import annotations

import pytest
from pydantic import BaseModel

from rag.agent.core.context import AgentRunConfig, RuntimeRegistry
from rag.agent.core.definition import AgentDefinition, ToolPolicy
from rag.agent.core.task import SubTaskNode, SubTaskStatus, TaskDAG
from rag.agent.graphs.base import build_agent_graph
from rag.agent.memory.models import InjectedContext
from rag.agent.service import AgentRunResult
from rag.agent.state import AgentState, ThinkOutput, ToolCallPlan
from rag.agent.tools.registry import ToolRegistry
from rag.agent.tools.spec import ToolError, ToolPermissions, ToolSpec
from rag.schema.query import RetrievalSignals
from rag.schema.runtime import AccessPolicy, ExecutionLocationPreference


class EchoInput(BaseModel):
    message: str


class EchoOutput(BaseModel):
    message: str


_echo_spec = ToolSpec(
    name="echo",
    description="Echo back the message",
    input_model=EchoInput,
    output_model=EchoOutput,
    error_model=ToolError,
    permissions=ToolPermissions(),
    timeout_seconds=1.0,
)

_hidden_spec = ToolSpec(
    name="hidden",
    description="Hidden tool outside the agent definition",
    input_model=EchoInput,
    output_model=EchoOutput,
    error_model=ToolError,
    permissions=ToolPermissions(),
    timeout_seconds=1.0,
)

_confirmation_spec = ToolSpec(
    name="confirm_me",
    description="Tool requiring confirmation",
    input_model=EchoInput,
    output_model=EchoOutput,
    error_model=ToolError,
    permissions=ToolPermissions(write_db=True),
    timeout_seconds=1.0,
    requires_confirmation=True,
)

_costly_spec = ToolSpec(
    name="costly",
    description="Tool with non-zero budget cost",
    input_model=EchoInput,
    output_model=EchoOutput,
    error_model=ToolError,
    permissions=ToolPermissions(),
    timeout_seconds=1.0,
    token_budget_cost=50,
)


def _make_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(_echo_spec)
    registry.register(_hidden_spec)
    registry.register(_confirmation_spec)
    return registry


def _make_registry_with_costly_runner(calls: list[str]) -> ToolRegistry:
    registry = _make_registry()

    def runner(payload: EchoInput) -> EchoOutput:
        calls.append(payload.message)
        return EchoOutput(message=payload.message)

    registry.register(_costly_spec, runner=runner)
    return registry


def _make_registry_with_echo_runner() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(
        _echo_spec,
        runner=lambda payload: EchoOutput(message=f"echo:{payload.message}"),
    )
    registry.register(_hidden_spec)
    registry.register(_confirmation_spec)
    return registry


def _make_config(
    *,
    run_id: str = "graph-test",
    max_parallel_calls: int = 4,
    budget_total: int = 10000,
) -> AgentRunConfig:
    return AgentRunConfig(
        run_id=run_id,
        thread_id=run_id,
        budget_total=budget_total,
        max_depth=2,
        access_policy=AccessPolicy.default(),
        execution_location_preference=ExecutionLocationPreference.LOCAL_FIRST,
        tool_policy=ToolPolicy(max_parallel_calls=max_parallel_calls),
    )


def _make_definition(*, allowed_tools: list[str] | None = None, max_iterations: int = 3) -> AgentDefinition:
    return AgentDefinition(
        agent_type="echo_agent",
        description="Test echo agent",
        system_prompt="You have an echo tool.",
        allowed_tools=allowed_tools or ["echo"],
        max_iterations=max_iterations,
    )


def _initial_state(
    *,
    pending_tool_calls: list[ToolCallPlan] | None = None,
    config: AgentRunConfig | None = None,
) -> AgentState:
    state = {
        "messages": [],
        "evidence": [],
        "citations": [],
        "tool_results": [],
        "task": "research task requiring the agent loop",
        "run_config": config or _make_config(),
        "plan": None,
        "iteration": 0,
        "status": "running",
        "pending_tool_calls": pending_tool_calls or [],
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
    RuntimeRegistry.remove(state["run_config"].run_id)
    RuntimeRegistry.get_or_create(state["run_config"])
    return state


def _initial_state_without_runtime_handles(
    *,
    pending_tool_calls: list[ToolCallPlan] | None = None,
    config: AgentRunConfig | None = None,
) -> AgentState:
    state = _initial_state(pending_tool_calls=pending_tool_calls, config=config)
    RuntimeRegistry.remove(state["run_config"].run_id)
    return state


class _FakeUnderstandingService:
    def __init__(self, understanding: RetrievalSignals) -> None:
        self.understanding = understanding

    def analyze(
        self,
        query: str,
        *,
        access_policy: object | None = None,
        execution_location_preference: object | None = None,
    ) -> RetrievalSignals:
        del query, access_policy, execution_location_preference
        return self.understanding


def _research_service() -> _FakeUnderstandingService:
    return _FakeUnderstandingService(RetrievalSignals())


def _make_graph(
    *,
    definition: AgentDefinition | None = None,
    tool_registry: ToolRegistry | None = None,
    evaluate_decision_provider: object | None = None,
    plan_provider: object | None = None,
    subagent_runner: object | None = None,
):
    return build_agent_graph(
        definition=definition or _make_definition(),
        tool_registry=tool_registry or _make_registry(),
        evaluate_decision_provider=evaluate_decision_provider,
        plan_provider=plan_provider,
        subagent_runner=subagent_runner,
    )


class _ScriptedDecisionProvider:
    def __init__(self, decisions: list[ThinkOutput]) -> None:
        self._decisions = list(decisions)
        self.calls = 0

    async def decide(
        self,
        state: AgentState,
        *,
        definition: AgentDefinition,
        budget_remaining: int,
        context: InjectedContext,
    ) -> ThinkOutput:
        del state, definition, budget_remaining, context
        self.calls += 1
        return self._decisions.pop(0)


class _SuccessfulSubAgentRunner:
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def run_subtask(self, *, subtask: SubTaskNode, parent_state: AgentState) -> AgentRunResult:
        del parent_state
        self.calls.append(subtask.subtask_id)
        return AgentRunResult(
            run_id=f"child-{subtask.subtask_id}",
            thread_id=f"child-{subtask.subtask_id}",
            status="done",
            final_answer=f"done:{subtask.subtask_id}",
        )


class _ScriptedPlanProvider:
    def __init__(self, plan: TaskDAG) -> None:
        self._plan = plan
        self.calls = 0

    async def create_plan(self, state: AgentState, *, definition: AgentDefinition) -> TaskDAG:
        del state, definition
        self.calls += 1
        return self._plan


class TestBaseGraph:
    def test_builds_without_errors(self) -> None:
        graph = _make_graph()
        assert graph is not None

    @pytest.mark.anyio
    async def test_direct_route_without_tools_synthesizes_final_answer(self) -> None:
        graph = _make_graph()
        result = await graph.ainvoke(_initial_state(), config={"configurable": {"thread_id": "graph-test"}})
        assert result["status"] == "done"
        assert result["final_answer"] is not None

    @pytest.mark.anyio
    async def test_evaluate_decision_provider_can_schedule_tool_call(self) -> None:
        call = ToolCallPlan.create("echo", {"message": "from-think"})
        provider = _ScriptedDecisionProvider(
            [
                ThinkOutput(action="execute", tool_calls=[call], thought="need echo"),
                ThinkOutput(action="synthesize", thought="done", stop_reason="evidence_sufficient"),
            ]
        )
        graph = _make_graph(
            tool_registry=_make_registry_with_echo_runner(),
            evaluate_decision_provider=provider,
        )

        result = await graph.ainvoke(
            _initial_state(config=_make_config(run_id="think-execute")),
            config={"configurable": {"thread_id": "think-execute"}},
        )

        assert result["status"] == "done"
        assert result["stop_reason"] == "evidence_sufficient"
        assert provider.calls == 2
        [tool_result] = result["tool_results"]
        assert tool_result.status == "ok"
        assert tool_result.output == EchoOutput(message="echo:from-think")

    @pytest.mark.anyio
    async def test_evaluate_decision_provider_can_pause_run(self) -> None:
        provider = _ScriptedDecisionProvider(
            [
                ThinkOutput(
                    action="pause",
                    thought="need user choice",
                    needs_user_input="Choose a data source",
                )
            ]
        )
        graph = _make_graph(evaluate_decision_provider=provider)

        result = await graph.ainvoke(
            _initial_state(config=_make_config(run_id="think-pause")),
            config={"configurable": {"thread_id": "think-pause"}},
        )

        assert result["status"] == "paused"
        assert result["needs_user_input"] == "Choose a data source"
        assert result["final_answer"] is None

    @pytest.mark.anyio
    async def test_task_dag_send_executes_subagent_and_re_evaluates_to_done(self) -> None:
        subtask = SubTaskNode(
            subtask_id="s1",
            agent_type="research",
            prompt="Research",
            priority=1,
            estimated_tokens=10,
        )
        runner = _SuccessfulSubAgentRunner()
        graph = _make_graph(subagent_runner=runner)
        state = _initial_state(config=_make_config(run_id="graph-subagent", budget_total=100))
        state["plan"] = TaskDAG(subtasks=[subtask])

        result = await graph.ainvoke(
            state,
            config={"configurable": {"thread_id": "graph-subagent"}},
        )

        assert result["status"] == "done"
        assert result["stop_reason"] == "all_subtasks_terminal"
        assert result["final_answer"] == "done:s1"
        assert runner.calls == ["s1"]
        assert result["terminal_subtasks"] == {"s1"}
        assert result["successful_subtasks"] == {"s1"}
        assert result["subtask_results"]["s1"].findings == ["done:s1"]

    @pytest.mark.anyio
    async def test_task_dag_without_subagent_runner_records_failed_subtask(self) -> None:
        subtask = SubTaskNode(
            subtask_id="s1",
            agent_type="research",
            prompt="Research",
            priority=1,
            estimated_tokens=10,
        )
        graph = _make_graph()
        state = _initial_state(config=_make_config(run_id="graph-subagent-missing", budget_total=100))
        state["plan"] = TaskDAG(subtasks=[subtask])

        result = await graph.ainvoke(
            state,
            config={"configurable": {"thread_id": "graph-subagent-missing"}},
        )

        subtask_result = result["subtask_results"]["s1"]
        assert subtask_result.status is SubTaskStatus.FAILED
        assert subtask_result.error_message == "subagent_runner_missing"
        assert result["final_answer"] == "No answer was generated because subtask execution failed: s1."
        assert result["insufficient_evidence_flag"] is True
        assert result["terminal_subtasks"] == {"s1"}
        assert result["successful_subtasks"] == set()

    @pytest.mark.anyio
    async def test_decompose_route_uses_plan_provider_and_executes_subagent(self) -> None:
        subtask = SubTaskNode(
            subtask_id="s1",
            agent_type="research",
            prompt="Research",
            priority=1,
            estimated_tokens=10,
        )
        plan_provider = _ScriptedPlanProvider(TaskDAG(subtasks=[subtask]))
        runner = _SuccessfulSubAgentRunner()
        graph = _make_graph(


            plan_provider=plan_provider,
            subagent_runner=runner,
        )
        state = _initial_state(config=_make_config(run_id="graph-plan", budget_total=100))

        result = await graph.ainvoke(
            state,
            config={"configurable": {"thread_id": "graph-plan"}},
        )

        assert result["status"] == "done"
        assert result["stop_reason"] == "all_subtasks_terminal"
        assert result["final_answer"] == "done:s1"
        assert plan_provider.calls == 1
        assert runner.calls == ["s1"]
        assert result["plan"] == TaskDAG(subtasks=[subtask])

    @pytest.mark.anyio
    async def test_decompose_route_without_plan_provider_fails_closed(self) -> None:
        graph = _make_graph(


        )

        result = await graph.ainvoke(
            _initial_state(config=_make_config(run_id="graph-plan-missing", budget_total=100)),
            config={"configurable": {"thread_id": "graph-plan-missing"}},
        )

        assert result["status"] == "failed"
        assert result["stop_reason"] == "plan_provider_missing"
        assert result["final_answer"] == "Agent failed: plan_provider_missing."

    @pytest.mark.anyio
    async def test_registered_tool_without_runner_fails_closed(self) -> None:
        graph = _make_graph()
        call = ToolCallPlan.create("echo", {"message": "hello"})
        result = await graph.ainvoke(
            _initial_state(pending_tool_calls=[call]),
            config={"configurable": {"thread_id": "graph-test"}},
        )
        [tool_result] = result["tool_results"]
        assert tool_result.status == "error"
        assert tool_result.error.code == "tool_not_implemented"
        assert result["insufficient_evidence_flag"] is True

    @pytest.mark.anyio
    async def test_registered_tool_with_runner_records_ok_result(self) -> None:
        graph = _make_graph(tool_registry=_make_registry_with_echo_runner())
        call = ToolCallPlan.create("echo", {"message": "hello"})
        result = await graph.ainvoke(
            _initial_state(pending_tool_calls=[call]),
            config={"configurable": {"thread_id": "graph-test"}},
        )
        [tool_result] = result["tool_results"]
        assert tool_result.status == "ok"
        assert tool_result.output == EchoOutput(message="echo:hello")
        assert result["groundedness_flag"] is True

    @pytest.mark.anyio
    async def test_unregistered_tool_records_failure_result(self) -> None:
        graph = _make_graph()
        call = ToolCallPlan.create("missing_tool", {"message": "hello"})
        result = await graph.ainvoke(
            _initial_state(pending_tool_calls=[call]),
            config={"configurable": {"thread_id": "graph-test"}},
        )
        [tool_result] = result["tool_results"]
        assert tool_result.status == "error"
        assert tool_result.error.code == "tool_not_registered"

    @pytest.mark.anyio
    async def test_tool_outside_agent_definition_records_failure_result(self) -> None:
        graph = _make_graph(definition=_make_definition(allowed_tools=["echo"]))
        call = ToolCallPlan.create("hidden", {"message": "hello"})
        result = await graph.ainvoke(
            _initial_state(pending_tool_calls=[call]),
            config={"configurable": {"thread_id": "graph-test"}},
        )
        [tool_result] = result["tool_results"]
        assert tool_result.status == "error"
        assert tool_result.error.code == "tool_not_allowed"

    @pytest.mark.anyio
    async def test_tool_spec_confirmation_pauses_before_execution(self) -> None:
        graph = _make_graph(
            definition=_make_definition(allowed_tools=["confirm_me"]),
            tool_registry=_make_registry(),
        )
        call = ToolCallPlan.create("confirm_me", {"message": "hello"})
        result = await graph.ainvoke(
            _initial_state(pending_tool_calls=[call]),
            config={"configurable": {"thread_id": "graph-test"}},
        )
        assert result["status"] == "paused"
        assert result["needs_user_input"] == "Confirm tool execution: ['confirm_me']"
        assert result["tool_results"] == []
        assert result["pending_tool_calls"] == [call]

    @pytest.mark.anyio
    async def test_invalid_tool_arguments_record_failure_result(self) -> None:
        graph = _make_graph()
        call = ToolCallPlan.create("echo", {"unexpected": "value"})
        result = await graph.ainvoke(
            _initial_state(pending_tool_calls=[call]),
            config={"configurable": {"thread_id": "graph-test"}},
        )
        [tool_result] = result["tool_results"]
        assert tool_result.status == "error"
        assert tool_result.error.code == "invalid_arguments"

    @pytest.mark.anyio
    async def test_budget_exhausted_records_failure_without_calling_runner(self) -> None:
        runner_calls: list[str] = []
        graph = _make_graph(
            definition=_make_definition(allowed_tools=["costly"]),
            tool_registry=_make_registry_with_costly_runner(runner_calls),
        )
        call = ToolCallPlan.create("costly", {"message": "hello"})
        result = await graph.ainvoke(
            _initial_state(
                pending_tool_calls=[call],
                config=_make_config(
                    run_id="budget-too-low",
                    max_parallel_calls=1,
                    budget_total=10,
                ),
            ),
            config={"configurable": {"thread_id": "budget-too-low"}},
        )
        [tool_result] = result["tool_results"]
        assert tool_result.status == "error"
        assert tool_result.error.code == "budget_exhausted"
        assert runner_calls == []

    @pytest.mark.anyio
    async def test_successful_tool_commits_budget_cost(self) -> None:
        runner_calls: list[str] = []
        graph = _make_graph(
            definition=_make_definition(allowed_tools=["costly"]),
            tool_registry=_make_registry_with_costly_runner(runner_calls),
        )
        call = ToolCallPlan.create("costly", {"message": "hello"})
        state = _initial_state(
            pending_tool_calls=[call],
            config=_make_config(
                run_id="budget-commit",
                max_parallel_calls=1,
                budget_total=100,
            ),
        )

        result = await graph.ainvoke(state, config={"configurable": {"thread_id": "budget-commit"}})

        [tool_result] = result["tool_results"]
        assert tool_result.status == "ok"
        assert tool_result.token_used == 50
        assert runner_calls == ["hello"]
        remaining = await RuntimeRegistry.get("budget-commit").budget_ledger.remaining()
        assert remaining == 50

    @pytest.mark.anyio
    async def test_missing_runtime_handles_fail_visibly(self) -> None:
        graph = _make_graph()
        result = await graph.ainvoke(
            _initial_state_without_runtime_handles(),
            config={"configurable": {"thread_id": "graph-test"}},
        )
        assert result["status"] == "failed"
        assert result["stop_reason"] == "runtime_handles_missing"
        assert result["final_answer"] == "Agent failed: runtime_handles_missing."

    @pytest.mark.anyio
    async def test_max_parallel_calls_batches_pending_tools(self) -> None:
        graph = _make_graph(definition=_make_definition(max_iterations=3))
        calls = [
            ToolCallPlan.create("echo", {"message": "one"}),
            ToolCallPlan.create("echo", {"message": "two"}),
        ]
        result = await graph.ainvoke(
            _initial_state(pending_tool_calls=calls, config=_make_config(max_parallel_calls=1)),
            config={"configurable": {"thread_id": "graph-test"}},
        )
        assert len(result["tool_results"]) == 2
        assert result["pending_tool_calls"] == []
        assert result["iteration"] == 2

    @pytest.mark.anyio
    async def test_max_iterations_fails_closed_with_pending_tools(self) -> None:
        graph = _make_graph(definition=_make_definition(max_iterations=1))
        calls = [
            ToolCallPlan.create("echo", {"message": "one"}),
            ToolCallPlan.create("echo", {"message": "two"}),
        ]
        result = await graph.ainvoke(
            _initial_state(pending_tool_calls=calls, config=_make_config(max_parallel_calls=1)),
            config={"configurable": {"thread_id": "graph-test"}},
        )
        assert result["status"] == "failed"
        assert len(result["tool_results"]) == 1
        assert len(result["pending_tool_calls"]) == 1
        assert result["stop_reason"] == "max_iterations"

    @pytest.mark.anyio
    async def test_route_node_defaults_to_direct(self) -> None:
        graph = build_agent_graph(
            definition=_make_definition(),
            tool_registry=_make_registry(),

        )
        result = await graph.ainvoke(_initial_state(), config={"configurable": {"thread_id": "graph-test"}})
        assert result["route_reason"] == "agent_research"

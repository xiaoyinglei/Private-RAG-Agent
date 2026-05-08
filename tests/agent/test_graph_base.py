from __future__ import annotations

import pytest
from pydantic import BaseModel

from rag.agent.core.context import AgentRunConfig, RuntimeRegistry
from rag.agent.core.definition import AgentDefinition, ToolPolicy
from rag.agent.graphs.base import build_agent_graph
from rag.agent.state import AgentState, ToolCallPlan
from rag.agent.tools.registry import ToolRegistry
from rag.agent.tools.spec import ToolError, ToolPermissions, ToolSpec
from rag.schema.query import QueryUnderstanding, TaskType
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


def _make_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(_echo_spec)
    registry.register(_hidden_spec)
    registry.register(_confirmation_spec)
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


def _make_config(*, run_id: str = "graph-test", max_parallel_calls: int = 4) -> AgentRunConfig:
    return AgentRunConfig(
        run_id=run_id,
        thread_id=run_id,
        budget_total=10000,
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
    def __init__(self, understanding: QueryUnderstanding) -> None:
        self.understanding = understanding

    def analyze(
        self,
        query: str,
        *,
        access_policy: object | None = None,
        execution_location_preference: object | None = None,
    ) -> QueryUnderstanding:
        del query, access_policy, execution_location_preference
        return self.understanding


def _research_service() -> _FakeUnderstandingService:
    return _FakeUnderstandingService(QueryUnderstanding(task_type=TaskType.RESEARCH, query_type="research"))


def _make_graph(
    *,
    definition: AgentDefinition | None = None,
    tool_registry: ToolRegistry | None = None,
):
    return build_agent_graph(
        definition=definition or _make_definition(),
        tool_registry=tool_registry or _make_registry(),
        query_understanding_service=_research_service(),
    )


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
    async def test_missing_runtime_handles_fail_visibly(self) -> None:
        graph = _make_graph()
        result = await graph.ainvoke(
            _initial_state_without_runtime_handles(),
            config={"configurable": {"thread_id": "graph-test"}},
        )
        assert result["status"] == "failed"
        assert result["stop_reason"] == "runtime_handles_missing"

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
    async def test_route_uses_query_understanding_service(self) -> None:
        service = _FakeUnderstandingService(
            QueryUnderstanding(task_type=TaskType.COMPARISON, query_type="comparison")
        )
        graph = build_agent_graph(
            definition=_make_definition(),
            tool_registry=_make_registry(),
            query_understanding_service=service,
        )
        result = await graph.ainvoke(_initial_state(), config={"configurable": {"thread_id": "graph-test"}})
        assert result["route_reason"] == "multi_hop_or_compare"

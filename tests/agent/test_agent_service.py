from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from pydantic import BaseModel

from rag.agent.builtin.generic import GENERIC_AGENT
from rag.agent.builtin_registry import create_builtin_tool_registry
from rag.agent.core.agent_service_factory import AgentServiceFactory
from rag.agent.core.context import AgentRunConfig, RunRegistry
from rag.agent.core.definition import AgentRuntimePolicy
from rag.agent.core.goal_contract import GoalDeliverable, GoalSpec
from rag.agent.core.runtime_diagnostics import AgentLatencyProfile
from rag.agent.core.turn_contracts import ToolCallPlan
from rag.agent.loop.state import LoopState, ModelTurnDraft
from rag.agent.primitive_ops import PrimitiveOps, WriteFileOutput
from rag.agent.runner.python_runner import LocalSubprocessPythonRunner
from rag.agent.service import AgentRunRequest, AgentRunResult, AgentService
from rag.agent.tooling import ToolSurfaceRequest
from rag.agent.tools.llm_tools import LLMTextOutput
from rag.agent.workspace import WorkspaceRuntime
from rag.schema.llm import LLMProviderResult
from rag.schema.query import RetrievalSignals
from rag.schema.runtime import AccessPolicy


class _StructuredAnswer(BaseModel):
    answer: str
    confidence: float


class _TextGenerator:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    def generate_text(self, *, prompt: str, **kwargs: object) -> str:
        del kwargs
        self.prompts.append(prompt)
        return "model-backed summary"


class _NativeToolGenerator:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def generate_with_tools(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        **kwargs: object,
    ) -> object:
        from types import SimpleNamespace

        del kwargs
        self.calls.append({"messages": messages, "tools": tools})
        if len(self.calls) == 1:
            message = SimpleNamespace(
                content="",
                tool_calls=[
                    SimpleNamespace(
                        id="call_echo",
                        function=SimpleNamespace(
                            name="run_command",
                            arguments=(
                                '{"command": "echo hello", '
                                '"working_dir": ".", "timeout_seconds": 3}'
                            ),
                        ),
                    )
                ],
            )
            return LLMProviderResult(
                value=SimpleNamespace(
                    choices=[SimpleNamespace(finish_reason="tool_calls", message=message)]
                )
            )
        return LLMProviderResult(
            value=SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        finish_reason="stop",
                        message=SimpleNamespace(content="hello", tool_calls=[]),
                    )
                ]
            )
        )


class _NativeToolSequenceGenerator:
    def __init__(
        self,
        *,
        tool_name: str | None,
        arguments: dict[str, Any] | None = None,
        final_answer: str,
    ) -> None:
        self.tool_name = tool_name
        self.arguments = arguments or {}
        self.final_answer = final_answer
        self.calls: list[dict[str, Any]] = []

    def generate_with_tools(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        **kwargs: object,
    ) -> object:
        del kwargs
        self.calls.append({"messages": messages, "tools": tools})
        if len(self.calls) == 1 and self.tool_name is not None:
            message = SimpleNamespace(
                content="",
                tool_calls=[
                    SimpleNamespace(
                        id=f"call_{self.tool_name}",
                        function=SimpleNamespace(
                            name=self.tool_name,
                            arguments=json.dumps(self.arguments),
                        ),
                    )
                ],
            )
            return LLMProviderResult(
                value=SimpleNamespace(
                    choices=[SimpleNamespace(finish_reason="tool_calls", message=message)]
                )
            )
        return LLMProviderResult(
            value=SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        finish_reason="stop",
                        message=SimpleNamespace(
                            content=self.final_answer,
                            tool_calls=[],
                        ),
                    )
                ]
            )
        )


class _ResolvedFakeModel:
    def __init__(self, generator: _TextGenerator) -> None:
        self.generator = generator
        self.kwargs: dict[str, object] = {}


class _FakeModelRegistry:
    default_model = "fake"
    fallback_model = "fake"

    def __init__(self, generator: _TextGenerator) -> None:
        self.generator = generator

    def resolve_for_node(self, *, node_model: str | None, node_name: str) -> _ResolvedFakeModel:
        del node_model, node_name
        return _ResolvedFakeModel(self.generator)


class _FailingModelRegistry:
    default_model = "broken"
    fallback_model = None

    def resolve_for_node(self, *, node_model: str | None, node_name: str) -> object:
        del node_model, node_name
        raise RuntimeError("model provider broken")


class _ResearchUnderstandingService:
    def analyze(
        self,
        query: str,
        *,
        access_policy: object | None = None,
    ) -> RetrievalSignals:
        del query, access_policy
        return RetrievalSignals()


class _FinishFromResultsProvider:
    async def next_turn(
        self,
        state: LoopState,
        *,
        definition: AgentRuntimePolicy,
        budget_remaining: int,
    ) -> ModelTurnDraft:
        del definition, budget_remaining
        if state["tool_results"]:
            latest = state["tool_results"][-1]
            if latest.error is not None:
                return ModelTurnDraft(action="finish")
            # PR2: answer_candidates no longer written to LoopState; use tool output directly
            if latest.output is not None:
                text = getattr(latest.output, "text", None) or getattr(latest.output, "output_text", None)
                if text:
                    return ModelTurnDraft(action="finish", final_answer=text)
            summary = ", ".join(f"{result.tool_name}:{result.status}" for result in state["tool_results"])
            return ModelTurnDraft(
                action="finish",
                final_answer=f"Completed: {summary}",
            )
        return ModelTurnDraft(
            action="pause",
            pause_reason="No result is available.",
        )


def _service_with_registry(runners: dict[str, Any] | None = None) -> AgentService:
    extra = runners or {}
    extra.setdefault(
        "llm_summarize",
        lambda payload: LLMTextOutput(
            text=f"Summary: {payload.task}",
            evidence_ids=payload.evidence_ids,
            citation_ids=payload.citation_ids,
        ),
    )
    return AgentService(
        definition=GENERIC_AGENT,
        tool_registry=create_builtin_tool_registry(runners=extra),
        model_turn_provider=_FinishFromResultsProvider(),
    )


def test_agent_service_initial_state_creates_runtime_handles() -> None:
    service = _service_with_registry()
    request = AgentRunRequest(task="Explain policy", run_id="svc-state", thread_id="svc-state")

    state = service.initial_state(request)

    assert state["task"] == "Explain policy"
    assert state["run_config"].run_id == "svc-state"
    assert state["run_config"].llm_budget_total is None
    assert "tool_action_proposals" not in state
    assert "plan" not in state
    assert "subtask_results" not in state
    assert RunRegistry.get("svc-state") is not None
    RunRegistry.remove("svc-state")


def test_agent_initial_state_does_not_persist_explicit_goal_spec() -> None:
    service = _service_with_registry()
    goal = GoalSpec(
        original_query="Explain policy",
        deliverables=[
            GoalDeliverable(
                deliverable_id="answer",
                kind="answer",
                acceptance_rule="non_empty_answer",
            ),
            GoalDeliverable(
                deliverable_id="evidence",
                kind="evidence",
                acceptance_rule="traceable_evidence",
            ),
        ],
    )

    state = service.initial_state(
        AgentRunRequest(
            task="Explain policy",
            run_id="explicit-goal",
            thread_id="explicit-goal",
            goal_spec=goal,
        )
    )

    assert "goal_spec" not in state
    RunRegistry.remove("explicit-goal")


@pytest.mark.anyio
async def test_agent_service_defaults_to_strict_model_provider() -> None:
    service = AgentService(
        definition=GENERIC_AGENT,
        tool_registry=create_builtin_tool_registry(),
        model_registry=cast(Any, _FailingModelRegistry()),
    )

    try:
        with pytest.raises(RuntimeError, match="model provider broken"):
            await service.run(
                AgentRunRequest(
                    task="Explain policy",
                    run_id="svc-strict-model-default",
                    thread_id="svc-strict-model-default",
                )
            )
    finally:
        RunRegistry.remove("svc-strict-model-default")


@pytest.mark.anyio
async def test_agent_service_factory_defaults_to_strict_model_provider() -> None:
    factory = AgentServiceFactory(
        tool_registry=create_builtin_tool_registry(),
        model_registry=cast(Any, _FailingModelRegistry()),
    )
    service = factory.create(GENERIC_AGENT)

    try:
        with pytest.raises(RuntimeError, match="model provider broken"):
            await service.run(
                AgentRunRequest(
                    task="Explain policy",
                    run_id="svc-factory-strict-model-default",
                    thread_id="svc-factory-strict-model-default",
                )
            )
    finally:
        RunRegistry.remove("svc-factory-strict-model-default")


def test_agent_run_result_clears_stale_human_input_when_done() -> None:
    service = _service_with_registry()
    state = service.initial_state(AgentRunRequest(task="Explain policy", run_id="svc-clear", thread_id="svc-clear"))
    state["status"] = "completed"
    raw_state = cast(dict[str, Any], state)
    raw_state["needs_user_input"] = "stale approval"
    raw_state["human_input_request"] = object()

    result = AgentRunResult.from_state(state)

    assert result.status == "done"
    assert result.needs_user_input is None
    assert result.human_input_request is None
    RunRegistry.remove("svc-clear")


def test_agent_run_result_exposes_latency_profile_from_state() -> None:
    service = _service_with_registry()
    state = service.initial_state(AgentRunRequest(task="Explain policy", run_id="svc-profile", thread_id="svc-profile"))
    state["latency_profile"] = AgentLatencyProfile(
        startup_ms=1.0,
        build_service_ms=2.0,
        model_ready_ms=3.0,
        model_latency_ms=4.0,
        tool_latency_ms=5.0,
        finalize_latency_ms=6.0,
        total_ms=21.0,
        prompt_bytes=7,
        tool_schema_bytes=8,
    )

    result = AgentRunResult.from_state(state)

    assert result.latency_profile is not None
    assert result.latency_profile.total_ms == 21.0
    assert result.latency_profile.model_latency_ms == 4.0
    assert result.latency_profile.tool_schema_bytes == 8
    RunRegistry.remove("svc-profile")


@pytest.mark.anyio
async def test_agent_service_run_populates_latency_profile() -> None:
    service = _service_with_registry()

    result = await service.run(
        AgentRunRequest(
            task="Explain policy",
            run_id="svc-runtime-profile",
            thread_id="svc-runtime-profile",
        )
    )

    assert result.latency_profile is not None
    assert result.latency_profile.total_ms > 0
    assert result.latency_profile.model_latency_ms > 0
    assert result.latency_profile.tool_latency_ms == 0
    RunRegistry.remove("svc-runtime-profile")


def test_agent_run_result_restores_configured_concrete_final_output() -> None:
    definition = AgentRuntimePolicy.test_factory(
        agent_type="structured",
        description="Structured",
        system_prompt="Return structured output.",
        allowed_tools=[],
        output_model=_StructuredAnswer,
    )
    service = AgentService(
        definition=definition,
        tool_registry=create_builtin_tool_registry(),
    )
    state = service.initial_state(
        AgentRunRequest(
            task="Return a structured answer",
            run_id="svc-final-output",
            thread_id="svc-final-output",
        )
    )
    state["status"] = "completed"
    state["finish_state"].final_output = cast(Any, {
        "model_path": f"{_StructuredAnswer.__module__}.{_StructuredAnswer.__qualname__}",
        "data": {"answer": "validated", "confidence": 0.9},
    })

    result = AgentRunResult.from_state(state, definition=definition)

    assert result.final_output == _StructuredAnswer(
        answer="validated",
        confidence=0.9,
    )
    RunRegistry.remove("svc-final-output")


@pytest.mark.anyio
async def test_agent_service_run_executes_explicit_tool_call_with_runner() -> None:
    call = ToolCallPlan.create(
        "llm_summarize",
        {"task": "Explain policy", "evidence_ids": ["ev1"], "citation_ids": ["cit1"]},
    )
    service = _service_with_registry(
        runners={
            "llm_summarize": lambda payload: LLMTextOutput(
                text=f"summary:{payload.task}",
                evidence_ids=payload.evidence_ids,
                citation_ids=payload.citation_ids,
            )
        }
    )

    result = await service.run(
        AgentRunRequest(
            task="Explain policy",
            run_id="svc-ok",
            thread_id="svc-ok",
            pending_tool_calls=[call],
        )
    )

    assert result.status == "done"
    assert result.final_answer == "summary:Explain policy"
    assert result.tool_results[0].status == "ok"
    assert result.tool_results[0].output == LLMTextOutput(
        text="summary:Explain policy",
        evidence_ids=["ev1"],
        citation_ids=["cit1"],
    )
    with pytest.raises(KeyError):
        RunRegistry.get("svc-ok")


@pytest.mark.anyio
async def test_agent_service_run_without_runner_fails_closed() -> None:
    call = ToolCallPlan.create("llm_summarize", {"task": "Explain policy"})
    # Service without llm_summarize runner — should fail closed
    service = AgentService(
        definition=GENERIC_AGENT,
        tool_registry=create_builtin_tool_registry(runners={}),
        model_turn_provider=_FinishFromResultsProvider(),
    )

    result = await service.run(
        AgentRunRequest(
            task="Explain policy",
            run_id="svc-fail-closed",
            thread_id="svc-fail-closed",
            pending_tool_calls=[call],
        )
    )

    assert result.status == "failed"
    assert result.stop_reason == "tool_error"
    # No RAG generation output from this tool error, so insufficient_evidence_flag stays False
    assert result.insufficient_evidence_flag is False
    assert result.tool_results[0].status == "error"
    error = result.tool_results[0].error
    assert error is not None
    assert error.code == "tool_not_implemented"


@pytest.mark.anyio
async def test_agent_service_injects_model_backed_llm_tool_runners() -> None:
    generator = _TextGenerator()
    call = ToolCallPlan.create(
        "llm_summarize",
        {
            "task": "Explain policy",
            "context_sections": ["tool result context"],
            "evidence_ids": ["ev1"],
            "citation_ids": ["cit1"],
        },
    )
    service = AgentService(
        definition=GENERIC_AGENT,
        tool_registry=create_builtin_tool_registry(runners={}),
        model_turn_provider=_FinishFromResultsProvider(),
        model_registry=_FakeModelRegistry(generator),  # type: ignore[arg-type]
    )

    result = await service.run(
        AgentRunRequest(
            task="Explain policy",
            run_id="svc-model-llm-tools",
            thread_id="svc-model-llm-tools",
            pending_tool_calls=[call],
        )
    )

    assert result.status == "done"
    assert result.final_answer == "model-backed summary"
    assert result.tool_results[0].status == "ok"
    assert result.tool_results[0].output == LLMTextOutput(
        text="model-backed summary",
        evidence_ids=["ev1"],
        citation_ids=["cit1"],
    )
    assert "tool result context" in generator.prompts[0]


@pytest.mark.anyio
async def test_agent_service_explicit_tool_surface_uses_new_tooling_main_path() -> None:
    generator = _NativeToolGenerator()
    service = AgentService(
        definition=AgentRuntimePolicy.test_factory(
            agent_type="generic",
            description="Generic",
            system_prompt="Use only visible tools.",
            allowed_tools=[],
            max_iterations=3,
        ),
        tool_registry=create_builtin_tool_registry(runners={}),
        model_registry=_FakeModelRegistry(generator),  # type: ignore[arg-type]
    )

    result = await service.run(
        AgentRunRequest(
            task="运行 echo hello",
            run_id="svc-new-tooling",
            thread_id="svc-new-tooling",
            tool_surface_request=ToolSurfaceRequest(
                requested_tool_names=["run_command"],
                allow_execute_tools=True,
            ),
        )
    )

    assert result.status == "done"
    assert result.final_answer == "hello"
    assert [tool["function"]["name"] for tool in generator.calls[0]["tools"]] == [
        "run_command"
    ]
    assert result.tool_results[0].status == "ok"
    assert result.tool_results[0].output.data["stdout"].strip() == "hello"


@pytest.mark.anyio
async def test_agent_service_new_tooling_path_does_not_call_legacy_visibility_resolver(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from rag.agent.core import llm_providers

    def fail_legacy_visibility(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise AssertionError("new tooling service path must not call resolve_visible_tools")

    monkeypatch.setattr(llm_providers, "resolve_visible_tools", fail_legacy_visibility)
    generator = _NativeToolSequenceGenerator(
        tool_name="read_file",
        arguments={"path": "does-not-exist.txt"},
        final_answer="file_not_found",
    )
    service = AgentService(
        definition=AgentRuntimePolicy.test_factory(
            agent_type="generic",
            description="Generic",
            system_prompt="Use only visible tools.",
            allowed_tools=[],
            max_iterations=3,
        ),
        tool_registry=create_builtin_tool_registry(runners={}),
        model_registry=_FakeModelRegistry(generator),  # type: ignore[arg-type]
    )

    result = await service.run(
        AgentRunRequest(
            task="Read does-not-exist.txt and report the error code.",
            run_id="svc-new-tooling-no-legacy-visibility",
            thread_id="svc-new-tooling-no-legacy-visibility",
            tool_surface_request=ToolSurfaceRequest(
                requested_tool_names=["read_file"],
            ),
        )
    )

    assert result.status == "done"
    assert [tool["function"]["name"] for tool in generator.calls[0]["tools"]] == [
        "read_file"
    ]


@pytest.mark.anyio
async def test_agent_service_default_tool_surface_keeps_direct_qa_no_tools() -> None:
    generator = _NativeToolSequenceGenerator(
        tool_name=None,
        final_answer="4",
    )
    service = AgentService(
        definition=AgentRuntimePolicy.test_factory(
            agent_type="generic",
            description="Generic",
            system_prompt="Use only visible tools.",
            allowed_tools=[],
            max_iterations=3,
        ),
        tool_registry=create_builtin_tool_registry(runners={}),
        model_registry=_FakeModelRegistry(generator),  # type: ignore[arg-type]
    )

    result = await service.run(
        AgentRunRequest(
            task="2+2 等于几？直接回答数字。",
            run_id="svc-default-no-tools",
            thread_id="svc-default-no-tools",
        )
    )

    assert result.status == "done"
    assert result.final_answer == "4"
    assert generator.calls[0]["tools"] == []


@pytest.mark.anyio
@pytest.mark.parametrize(
    "task",
    [
        "Find AgentService in this repository.",
        "Read does-not-exist.txt and report the error code.",
        "Run echo hello and answer with stdout.",
    ],
)
async def test_agent_service_does_not_infer_tools_from_task_text(task: str) -> None:
    generator = _NativeToolSequenceGenerator(
        tool_name=None,
        final_answer="no tools",
    )
    service = AgentService(
        definition=AgentRuntimePolicy.test_factory(
            agent_type="generic",
            description="Generic",
            system_prompt="Use only visible tools.",
            allowed_tools=[],
            max_iterations=3,
        ),
        tool_registry=create_builtin_tool_registry(runners={}),
        model_registry=_FakeModelRegistry(generator),  # type: ignore[arg-type]
    )

    result = await service.run(
        AgentRunRequest(
            task=task,
            run_id=f"svc-no-infer-{abs(hash(task))}",
            thread_id=f"svc-no-infer-{abs(hash(task))}",
            workspace_path=str(Path(__file__).parents[2]),
        )
    )

    assert result.status == "done"
    assert result.final_answer == "no tools"
    assert generator.calls[0]["tools"] == []
    assert result.tool_results == []


@pytest.mark.anyio
async def test_agent_service_structured_tool_surface_finds_agent_service_via_workspace() -> None:
    generator = _NativeToolSequenceGenerator(
        tool_name="search_text",
        arguments={
            "pattern": "class AgentService",
            "path": "rag/agent/service.py",
            "max_results": 1,
        },
        final_answer="Found AgentService in rag/agent/service.py",
    )
    service = AgentService(
        definition=AgentRuntimePolicy.test_factory(
            agent_type="generic",
            description="Generic",
            system_prompt="Use only visible tools.",
            allowed_tools=[],
            max_iterations=3,
        ),
        tool_registry=create_builtin_tool_registry(runners={}),
        model_registry=_FakeModelRegistry(generator),  # type: ignore[arg-type]
    )

    result = await service.run(
        AgentRunRequest(
            task="Find AgentService in this repository.",
            run_id="svc-default-find-agent-service",
            thread_id="svc-default-find-agent-service",
            workspace_path=str(Path(__file__).parents[2]),
            tool_surface_request=ToolSurfaceRequest(
                requested_tool_names=["search_text", "list_files", "read_file"],
            ),
        )
    )

    assert result.status == "done"
    assert "AgentService" in (result.final_answer or "")
    assert [tool["function"]["name"] for tool in generator.calls[0]["tools"]] == [
        "search_text",
        "list_files",
        "read_file",
    ]
    assert result.tool_results[0].status == "ok"
    assert result.tool_results[0].output.data["total_matches"] >= 1


@pytest.mark.anyio
async def test_agent_service_structured_tool_surface_reads_missing_file_as_tool_error() -> None:
    generator = _NativeToolSequenceGenerator(
        tool_name="read_file",
        arguments={"path": "does-not-exist.txt"},
        final_answer="file_not_found",
    )
    service = AgentService(
        definition=AgentRuntimePolicy.test_factory(
            agent_type="generic",
            description="Generic",
            system_prompt="Use only visible tools.",
            allowed_tools=[],
            max_iterations=3,
        ),
        tool_registry=create_builtin_tool_registry(runners={}),
        model_registry=_FakeModelRegistry(generator),  # type: ignore[arg-type]
    )

    result = await service.run(
        AgentRunRequest(
            task="Read does-not-exist.txt and report the error code.",
            run_id="svc-default-missing-file",
            thread_id="svc-default-missing-file",
            tool_surface_request=ToolSurfaceRequest(
                requested_tool_names=["list_files", "read_file"],
            ),
        )
    )

    assert result.status == "done"
    assert result.final_answer == "file_not_found"
    assert [tool["function"]["name"] for tool in generator.calls[0]["tools"]] == [
        "list_files",
        "read_file",
    ]
    assert result.tool_results[0].status == "error"
    assert result.tool_results[0].error.code == "file_not_found"


@pytest.mark.anyio
async def test_agent_service_structured_tool_surface_runs_echo_command() -> None:
    generator = _NativeToolSequenceGenerator(
        tool_name="run_command",
        arguments={"command": "echo hello", "working_dir": ".", "timeout_seconds": 3},
        final_answer="hello",
    )
    service = AgentService(
        definition=AgentRuntimePolicy.test_factory(
            agent_type="generic",
            description="Generic",
            system_prompt="Use only visible tools.",
            allowed_tools=[],
            max_iterations=3,
        ),
        tool_registry=create_builtin_tool_registry(runners={}),
        model_registry=_FakeModelRegistry(generator),  # type: ignore[arg-type]
    )

    result = await service.run(
        AgentRunRequest(
            task="Run echo hello and answer with stdout.",
            run_id="svc-default-echo",
            thread_id="svc-default-echo",
            tool_surface_request=ToolSurfaceRequest(
                requested_tool_names=["run_command"],
                allow_execute_tools=True,
            ),
        )
    )

    assert result.status == "done"
    assert result.final_answer == "hello"
    assert [tool["function"]["name"] for tool in generator.calls[0]["tools"]] == [
        "run_command"
    ]
    assert result.tool_results[0].status == "ok"
    assert result.tool_results[0].output.data["stdout"].strip() == "hello"


@pytest.mark.anyio
async def test_agent_service_run_with_config_uses_supplied_runtime_contract() -> None:
    call = ToolCallPlan.create(
        "llm_summarize",
        {"task": "Explain policy", "evidence_ids": ["ev1"], "citation_ids": ["cit1"]},
    )
    config = AgentRunConfig(
        run_id="svc-child",
        thread_id="svc-child-thread",
        parent_run_id="svc-parent",
        source_scope=("doc-1",),
        llm_budget_total=5000,
        max_depth=1,
        access_policy=AccessPolicy.default(),
    )
    service = _service_with_registry(
        runners={
            "llm_summarize": lambda payload: LLMTextOutput(
                text=f"summary:{payload.task}",
                evidence_ids=payload.evidence_ids,
                citation_ids=payload.citation_ids,
            )
        }
    )

    result = await service.run_with_config(
        task="Explain policy",
        run_config=config,
        pending_tool_calls=[call],
    )

    assert result.run_id == "svc-child"
    assert result.thread_id == "svc-child-thread"
    assert result.status == "done"
    assert result.final_answer == "summary:Explain policy"
    with pytest.raises(KeyError):
        RunRegistry.get("svc-child")


@pytest.mark.anyio
async def test_agent_service_run_creates_workspace_and_injects_workspace_tools() -> None:
    """Verify AgentService.run() creates workspace and BaseTool instances are registered."""
    from rag.agent.builtin_registry import create_builtin_tool_registry
    from rag.agent.tools.workspace_tools import create_workspace_tools
    from rag.agent.workspace import create_temp_workspace

    workspace = create_temp_workspace(prefix="test_integ_")
    tools = create_workspace_tools(workspace)
    registry = create_builtin_tool_registry()
    for tool in tools:
        registry.register_tool(tool)

    # Verify workspace tools have runners
    assert registry.has_runner("list_files")
    assert registry.has_runner("read_file")
    assert registry.has_runner("write_file")
    assert registry.has_runner("run_python")

    # Verify list_files actually works through the registry
    result = await registry.run("list_files", {"path": ""})
    assert hasattr(result, "files")


@pytest.mark.anyio
async def test_agent_service_run_with_primitive_ops_through_agent_loop() -> None:
    """Verify write_file works through the full agent loop with workspace_path returned."""

    write_call = ToolCallPlan.create(
        "write_file",
        {"path": "scratch/hello.py", "content": "print('hello')"},
    )
    service = _service_with_registry()

    result = await service.run(
        AgentRunRequest(
            task="Write a Python script",
            run_id="prim-integ",
            thread_id="prim-integ",
            pending_tool_calls=[write_call],
            approved_tool_call_ids=[write_call.tool_call_id],
        )
    )

    assert result.status == "done", (
        f"status={result.status}, stop_reason={result.stop_reason}, needs_user_input={result.needs_user_input}"
    )
    assert result.workspace_path is not None
    write_result = result.tool_results[0]
    assert write_result.status == "ok"
    assert isinstance(write_result.output, WriteFileOutput)
    assert write_result.output.path == "scratch/hello.py"


@pytest.mark.anyio
async def test_agent_service_run_python_nonzero_exit_is_tool_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A script process failure must be visible as ToolResult.status='error'."""
    workspace = tmp_path / "workspace"
    for name in ("scratch", "artifacts", "reports", "logs", "input_files"):
        (workspace / name).mkdir(parents=True)
    (workspace / "scratch" / "fail.py").write_text("import sys\nsys.exit(7)\n")
    call = ToolCallPlan.create("run_python", {"script_path": "scratch/fail.py"})
    from rag.agent import primitive_ops as primitive_ops_module

    def _local_primitive_ops(*, workspace: WorkspaceRuntime) -> PrimitiveOps:
        return PrimitiveOps(
            workspace=workspace,
            python_runner=LocalSubprocessPythonRunner(),
        )

    monkeypatch.setattr(primitive_ops_module, "PrimitiveOps", _local_primitive_ops)
    service = AgentService(
        definition=AgentRuntimePolicy.test_factory(
            agent_type="python_test",
            description="Python test",
            system_prompt="Run Python.",
            allowed_tools=["run_python"],
            max_iterations=3,
        ),
        tool_registry=create_builtin_tool_registry(),
    )

    result = await service.run(
        AgentRunRequest(
            task="run failing script",
            run_id="python-nonzero-error",
            thread_id="python-nonzero-error",
            workspace_path=str(workspace),
            pending_tool_calls=[call],
            approved_tool_call_ids=[call.tool_call_id],
        )
    )

    [tool_result] = result.tool_results
    assert tool_result.status == "error"
    assert tool_result.error is not None
    assert tool_result.error.code == "tool_failed"
    assert tool_result.error.detail["exit_code"] == 7

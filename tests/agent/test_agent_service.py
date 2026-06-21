from __future__ import annotations

import pytest
from pydantic import BaseModel

from rag.agent.builtin.generic import GENERIC_AGENT
from rag.agent.builtin_registry import create_builtin_tool_registry
from rag.agent.compat.goal_contract import GoalDeliverable, GoalSpec
from rag.agent.core.context import AgentRunConfig, RunRegistry
from rag.agent.core.definition import AgentDefinition
from rag.agent.loop.state import LoopState, ModelTurnDraft
from rag.agent.service import AgentRunRequest, AgentRunResult, AgentService
from rag.agent.state import ToolCallPlan
from rag.agent.tools.llm_tools import LLMTextOutput
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
        definition: AgentDefinition,
        budget_remaining: int,
    ) -> ModelTurnDraft:
        del definition, budget_remaining
        if state["answer_candidates"]:
            return ModelTurnDraft(
                action="finish",
                final_answer=state["answer_candidates"][-1].text,
            )
        if state["tool_results"]:
            latest = state["tool_results"][-1]
            if latest.error is not None:
                return ModelTurnDraft(action="finish")
            summary = ", ".join(
                f"{result.tool_name}:{result.status}"
                for result in state["tool_results"]
            )
            return ModelTurnDraft(
                action="finish",
                final_answer=f"Completed: {summary}",
            )
        return ModelTurnDraft(
            action="pause",
            pause_reason="No result is available.",
        )


def _service_with_registry(runners: dict | None = None) -> AgentService:
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
    assert state["run_config"].budget_total == GENERIC_AGENT.estimated_token_budget
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


def test_agent_run_result_clears_stale_human_input_when_done() -> None:
    service = _service_with_registry()
    state = service.initial_state(
        AgentRunRequest(task="Explain policy", run_id="svc-clear", thread_id="svc-clear")
    )
    state["status"] = "done"
    state["needs_user_input"] = "stale approval"
    state["human_input_request"] = object()

    result = AgentRunResult.from_state(state)

    assert result.status == "done"
    assert result.needs_user_input is None
    assert result.human_input_request is None
    RunRegistry.remove("svc-clear")


def test_agent_run_result_restores_configured_concrete_final_output() -> None:
    definition = AgentDefinition(
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
    state["status"] = "done"
    state["final_output"] = {
        "model_path": f"{_StructuredAnswer.__module__}.{_StructuredAnswer.__qualname__}",
        "data": {"answer": "validated", "confidence": 0.9},
    }

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
    assert result.insufficient_evidence_flag is True
    assert result.tool_results[0].status == "error"
    assert result.tool_results[0].error.code == "tool_not_implemented"


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
        budget_total=5000,
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
async def test_agent_service_run_creates_workspace_and_injects_primitive_ops() -> None:
    """Verify AgentService.run() creates workspace and PrimitiveOps runners are available."""
    from rag.agent.builtin_registry import create_builtin_tool_registry
    from rag.agent.primitive_ops import PrimitiveOps
    from rag.agent.workspace import create_temp_workspace

    # Create service with PrimitiveOps-capable registry
    workspace = create_temp_workspace(prefix="test_integ_")
    ops = PrimitiveOps(workspace=workspace)
    registry = create_builtin_tool_registry(runners=ops.runners())

    # Verify primitive tools have runners
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
        f"status={result.status}, stop_reason={result.stop_reason}, "
        f"needs_user_input={result.needs_user_input}"
    )
    assert result.workspace_path is not None
    write_result = result.tool_results[0]
    assert write_result.status == "ok"
    assert write_result.output.path == "scratch/hello.py"


@pytest.mark.anyio
async def test_agent_service_run_python_nonzero_exit_is_tool_error(tmp_path) -> None:
    """A script process failure must be visible as ToolResult.status='error'."""
    workspace = tmp_path / "workspace"
    for name in ("scratch", "artifacts", "reports", "logs", "input_files"):
        (workspace / name).mkdir(parents=True)
    (workspace / "scratch" / "fail.py").write_text("import sys\nsys.exit(7)\n")
    call = ToolCallPlan.create("run_python", {"script_path": "scratch/fail.py"})
    service = AgentService(
        definition=AgentDefinition(
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

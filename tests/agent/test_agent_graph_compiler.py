from __future__ import annotations

import pytest
from pydantic import BaseModel

from rag.agent.core.compiler import GraphCompiler
from rag.agent.core.definition import AgentDefinition
from rag.agent.core.runtime_diagnostics import RuntimeDiagnostic
from rag.agent.loop.state import LoopState, ModelTurnDraft
from rag.agent.service import AgentRunRequest, AgentService
from rag.agent.state import AgentState, ThinkOutput
from rag.agent.tools.registry import ToolRegistry
from rag.agent.tools.spec import ToolError, ToolPermissions, ToolSpec


class SearchInput(BaseModel):
    query: str


class SearchOutput(BaseModel):
    items: list[str]


def _registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            name="vector_search",
            description="Vector search",
            input_model=SearchInput,
            output_model=SearchOutput,
            error_model=ToolError,
            permissions=ToolPermissions(read_db=True, embed=True),
            timeout_seconds=5.0,
        )
    )
    return registry


def _definition(*, allowed_tools: list[str]) -> AgentDefinition:
    return AgentDefinition(
        agent_type="research",
        description="Research agent",
        system_prompt="Use grounded evidence.",
        allowed_tools=allowed_tools,
    )


def test_compiler_builds_graph_for_registered_agent_tools() -> None:
    compiler = GraphCompiler(tool_registry=_registry())

    graph = compiler.compile(_definition(allowed_tools=["vector_search"]))

    assert hasattr(graph, "ainvoke")


def test_compiler_rejects_unregistered_agent_tools() -> None:
    compiler = GraphCompiler(tool_registry=_registry())

    with pytest.raises(ValueError, match="unregistered tools: missing_tool"):
        compiler.compile(_definition(allowed_tools=["vector_search", "missing_tool"]))


class _HintProvider:
    def hint(self, state: AgentState) -> dict[str, object]:
        del state
        return {"decision_reason": "test"}


class _ToolDecisionProvider:
    def decide(
        self,
        state: AgentState,
        *,
        definition: AgentDefinition,
        budget_remaining: int,
        context: object,
    ) -> ThinkOutput:
        del state, definition, budget_remaining, context
        return ThinkOutput(action="synthesize", thought="done")


class _RaisingHintProvider:
    def hint(self, state: AgentState) -> dict[str, object]:
        del state
        raise AssertionError("default LLM retrieval hint provider should not be called")


def test_compiler_constructs_only_model_driven_runtime_providers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def fake_create_default_providers(
        registry: object,
        selection: object,
        definition: AgentDefinition,
    ) -> tuple[_HintProvider, _ToolDecisionProvider]:
        nonlocal calls
        del registry, selection, definition
        calls += 1
        return _HintProvider(), _ToolDecisionProvider()

    monkeypatch.setattr(
        "rag.agent.core.compiler.create_default_providers",
        fake_create_default_providers,
    )
    compiler = GraphCompiler(
        tool_registry=_registry(),
        model_registry=object(),  # type: ignore[arg-type]
    )

    compiler.compile(_definition(allowed_tools=["vector_search"]))

    assert calls == 1


def test_compiler_constructs_output_finalizer_for_output_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    finalizer = object()
    captured: dict[str, object] = {}

    def fake_create_output_finalizer(registry: object) -> object:
        captured["registry"] = registry
        return finalizer

    def fake_build_agent_graph(**kwargs: object) -> object:
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(
        "rag.agent.core.compiler.create_model_structured_output_finalizer",
        fake_create_output_finalizer,
    )
    monkeypatch.setattr(
        "rag.agent.core.compiler.build_agent_graph",
        fake_build_agent_graph,
    )
    model_registry = object()
    compiler = GraphCompiler(
        tool_registry=_registry(),
        model_registry=model_registry,  # type: ignore[arg-type]
    )
    definition = AgentDefinition(
        agent_type="research",
        description="Research agent",
        system_prompt="Use grounded evidence.",
        allowed_tools=["vector_search"],
        output_model=SearchOutput,
    )

    compiler.compile(definition)

    assert captured["registry"] is model_registry
    assert captured["output_finalizer"] is finalizer


def test_compiler_preserves_explicit_output_finalizer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    finalizer = object()

    def fail_create_output_finalizer(registry: object) -> object:
        del registry
        raise AssertionError("explicit output finalizer must be preserved")

    def fake_build_agent_graph(**kwargs: object) -> object:
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(
        "rag.agent.core.compiler.create_model_structured_output_finalizer",
        fail_create_output_finalizer,
    )
    monkeypatch.setattr(
        "rag.agent.core.compiler.build_agent_graph",
        fake_build_agent_graph,
    )
    compiler = GraphCompiler(
        tool_registry=_registry(),
        output_finalizer=finalizer,  # type: ignore[arg-type]
        model_registry=object(),  # type: ignore[arg-type]
    )
    definition = AgentDefinition(
        agent_type="research",
        description="Research agent",
        system_prompt="Use grounded evidence.",
        allowed_tools=["vector_search"],
        output_model=SearchOutput,
    )

    compiler.compile(definition)

    assert captured["output_finalizer"] is finalizer


def test_compiler_records_optional_provider_initialization_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fail_default_providers(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise RuntimeError("decision model unavailable")

    def fail_output_finalizer(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise ValueError("structured output unavailable")

    def fail_goal_contract(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise LookupError("goal model unavailable")

    def fake_build_agent_graph(**kwargs: object) -> object:
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(
        "rag.agent.core.compiler.create_default_providers",
        fail_default_providers,
    )
    monkeypatch.setattr(
        "rag.agent.core.compiler.create_model_structured_output_finalizer",
        fail_output_finalizer,
    )
    monkeypatch.setattr(
        "rag.agent.core.compiler.create_goal_contract_provider",
        fail_goal_contract,
    )
    monkeypatch.setattr(
        "rag.agent.core.compiler.build_agent_graph",
        fake_build_agent_graph,
    )
    compiler = GraphCompiler(
        tool_registry=_registry(),
        model_registry=object(),  # type: ignore[arg-type]
    )
    definition = AgentDefinition(
        agent_type="research",
        description="Research agent",
        system_prompt="Use grounded evidence.",
        allowed_tools=["vector_search"],
        output_model=SearchOutput,
    )

    compiler.compile(definition)

    diagnostics = captured["runtime_diagnostics"]
    assert diagnostics == (
        RuntimeDiagnostic(
            code="default_providers_initialization_failed",
            component="model_providers",
            message="decision model unavailable",
            error_type="RuntimeError",
        ),
        RuntimeDiagnostic(
            code="structured_output_finalizer_initialization_failed",
            component="structured_output_finalizer",
            message="structured output unavailable",
            error_type="ValueError",
        ),
        RuntimeDiagnostic(
            code="goal_contract_provider_initialization_failed",
            component="goal_contract_provider",
            message="goal model unavailable",
            error_type="LookupError",
        ),
    )


@pytest.mark.anyio
async def test_loop_provider_degradation_reaches_agent_run_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_default_providers(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise RuntimeError("decision model unavailable")

    monkeypatch.setattr(
        "rag.agent.service.create_loop_model_turn_provider",
        fail_default_providers,
    )
    service = AgentService(
        definition=_definition(allowed_tools=["vector_search"]),
        tool_registry=_registry(),
        model_registry=object(),  # type: ignore[arg-type]
    )

    result = await service.run(
        AgentRunRequest(
            task="Explain policy",
            run_id="compiler-degradation-result",
            thread_id="compiler-degradation-result",
        )
    )

    assert result.status == "paused"
    assert [diagnostic.code for diagnostic in result.runtime_diagnostics] == [
        "default_providers_initialization_failed",
    ]
    assert result.runtime_diagnostics[0].message == "decision model unavailable"


@pytest.mark.anyio
async def test_model_can_finalize_without_explicit_goal_contract() -> None:
    class _DirectLoopProvider:
        async def next_turn(
            self,
            state: LoopState,
            *,
            definition: AgentDefinition,
            budget_remaining: int,
        ) -> ModelTurnDraft:
            del state, definition, budget_remaining
            return ModelTurnDraft(
                action="finish",
                final_answer="Direct answer.",
            )

    service = AgentService(
        definition=_definition(allowed_tools=["vector_search"]),
        tool_registry=_registry(),
        model_turn_provider=_DirectLoopProvider(),
    )

    result = await service.run(
        AgentRunRequest(
            task="Explain policy",
            run_id="default-goal-controller-disabled",
            thread_id="default-goal-controller-disabled",
        )
    )

    assert result.status == "done"
    assert result.stop_reason == "accepted"
    assert result.final_answer == "Direct answer."

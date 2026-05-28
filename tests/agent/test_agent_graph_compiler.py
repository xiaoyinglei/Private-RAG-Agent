from __future__ import annotations

import pytest
from pydantic import BaseModel

from rag.agent.core.compiler import AgentGraphCompiler
from rag.agent.core.definition import AgentDefinition
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
    compiler = AgentGraphCompiler(tool_registry=_registry())

    graph = compiler.compile(_definition(allowed_tools=["vector_search"]))

    assert hasattr(graph, "ainvoke")


def test_compiler_rejects_unregistered_agent_tools() -> None:
    compiler = AgentGraphCompiler(tool_registry=_registry())

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
    ) -> tuple[_HintProvider, _ToolDecisionProvider]:
        nonlocal calls
        del registry, selection
        calls += 1
        return _HintProvider(), _ToolDecisionProvider()

    monkeypatch.setattr(
        "rag.agent.core.compiler.create_default_providers",
        fake_create_default_providers,
    )
    compiler = AgentGraphCompiler(
        tool_registry=_registry(),
        model_registry=object(),  # type: ignore[arg-type]
    )

    compiler.compile(_definition(allowed_tools=["vector_search"]))

    assert calls == 1


@pytest.mark.anyio
async def test_model_cannot_finalize_while_required_goal_gaps_are_open(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_create_default_providers(
        registry: object,
        selection: object,
    ) -> tuple[_RaisingHintProvider, _ToolDecisionProvider]:
        del registry, selection
        return _RaisingHintProvider(), _ToolDecisionProvider()

    monkeypatch.setattr(
        "rag.agent.core.compiler.create_default_providers",
        fake_create_default_providers,
    )
    service = AgentService(
        definition=_definition(allowed_tools=["vector_search"]),
        tool_registry=_registry(),
        model_registry=object(),  # type: ignore[arg-type]
    )

    result = await service.run(
        AgentRunRequest(
            task="Explain policy",
            run_id="default-route-disabled",
            thread_id="default-route-disabled",
        )
    )

    assert result.status == "paused"
    assert result.stop_reason == "premature_synthesis"
    assert result.insufficient_evidence_flag is True

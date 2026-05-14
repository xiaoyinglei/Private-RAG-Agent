from __future__ import annotations

import pytest
from pydantic import BaseModel

from rag.agent.core.compiler import AgentGraphCompiler
from rag.agent.core.definition import AgentDefinition
from rag.agent.core.task import SubTaskNode
from rag.agent.graphs.nodes.execute_subagent import SubAgentRunResult
from rag.agent.state import AgentState
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


class _RouteProvider:
    def route(self, state: AgentState) -> dict[str, object]:
        del state
        return {"status": "direct", "execution_mode": "direct", "route_reason": "test"}


class _EvaluateProvider:
    pass


class _PlanProvider:
    pass


class _SubAgentRunner:
    async def run_subtask(
        self,
        *,
        subtask: SubTaskNode,
        parent_state: AgentState,
    ) -> SubAgentRunResult:
        del subtask, parent_state
        raise RuntimeError("not used")


def test_compiler_enables_decompose_when_subagent_runner_is_bound(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[bool] = []

    def fake_create_default_providers(
        registry: object,
        selection: object,
        *,
        decompose_enabled: bool = False,
    ) -> tuple[_RouteProvider, _EvaluateProvider, _PlanProvider]:
        del registry, selection
        seen.append(decompose_enabled)
        return _RouteProvider(), _EvaluateProvider(), _PlanProvider()

    monkeypatch.setattr(
        "rag.agent.core.compiler.create_default_providers",
        fake_create_default_providers,
    )
    compiler = AgentGraphCompiler(
        tool_registry=_registry(),
        model_registry=object(),  # type: ignore[arg-type]
        subagent_runner=_SubAgentRunner(),
    )

    compiler.compile(_definition(allowed_tools=["vector_search"]))

    assert seen == [True]


def test_compiler_keeps_decompose_disabled_without_subagent_runner(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[bool] = []

    def fake_create_default_providers(
        registry: object,
        selection: object,
        *,
        decompose_enabled: bool = False,
    ) -> tuple[_RouteProvider, _EvaluateProvider, _PlanProvider]:
        del registry, selection
        seen.append(decompose_enabled)
        return _RouteProvider(), _EvaluateProvider(), _PlanProvider()

    monkeypatch.setattr(
        "rag.agent.core.compiler.create_default_providers",
        fake_create_default_providers,
    )
    compiler = AgentGraphCompiler(
        tool_registry=_registry(),
        model_registry=object(),  # type: ignore[arg-type]
    )

    compiler.compile(_definition(allowed_tools=["vector_search"]))

    assert seen == [False]

from __future__ import annotations

from typing import NotRequired

import pytest
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from rag.agent.core.compiler import GraphCompiler
from rag.agent.core.definition import AgentRuntimePolicy
from rag.agent.loop.state import LoopState, ModelTurnDraft
from rag.agent.service import AgentRunRequest, AgentRunResult
from rag.agent.tools.registry import ToolRegistry
from rag.agent.tools.tool import (
    CancellationMode,
    InterruptBehavior,
    NormalizedToolOutput,
    ResolvedToolUse,
    Tool,
    ToolContentBlock,
    ToolDefinition,
    json_schema_input,
)


class _OuterWorkflowState(TypedDict):
    request: AgentRunRequest
    run_id: str
    result: NotRequired[AgentRunResult]
    audited: NotRequired[bool]


class _DirectProvider:
    async def next_turn(
        self,
        state: LoopState,
        *,
        definition: AgentRuntimePolicy,
        budget_remaining: int,
    ) -> ModelTurnDraft:
        del state, definition, budget_remaining
        return ModelTurnDraft(
            action="finish",
            final_answer="outer graph answer",
        )


class _PauseProvider:
    async def next_turn(
        self,
        state: LoopState,
        *,
        definition: AgentRuntimePolicy,
        budget_remaining: int,
    ) -> ModelTurnDraft:
        del state, definition, budget_remaining
        return ModelTurnDraft(
            action="pause",
            pause_reason="Need a user choice.",
        )


class _FailingProvider:
    async def next_turn(
        self,
        state: LoopState,
        *,
        definition: AgentRuntimePolicy,
        budget_remaining: int,
    ) -> ModelTurnDraft:
        del state, definition, budget_remaining
        raise RuntimeError("model unavailable")


def _registry() -> ToolRegistry:
    schema = {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
        "additionalProperties": False,
    }
    registry = ToolRegistry()
    registry.register(
        Tool(
            definition=ToolDefinition(
                name="vector_search",
                description="Vector search",
                input_schema=schema,
            ),
            validate_input=json_schema_input(schema),
            run=lambda arguments: {"items": [arguments["query"]]},
            normalize_output=lambda raw: NormalizedToolOutput(
                content=(
                    ToolContentBlock(type="text", data={"text": str(raw)}),
                ),
                structured_content=raw,  # type: ignore[arg-type]
            ),
            output_schema=None,
            static_effects=frozenset(),
            resolve_use=lambda _arguments: ResolvedToolUse(
                effects=frozenset(),
                targets=(),
            ),
            execution_revision="vector-search-v1",
            idempotent=True,
            concurrency_safe=True,
            cancellation_mode=CancellationMode.COOPERATIVE,
            interrupt_behavior=InterruptBehavior.CANCEL,
            timeout_seconds=5.0,
            max_model_output_bytes=4096,
        )
    )
    return registry


def _definition(*, allowed_tools: list[str]) -> AgentRuntimePolicy:
    return AgentRuntimePolicy.test_factory(
        agent_type="research",
        description="Research agent",
        system_prompt="Use grounded evidence.",
        allowed_tools=allowed_tools,
    )


def _request(run_id: str) -> AgentRunRequest:
    return AgentRunRequest(
        task="Explain policy",
        run_id=run_id,
        thread_id=run_id,
    )


def test_compiler_builds_only_a_coarse_agent_loop_node() -> None:
    graph = GraphCompiler(
        tool_registry=_registry(),
        model_turn_provider=_DirectProvider(),
    ).compile(_definition(allowed_tools=["vector_search"]))

    node_names = set(graph.get_graph().nodes)

    assert "agent_loop" in node_names
    assert {
        "initialize_goal",
        "controller",
        "execute",
        "llm_decide",
        "pause",
        "finalize",
    }.isdisjoint(node_names)


def test_compiler_rejects_unregistered_agent_tools() -> None:
    compiler = GraphCompiler(tool_registry=_registry())

    with pytest.raises(
        ValueError,
        match="unregistered tools: missing_tool",
    ):
        compiler.compile(
            _definition(
                allowed_tools=["vector_search", "missing_tool"]
            )
        )


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("provider", "expected_status"),
    [
        (_DirectProvider(), "done"),
        (_PauseProvider(), "paused"),
        (_FailingProvider(), "failed"),
    ],
)
async def test_outer_graph_receives_kernel_result(
    provider: object,
    expected_status: str,
) -> None:
    run_id = f"outer-{expected_status}"
    graph = GraphCompiler(
        tool_registry=_registry(),
        model_turn_provider=provider,  # type: ignore[arg-type]
    ).compile(_definition(allowed_tools=["vector_search"]))

    result = await graph.ainvoke(
        {
            "request": _request(run_id),
            "run_id": run_id,
        },
        config={"configurable": {"thread_id": f"workflow-{run_id}"}},
    )

    assert result["run_id"] == run_id
    assert result["result"].status == expected_status


@pytest.mark.anyio
async def test_compiled_kernel_node_participates_in_larger_langgraph() -> None:
    kernel_graph = GraphCompiler(
        tool_registry=_registry(),
        model_turn_provider=_DirectProvider(),
    ).compile(_definition(allowed_tools=["vector_search"]))
    workflow = StateGraph(_OuterWorkflowState)

    def mark_audited(state: _OuterWorkflowState) -> dict[str, bool]:
        assert state["result"].status == "done"
        return {"audited": True}

    workflow.add_node("kernel", kernel_graph)
    workflow.add_node("audit", mark_audited)
    workflow.add_edge(START, "kernel")
    workflow.add_edge("kernel", "audit")
    workflow.add_edge("audit", END)
    graph = workflow.compile()

    result = await graph.ainvoke(
        {
            "request": _request("outer-composed"),
            "run_id": "outer-composed",
        }
    )

    assert result["result"].final_answer == "outer graph answer"
    assert result["audited"] is True

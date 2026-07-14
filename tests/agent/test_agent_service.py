from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import pytest
from pydantic import BaseModel

from rag.agent.core.context import RunRegistry
from rag.agent.core.definition import AgentRuntimePolicy
from rag.agent.core.model_request import ModelCallRecord
from rag.agent.core.output_models import ValidatedFinalOutput
from rag.agent.core.turn_contracts import ToolCallPlan
from rag.agent.loop.runtime import ModelTurnEnvelope
from rag.agent.loop.state import LoopState, ModelTurnDraft
from rag.agent.service import AgentRunRequest, AgentRunResult, AgentService
from rag.agent.tools.registry import ToolRegistry
from rag.agent.tools.selection import ToolConfigurationError
from rag.agent.tools.tool import (
    CancellationMode,
    InterruptBehavior,
    JsonValue,
    NormalizedToolOutput,
    ResolvedToolUse,
    Tool,
    ToolContentBlock,
    ToolDefinition,
    ToolEffect,
    ToolTarget,
    json_schema_input,
)
from rag.schema.llm import LLMUsage


class _StructuredAnswer(BaseModel):
    answer: str
    confidence: float


class _FinishProvider:
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
            payload = latest.structured_content
            if isinstance(payload, Mapping):
                text = payload.get("text")
                if isinstance(text, str):
                    return ModelTurnDraft(
                        action="finish",
                        final_answer=text,
                    )
        return ModelTurnDraft(action="finish", final_answer="direct answer")


class _UsageProvider:
    async def next_turn(
        self,
        state: LoopState,
        *,
        definition: AgentRuntimePolicy,
        budget_remaining: int,
    ) -> ModelTurnEnvelope:
        del state, definition, budget_remaining
        return ModelTurnEnvelope(
            draft=ModelTurnDraft(action="finish", final_answer="metered"),
            model_call_record=ModelCallRecord(
                request_id="request-service",
                prompt_revision="prompt-service",
                toolset_revision="tools-service",
                provider_wire_hash="wire-service",
                usage=LLMUsage(
                    input_tokens=4,
                    output_tokens=2,
                    source="provider",
                    logical_input_tokens=4,
                    uncached_input_tokens=4,
                    usage_source="provider",
                ),
            ),
        )


class _ManifestProvider:
    def __init__(self) -> None:
        self.paths: tuple[str, ...] = ()

    async def next_turn(
        self,
        state: LoopState,
        *,
        definition: AgentRuntimePolicy,
        budget_remaining: int,
    ) -> ModelTurnDraft:
        del definition, budget_remaining
        manifest = state["file_manifest"]
        self.paths = (
            ()
            if manifest is None
            else tuple(entry.path for entry in manifest.files)
        )
        return ModelTurnDraft(action="finish", final_answer="inspected")


def _tool(
    name: str,
    calls: list[str] | None = None,
    *,
    write: bool = False,
) -> Tool:
    schema: Mapping[str, JsonValue] = {
        "type": "object",
        "properties": {"value": {"type": "string"}},
        "required": ["value"],
        "additionalProperties": False,
    }

    def run(arguments: Mapping[str, JsonValue]) -> object:
        value = str(arguments["value"])
        if calls is not None:
            calls.append(value)
        return {"text": f"ran:{value}"}

    def normalize(raw: object) -> NormalizedToolOutput:
        assert isinstance(raw, Mapping)
        text = str(raw["text"])
        return NormalizedToolOutput(
            content=(ToolContentBlock(type="text", data={"text": text}),),
            structured_content={"text": text},
        )

    effects = (
        frozenset({ToolEffect.WRITE_WORKSPACE})
        if write
        else frozenset()
    )
    return Tool(
        definition=ToolDefinition(
            name=name,
            description=f"Use {name}.",
            input_schema=schema,
        ),
        validate_input=json_schema_input(schema),
        run=run,
        normalize_output=normalize,
        output_schema=None,
        static_effects=effects,
        resolve_use=lambda _arguments: ResolvedToolUse(
            effects=effects,
            targets=(
                (ToolTarget(kind="workspace_path", value="."),)
                if write
                else ()
            ),
        ),
        execution_revision=f"{name}-v1",
        idempotent=not write,
        concurrency_safe=True,
        cancellation_mode=CancellationMode.COOPERATIVE,
        interrupt_behavior=InterruptBehavior.CANCEL,
        timeout_seconds=1.0,
        max_model_output_bytes=4096,
    )


def _registry(*tools: Tool) -> ToolRegistry:
    registry = ToolRegistry()
    for tool in tools:
        registry.register(tool)
    return registry


def _definition(
    names: tuple[str, ...],
    *,
    output_model: type[BaseModel] | None = None,
) -> AgentRuntimePolicy:
    return AgentRuntimePolicy.test_factory(
        agent_type="service_test",
        system_prompt="Use canonical tools.",
        allowed_tools=list(names),
        output_model=output_model,
        max_iterations=4,
    )


def test_initial_state_creates_runtime_handles_and_manifest() -> None:
    service = AgentService(
        definition=_definition(("read_file",)),
        tool_registry=_registry(_tool("read_file")),
        model_turn_provider=_FinishProvider(),
    )
    state = service.initial_state(
        AgentRunRequest(
            task="Inspect.",
            run_id="service-state",
            thread_id="service-state",
        )
    )

    assert RunRegistry.get("service-state") is not None
    assert state["resident_tool_names"] == ["read_file"]
    assert state["tool_manifest"] is not None
    assert state["tool_manifest"].resident_tool_names == ("read_file",)
    RunRegistry.remove("service-state")


@pytest.mark.anyio
async def test_service_executes_pending_call_through_final_executor() -> None:
    calls: list[str] = []
    service = AgentService(
        definition=_definition(("echo",)),
        tool_registry=_registry(_tool("echo", calls)),
        model_turn_provider=_FinishProvider(),
    )

    result = await service.run(
        AgentRunRequest(
            task="Echo.",
            run_id="service-tool",
            thread_id="service-tool",
            pending_tool_calls=[
                ToolCallPlan.create("echo", {"value": "once"})
            ],
        )
    )

    assert result.status == "done"
    assert result.final_answer == "ran:once"
    assert calls == ["once"]
    assert result.tool_results[0].is_error is False


@pytest.mark.anyio
async def test_service_pauses_write_until_approved() -> None:
    service = AgentService(
        definition=_definition(("write_tool",)),
        tool_registry=_registry(_tool("write_tool", write=True)),
        model_turn_provider=_FinishProvider(),
    )

    result = await service.run(
        AgentRunRequest(
            task="Write.",
            run_id="service-approval",
            thread_id="service-approval",
            pending_tool_calls=[
                ToolCallPlan.create("write_tool", {"value": "x"})
            ],
        )
    )

    assert result.status == "paused"
    assert result.human_input_request is not None
    assert result.human_input_request.kind == "tool_approval"


@pytest.mark.anyio
async def test_public_explicit_names_replace_defaults_and_disabled_wins() -> None:
    service = AgentService(
        definition=_definition(("first", "second")),
        tool_registry=_registry(_tool("first"), _tool("second")),
        model_turn_provider=_FinishProvider(),
    )
    state = service.initial_state(
        AgentRunRequest(
            task="Answer.",
            run_id="service-options",
            thread_id="service-options",
            tools=("second", "first"),
            disabled_tools=("first",),
        )
    )

    assert state["resident_tool_names"] == []
    assert state["explicit_tool_names"] == ["second"]
    assert state["disabled_tool_names"] == ["first"]
    RunRegistry.remove("service-options")


def test_unknown_public_tool_fails_before_model_call() -> None:
    service = AgentService(
        definition=_definition(("read_file",)),
        tool_registry=_registry(_tool("read_file")),
        model_turn_provider=_FinishProvider(),
    )

    with pytest.raises(ToolConfigurationError, match="unknown"):
        service.initial_state(
            AgentRunRequest(
                task="Answer.",
                tools=("missing",),
            )
        )


@pytest.mark.anyio
async def test_strict_model_initialization_failure_is_visible() -> None:
    class _BrokenRegistry:
        def resolve_for_node(self, **kwargs: object) -> object:
            del kwargs
            raise RuntimeError("model provider broken")

    service = AgentService(
        definition=_definition(("read_file",)),
        tool_registry=_registry(_tool("read_file")),
        model_registry=_BrokenRegistry(),  # type: ignore[arg-type]
    )

    with pytest.raises(RuntimeError, match="model provider broken"):
        await service.run(AgentRunRequest(task="Answer."))


@pytest.mark.anyio
async def test_service_projects_model_call_records_and_latency() -> None:
    service = AgentService(
        definition=_definition(("read_file",)),
        tool_registry=_registry(_tool("read_file")),
        model_turn_provider=_UsageProvider(),
    )

    result = await service.run(
        AgentRunRequest(
            task="Answer.",
            run_id="service-usage",
            thread_id="service-usage",
        )
    )

    assert result.model_call_records[0].request_id == "request-service"
    assert result.latency_profile is not None
    assert result.latency_profile.total_ms > 0


@pytest.mark.anyio
async def test_service_builds_manifest_for_imported_input_files(
    tmp_path: Path,
) -> None:
    source = tmp_path / "fixture.txt"
    source.write_text("before\n", encoding="utf-8")
    provider = _ManifestProvider()
    service = AgentService(
        definition=_definition(()),
        tool_registry=_registry(),
        model_turn_provider=provider,
    )

    result = await service.run(
        AgentRunRequest(
            task="Inspect the fixture.",
            run_id="service-input-manifest",
            thread_id="service-input-manifest",
            input_files=[str(source)],
            workspace_path=str(tmp_path / "workspace"),
        )
    )

    assert result.status == "done"
    assert provider.paths == ("input_files/fixture.txt",)


def test_result_restores_configured_concrete_final_output() -> None:
    definition = _definition((), output_model=_StructuredAnswer)
    service = AgentService(
        definition=definition,
        tool_registry=_registry(),
        model_turn_provider=_FinishProvider(),
    )
    state = service.initial_state(
        AgentRunRequest(
            task="Answer.",
            run_id="service-output",
            thread_id="service-output",
        )
    )
    state["finish_state"].final_output = ValidatedFinalOutput(
        model_path=(
            f"{_StructuredAnswer.__module__}."
            f"{_StructuredAnswer.__qualname__}"
        ),
        data={"answer": "grounded", "confidence": 0.9},
    )

    result = AgentRunResult.from_state(state, definition=definition)

    assert result.final_output == _StructuredAnswer(
        answer="grounded",
        confidence=0.9,
    )
    RunRegistry.remove("service-output")

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Literal

from pydantic import BaseModel, Field

from rag.agent.core.definition import AgentRuntimePolicy, ModelSelectionPolicy
from rag.agent.core.llm_registry import ModelResolver
from rag.agent.core.messages import ModelMessage, StopReason, ToolUseResult
from rag.agent.core.model_request import (
    ModelRequest,
    ModelSettings,
    bind_model_call_record,
    build_model_request,
    build_stable_context,
)
from rag.agent.core.turn_contracts import ToolCallPlan
from rag.agent.loop.runtime import ModelTurnEnvelope
from rag.agent.loop.state import LoopState, ModelTurnDraft
from rag.agent.streaming.events import text_delta
from rag.agent.tools.selection import select_tools
from rag.agent.tools.tool import Tool, ToolCallOrigin
from rag.providers.llm_gateway import LLMGateway
from rag.schema.llm import LLMCallStage


class LoopModelDecision(BaseModel):
    """Small compatibility input accepted by ``parse_loop_model_turn``."""

    action: Literal["execute", "finish", "pause"]
    tool_calls: list[ToolCallPlan] = Field(default_factory=list)
    final_answer: str | None = None
    pause_reason: str | None = None
    needs_user_input: str | None = None
    stop_reason: str | None = None
    thought: str | None = None


def parse_loop_model_turn(
    value: ModelTurnDraft | LoopModelDecision | Mapping[str, object],
) -> ModelTurnDraft:
    """Normalize a typed decision without giving labels routing authority."""

    if isinstance(value, ModelTurnDraft):
        return value
    decision = (
        value
        if isinstance(value, LoopModelDecision)
        else LoopModelDecision.model_validate(value)
    )
    calls = tuple(decision.tool_calls)
    if calls:
        return ModelTurnDraft(action="execute", tool_calls=calls)
    if decision.action == "finish":
        return ModelTurnDraft(action="finish", final_answer=decision.final_answer)
    if decision.action == "pause":
        return ModelTurnDraft(
            action="pause",
            pause_reason=(
                decision.pause_reason
                or decision.needs_user_input
                or decision.stop_reason
                or decision.thought
            ),
        )
    return ModelTurnDraft(action="execute")


class LLMLoopModelTurnProvider:
    """Build one canonical request and delegate only wire work to the gateway."""

    manages_llm_context = True

    def __init__(
        self,
        gateway: LLMGateway,
        *,
        model: str,
        provider: str,
        supports_native_tools: bool,
        registry_snapshot: Mapping[str, Tool],
        resident_tool_names: Sequence[str],
        disabled_tool_names: Sequence[str] = (),
        kwargs: Mapping[str, object] | None = None,
        stream_sink: object | None = None,
    ) -> None:
        if not hasattr(gateway, "agenerate_model_request"):
            raise TypeError("gateway must execute canonical ModelRequest values")
        if not isinstance(model, str) or not model:
            raise ValueError("model must be non-empty")
        if not isinstance(provider, str) or not provider:
            raise ValueError("provider must be non-empty")
        if type(supports_native_tools) is not bool:
            raise TypeError("supports_native_tools must be a bool")
        self._gateway = gateway
        self._model = model
        self._provider = provider
        self._supports_native_tools = supports_native_tools
        self._registry_snapshot = registry_snapshot
        self._resident_tool_names = tuple(resident_tool_names)
        self._disabled_tool_names = tuple(disabled_tool_names)
        self._kwargs = dict(kwargs or {})
        self._stream_sink = stream_sink

    async def next_turn(
        self,
        state: LoopState,
        *,
        definition: AgentRuntimePolicy,
        budget_remaining: int,
    ) -> ModelTurnEnvelope:
        del budget_remaining
        state_resident_names = (
            *state.get("resident_tool_names", ()),
            *state.get("explicit_tool_names", ()),
        )
        resident_names = tuple(
            state_resident_names or self._resident_tool_names
        )
        disabled_names = tuple(
            state.get("disabled_tool_names") or self._disabled_tool_names
        )
        selected_tools = select_tools(
            self._registry_snapshot,
            resident_names=resident_names,
            active_names=tuple(state.get("active_tool_names", ())),
            disabled_names=disabled_names,
        )
        context = build_stable_context(
            instructions=(definition.system_instructions or "You are a helpful agent.",),
            initial_user_task=state["task"],
            initial_memory=tuple(state.get("persistent_memories", ())),
            transcript=tuple(state.get("canonical_transcript", ())),
        )
        request = build_model_request(
            request_id=(
                f"{state['run_config'].run_id}:turn:{state['iteration']}"
            ),
            context=context,
            selected_tools=selected_tools,
            settings=self._model_settings(definition.model_selection),
        )
        response = await self._gateway.agenerate_model_request(
            stage=LLMCallStage.TOOL_DECISION,
            request=request,
            provider=self._provider,
            supports_native_tools=self._supports_native_tools,
            stream=self._stream_sink is not None,
            text_delta_sink=self._emit_text_delta,
        )
        turn = response.turn
        if not isinstance(turn, ToolUseResult):
            raise TypeError("gateway must return a provider-neutral ToolUseResult")
        record = bind_model_call_record(
            request=request,
            provider_wire_hash=response.provider_wire_hash,
            usage=response.usage,
        )
        assistant_message = ModelMessage(
            role="assistant",
            content=turn.text,
            tool_calls=tuple(turn.tool_calls),
        )
        return ModelTurnEnvelope(
            draft=_draft_from_turn(turn, request=request),
            request=request,
            model_call_record=record,
            assistant_message=assistant_message,
            context_revision=context.context_revision,
            provider_serializer_revision=response.serializer_revision,
        )

    def _model_settings(
        self,
        selection: ModelSelectionPolicy,
    ) -> ModelSettings:
        max_output_tokens = self._kwargs.get(
            "max_tokens",
            selection.tool_decision_max_tokens or 2048,
        )
        temperature = self._kwargs.get(
            "temperature",
            selection.tool_decision_temperature,
        )
        return ModelSettings(
            model=self._model,
            max_output_tokens=int(max_output_tokens),
            temperature=float(temperature),
            top_p=(
                None
                if self._kwargs.get("top_p", 1.0) is None
                else float(self._kwargs.get("top_p", 1.0))
            ),
            parallel_tool_calls=bool(
                self._kwargs.get("parallel_tool_calls", True)
            ),
            seed=(
                None
                if self._kwargs.get("seed") is None
                else int(self._kwargs["seed"])
            ),
        )

    async def _emit_text_delta(self, value: str) -> None:
        sink = self._stream_sink
        if sink is None:
            return
        emit = getattr(sink, "emit", None)
        if not callable(emit):
            return
        await emit(text_delta(value))


def _draft_from_turn(
    turn: ToolUseResult,
    *,
    request: ModelRequest,
) -> ModelTurnDraft:
    if turn.tool_calls:
        origin = ToolCallOrigin(
            request_id=request.request_id,
            toolset_revision=request.toolset_revision,
            exposed_tool_names=request.exposed_tool_names,
        )
        return ModelTurnDraft(
            action="execute",
            tool_calls=tuple(
                ToolCallPlan(
                    tool_call_id=call.id,
                    tool_name=call.name,
                    arguments=dict(call.input),
                    origin=origin,
                )
                for call in turn.tool_calls
            ),
        )
    if turn.stop_reason is StopReason.MAX_TOKENS:
        return ModelTurnDraft(
            action="pause",
            pause_reason="Model output reached its configured token limit.",
        )
    return ModelTurnDraft(
        action="finish",
        final_answer=turn.text or "The model returned an empty final response.",
    )


def create_loop_model_turn_provider(
    registry: ModelResolver,
    selection: ModelSelectionPolicy,
    *,
    registry_snapshot: Mapping[str, Tool],
    resident_tool_names: Sequence[str],
    disabled_tool_names: Sequence[str] = (),
    stream_sink: object | None = None,
) -> LLMLoopModelTurnProvider:
    resolved = registry.resolve_for_node(
        node_model=selection.tool_decision_model,
        node_name="tool_decision",
    )
    gateway = resolved.gateway
    if gateway is None:
        raise RuntimeError("resolved model does not provide an LLM gateway")
    provider = resolved.provider
    model = resolved.model
    supports_native_tools = resolved.supports_native_tools
    return LLMLoopModelTurnProvider(
        gateway,
        model=model,
        provider=provider,
        supports_native_tools=supports_native_tools,
        registry_snapshot=registry_snapshot,
        resident_tool_names=resident_tool_names,
        disabled_tool_names=disabled_tool_names,
        kwargs=resolved.kwargs,
        stream_sink=stream_sink,
    )


__all__ = [
    "LLMLoopModelTurnProvider",
    "LoopModelDecision",
    "create_loop_model_turn_provider",
    "parse_loop_model_turn",
]

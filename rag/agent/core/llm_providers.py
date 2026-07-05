from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from typing import Any, Literal

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from pydantic import BaseModel, Field

from rag.agent.capabilities.catalog import (
    DeferredToolStore,
    ToolCatalog,
    resolve_visible_tools,
)
from rag.agent.core.definition import AgentRuntimePolicy, ModelSelectionPolicy
from rag.agent.core.llm_context import (
    AgentLLMContextAssembler,
)
from rag.agent.core.llm_registry import ModelResolver
from rag.agent.core.messages import ModelMessage, StopReason, ToolCall, ToolUseResult
from rag.agent.core.runtime_diagnostics import AgentLatencyProfile
from rag.agent.core.tool_schema import AgentMessageAssembler, OpenAIAdapter
from rag.agent.core.turn_contracts import ToolCallPlan
from rag.agent.loop.state import (
    LoopState,
    ModelTurnDraft,
)
from rag.agent.memory.models import ExternalizedToolOutput
from rag.agent.tools.spec import ToolResult, ToolSpec
from rag.assembly.tokenizer import TokenAccountingService, TokenizerContract
from rag.providers.llm_gateway import LLMGateway
from rag.schema.llm import DEFAULT_LLM_STAGE_BUDGETS, LLMCallStage

# ── Tool decisions ──


class LoopModelDecision(BaseModel):
    """Structured model outcome for one loop turn."""

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
    """Normalize structured model output without giving labels routing authority."""

    if isinstance(value, ModelTurnDraft):
        return value
    if isinstance(value, LoopModelDecision):
        decision = value
    else:
        decision = LoopModelDecision.model_validate(value)

    calls = tuple(decision.tool_calls)
    if calls:
        return ModelTurnDraft(action="execute", tool_calls=calls)
    if decision.action == "finish":
        return ModelTurnDraft(
            action="finish",
            final_answer=decision.final_answer,
        )
    if decision.action == "pause":
        return ModelTurnDraft(
            action="pause",
            pause_reason=(
                decision.pause_reason or decision.needs_user_input or decision.stop_reason or decision.thought
            ),
        )
    return ModelTurnDraft(action="execute")


class LLMLoopModelTurnProvider:
    """Loop-specific provider returning a focused draft with no goal routing.

    When ``tool_specs`` is provided, uses the OpenAI-compatible native tool
    calling path (``OpenAIAdapter`` + ``AgentMessageAssembler``).  Otherwise
    falls back to the legacy ``AgentLLMContextAssembler`` path.
    """

    manages_llm_context = True

    def __init__(
        self,
        generator: Any,
        *,
        kwargs: dict[str, Any] | None = None,
        gateway: LLMGateway | None = None,
        context_assembler: AgentLLMContextAssembler | None = None,
        tool_specs: list[ToolSpec] | None = None,
        catalog: ToolCatalog | None = None,
        deferred_store: DeferredToolStore | None = None,
        stream_sink: Any = None,
        formatter_resolver: Any | None = None,
        skill_context_provider: Callable[[LoopState], str] | None = None,
    ) -> None:
        self._kwargs = kwargs or {}
        self._gateway = gateway or _fallback_gateway(
            generator,
            LLMCallStage.TOOL_DECISION,
        )
        _validate_shared_token_accounting(
            gateway=self._gateway,
            context_assembler=context_assembler,
        )
        self._context_assembler = context_assembler or _assembler_from_gateway(
            self._gateway,
            LLMCallStage.TOOL_DECISION,
            kwargs=self._kwargs,
        )
        self._tool_specs = tool_specs or []
        self._catalog = catalog
        self._deferred_store = deferred_store
        self._skill_context_provider = skill_context_provider
        self._assembler = AgentMessageAssembler(
            skill_context_provider=skill_context_provider,
        )
        self._formatter_resolver = formatter_resolver
        self._stream_sink = stream_sink

    async def next_turn(
        self,
        state: LoopState,
        *,
        definition: AgentRuntimePolicy,
        budget_remaining: int,
    ) -> ModelTurnDraft:
        if self._tool_specs:
            return await self._next_turn_with_tools(
                state,
                definition=definition,
            )
        return await self._next_turn_legacy(
            state,
            definition=definition,
            budget_remaining=budget_remaining,
        )

    async def _next_turn_with_tools(
        self,
        state: LoopState,
        *,
        definition: AgentRuntimePolicy,
    ) -> ModelTurnDraft:
        """OpenAI-compatible native tool calling path.

        When stream_sink is configured, emits TEXT_DELTA events as
        chunks arrive from the LLM.
        """
        # 1. Rebuild tool transcript from ledger + tool_results
        transcript = _rebuild_tool_transcript(state, formatter_resolver=self._formatter_resolver)

        # 2. Resolve visible tools BEFORE building system message
        visible_specs = self._filter_visible_tools(definition)
        visible_tool_names = [s.name for s in visible_specs]

        # 3. Build system message with actual visible tools
        system_msg = self._assembler.build_system_message(
            definition=definition,
            state=state,
            visible_tool_names=visible_tool_names,
        )

        # 4. Convert conversation history + ledger-derived tool transcript
        conversation_msgs = _base_messages_to_model_messages(state.get("messages", []))
        conversation_msgs.extend(transcript)
        all_messages = [system_msg, *conversation_msgs]
        openai_messages = OpenAIAdapter.messages(all_messages)
        openai_tools = OpenAIAdapter.tools(visible_specs)
        _record_llm_payload_size(
            state,
            prompt_bytes=_json_payload_bytes(openai_messages),
            tool_schema_bytes=_json_payload_bytes(openai_tools),
        )

        # 5. Call gateway (no budget — Claude Code uses max_turns as the only cap)
        if self._stream_sink is not None:
            result_value = await self._streaming_call(
                state=state,
                messages=openai_messages,
                tools=openai_tools,
            )
        else:
            result = await self._gateway.agenerate_with_tools(
                stage=LLMCallStage.TOOL_DECISION,
                messages=openai_messages,
                tools=openai_tools,
                kwargs=self._kwargs,
            )
            result_value = result.value

        # 6. Parse response
        tool_result = OpenAIAdapter.parse_tool_calls(result_value)

        # 7. Convert to ModelTurnDraft
        return _tool_use_result_to_draft(tool_result, state)

    async def _streaming_call(
        self,
        *,
        state: LoopState,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> Any:
        """流式调用 LLM，emit TEXT_DELTA 事件。

        从流式 chunks 中收集文本和工具调用，构造 OpenAI 格式的 result。
        不执行工具 — 工具执行由 AgentLoop 的主路径负责。
        """
        from rag.agent.streaming.events import text_delta as make_text_delta

        run_id = state["run_config"].run_id
        turn = state.get("iteration", 0)
        accumulated_text = ""

        tool_calls: list[dict[str, Any]] = []
        current_tool: dict[str, Any] | None = None
        current_tool_json = ""

        async for chunk in self._gateway.astream_with_tools(
            stage=LLMCallStage.TOOL_DECISION,
            messages=messages,
            tools=tools,
            kwargs=self._kwargs,
        ):
            if chunk.type == "text_delta" and chunk.content:
                accumulated_text += chunk.content
                await self._stream_sink.emit(make_text_delta(chunk.content, run_id=run_id, turn=turn))
            elif chunk.type == "tool_use_start":
                current_tool = {
                    "id": chunk.tool_id,
                    "type": "function",
                    "function": {"name": chunk.tool_name, "arguments": ""},
                }
                current_tool_json = ""
            elif chunk.type == "tool_input_delta":
                current_tool_json += chunk.content
            elif chunk.type == "content_block_stop":
                if current_tool is not None:
                    current_tool["function"]["arguments"] = current_tool_json
                    tool_calls.append(current_tool)
                    current_tool = None
                    current_tool_json = ""

        message: dict[str, Any] = {
            "role": "assistant",
            "content": accumulated_text,
        }
        if tool_calls:
            message["tool_calls"] = tool_calls
        return {"choices": [{"message": message}]}

    def _filter_visible_tools(
        self,
        definition: AgentRuntimePolicy,
    ) -> list[ToolSpec]:
        """Return only the tool specs currently visible to the model.

        Uses resolve_visible_tools to determine which tools are visible
        based on category (core/deferred) and activation state.
        """
        if self._catalog is None or self._deferred_store is None:
            return self._tool_specs
        visible_names = set(
            resolve_visible_tools(
                list(definition.allowed_tools),
                catalog=self._catalog,
                store=self._deferred_store,
            )
        )
        return [s for s in self._tool_specs if s.name in visible_names]

    async def _next_turn_legacy(
        self,
        state: LoopState,
        *,
        definition: AgentRuntimePolicy,
        budget_remaining: int,
    ) -> ModelTurnDraft:
        """Legacy structured-output path (no native tool calling)."""
        assembler = self._context_assembler
        if assembler is None:
            raise RuntimeError("loop model context assembler is not configured")
        if self._skill_context_provider is not None:
            from dataclasses import replace

            skill_context = self._skill_context_provider(state).strip()
            if skill_context:
                definition = replace(
                    definition,
                    system_instructions=(
                        f"{definition.system_instructions}\n\n{skill_context}"
                    ),
                )
        assembled = assembler.assemble_loop_turn(
            definition=definition,
            state=state,
            budget_remaining=budget_remaining,
            output_schema=LoopModelDecision,
        )
        _record_llm_payload_size(
            state,
            prompt_bytes=len(assembled.prompt.encode("utf-8")),
            tool_schema_bytes=0,
        )
        result = await self._gateway.agenerate_structured(
            stage=LLMCallStage.TOOL_DECISION,
            prompt=assembled.prompt,
            schema=LoopModelDecision,
            kwargs=self._kwargs,
        )
        return parse_loop_model_turn(result.value)


def _json_payload_bytes(value: object) -> int:
    return len(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )


def _record_llm_payload_size(
    state: LoopState,
    *,
    prompt_bytes: int,
    tool_schema_bytes: int,
) -> None:
    profile = state.get("latency_profile")
    if not isinstance(profile, AgentLatencyProfile):
        profile = AgentLatencyProfile()
    state["latency_profile"] = profile.model_copy(
        update={
            "prompt_bytes": profile.prompt_bytes + prompt_bytes,
            "tool_schema_bytes": profile.tool_schema_bytes + tool_schema_bytes,
        }
    )


def _base_messages_to_model_messages(
    messages: list[BaseMessage],
) -> list[ModelMessage]:
    """Convert langchain BaseMessage list to ModelMessage list."""
    result: list[ModelMessage] = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            result.append(ModelMessage(role="user", content=str(msg.content)))
        elif isinstance(msg, AIMessage):
            tool_calls: list[ToolCall] = []
            for tc in getattr(msg, "tool_calls", []) or []:
                tool_calls.append(
                    ToolCall(
                        id=tc.get("id", f"tc_{id(tc)}"),
                        name=tc.get("name", ""),
                        input=tc.get("args", {}),
                    )
                )
            result.append(
                ModelMessage(
                    role="assistant",
                    content=str(msg.content) if msg.content else "",
                    tool_calls=tuple(tool_calls),
                )
            )
        elif isinstance(msg, ToolMessage):
            result.append(
                ModelMessage(
                    role="tool",
                    content=str(msg.content),
                    tool_call_id=getattr(msg, "tool_call_id", None),
                )
            )
    return result


def _tool_use_result_to_draft(
    tool_result: ToolUseResult,
    state: LoopState,
) -> ModelTurnDraft:
    """Convert ToolUseResult to ModelTurnDraft for the loop kernel."""
    if tool_result.tool_calls:
        calls = tuple(
            ToolCallPlan(
                tool_call_id=tc.id,
                tool_name=tc.name,
                arguments=tc.input,
            )
            for tc in tool_result.tool_calls
        )
        return ModelTurnDraft(action="execute", tool_calls=calls)

    if tool_result.stop_reason == StopReason.TOOL_USE:
        return ModelTurnDraft(
            action="pause",
            pause_reason="Model requested tool use but no tool calls were parsed.",
        )

    text = tool_result.text.strip()
    if text:
        return ModelTurnDraft(action="finish", final_answer=text)

    return ModelTurnDraft(
        action="pause",
        pause_reason="Model produced no text or tool calls.",
    )


def _rebuild_tool_transcript(state: LoopState, formatter_resolver: Any = None) -> list[ModelMessage]:
    """Rebuild native tool-call transcript from ledger + tool_results."""
    transcript: list[ModelMessage] = []
    results_by_id = {result.tool_call_id: result for result in state["tool_results"]}
    for entry in state["tool_call_ledger"].entries:
        result = results_by_id.get(entry.plan.tool_call_id)
        transcript.append(
            ModelMessage(
                role="assistant",
                content="",
                tool_calls=(
                    ToolCall(
                        id=entry.plan.tool_call_id,
                        name=entry.plan.tool_name,
                        input=dict(entry.plan.arguments),
                    ),
                ),
            )
        )
        if result is not None:
            transcript.append(
                ModelMessage(
                    role="tool",
                    tool_call_id=result.tool_call_id,
                    content=_tool_result_content(result, formatter_resolver=formatter_resolver),
                )
            )
    return transcript


def _tool_result_content(result: ToolResult, formatter_resolver: Any = None) -> str:
    """Render tool result content for transcript, using PR2 formatter when available.

    Falls back through:
      1. PR2 formatter (when formatter_resolver is provided)
      2. PR2 format_tool_result_fallback
      3. model_dump_json (truncated)
    """

    if formatter_resolver is not None:
        formatter = formatter_resolver(result.tool_name)
        if formatter is not None:
            if isinstance(result.output, ExternalizedToolOutput):
                section = formatter.format_externalized(result.output)
            else:
                section = formatter.format_result(result)
            if section is not None:
                content = getattr(section, "content", None)
                if content:
                    return str(content)

    # Fallback: PR2 format_tool_result_fallback
    from rag.agent.tools.formatter import format_tool_result_fallback

    section = format_tool_result_fallback(result)
    if section is not None:
        content = getattr(section, "content", None)
        if content:
            return str(content)

    # Ultimate fallback
    if result.status == "ok":
        output = result.output
        if isinstance(output, ExternalizedToolOutput):
            return f"externalized_ref={output.ref.ref_id} status={output.status} summary={output.summary}"
        if output is not None:
            return output.model_dump_json(exclude_none=True)[:2000]
    if result.error is not None:
        return f"Error: {result.error.message}"
    return f"Tool {result.tool_name} returned no output."


def create_loop_model_turn_provider(
    registry: ModelResolver,
    selection: ModelSelectionPolicy,
    *,
    tool_registry: Any | None = None,
    definition: AgentRuntimePolicy | None = None,
    catalog: ToolCatalog | None = None,
    deferred_store: DeferredToolStore | None = None,
    stream_sink: Any = None,
    formatter_resolver: Any | None = None,
    skill_context_provider: Callable[[LoopState], str] | None = None,
) -> LLMLoopModelTurnProvider:
    resolved = registry.resolve_for_node(
        node_model=selection.tool_decision_model,
        node_name="tool_decision",
    )
    kwargs = dict(resolved.kwargs)
    kwargs.setdefault(
        "temperature",
        selection.tool_decision_temperature,
    )
    if selection.tool_decision_max_tokens is not None:
        kwargs["max_tokens"] = selection.tool_decision_max_tokens
    gateway = getattr(resolved, "gateway", None)

    # Resolve tool specs for native tool calling path
    tool_specs: list[ToolSpec] | None = None
    if tool_registry is not None and definition is not None:
        tool_specs = _resolve_tool_specs(tool_registry, definition.allowed_tools)

    return LLMLoopModelTurnProvider(
        resolved.generator,
        kwargs=kwargs,
        gateway=gateway,
        context_assembler=_assembler_from_gateway(
            gateway,
            LLMCallStage.TOOL_DECISION,
            kwargs=kwargs,
            formatter_resolver=formatter_resolver,
        ),
        tool_specs=tool_specs,
        catalog=catalog,
        deferred_store=deferred_store,
        stream_sink=stream_sink,
        formatter_resolver=formatter_resolver,
        skill_context_provider=skill_context_provider,
    )


def _resolve_tool_specs(
    tool_registry: Any,
    allowed_tools: list[str],
) -> list[ToolSpec]:
    """Resolve allowed tool names to ToolSpec objects."""
    specs: list[ToolSpec] = []
    for name in allowed_tools:
        try:
            spec = tool_registry.get(name)
            if isinstance(spec, ToolSpec):
                specs.append(spec)
        except (KeyError, Exception):
            continue
    return specs


def _assembler_from_gateway(
    gateway: LLMGateway | None,
    stage: LLMCallStage,
    *,
    kwargs: dict[str, Any] | None = None,
    formatter_resolver: Any | None = None,
) -> AgentLLMContextAssembler | None:
    if gateway is None:
        return None
    return AgentLLMContextAssembler(
        token_accounting=gateway.token_accounting,
        stage_budgets={stage: gateway.effective_stage_budget(stage, kwargs=kwargs)},
        formatter_resolver=formatter_resolver,
    )


def _fallback_gateway(
    generator: object,
    stage: LLMCallStage,
) -> LLMGateway:
    model_context_tokens = 32_768
    accounting = TokenAccountingService(
        TokenizerContract(
            embedding_model_name="agent-fallback",
            tokenizer_model_name="agent-fallback",
            chunking_tokenizer_model_name="agent-fallback",
            tokenizer_backend="simple",
            max_context_tokens=model_context_tokens,
            prompt_reserved_tokens=512,
            local_files_only=True,
        )
    )
    return LLMGateway(
        generator=generator,
        token_accounting=accounting,
        model_context_tokens=model_context_tokens,
        stage_budgets={stage: DEFAULT_LLM_STAGE_BUDGETS[stage]},
    )


def _validate_shared_token_accounting(
    *,
    gateway: LLMGateway,
    context_assembler: AgentLLMContextAssembler | None,
) -> None:
    if context_assembler is not None and context_assembler.token_accounting is not gateway.token_accounting:
        raise ValueError("AgentLLMContextAssembler and LLMGateway must share the same TokenAccountingService instance")

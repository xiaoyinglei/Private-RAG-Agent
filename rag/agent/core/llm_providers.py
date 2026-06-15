from __future__ import annotations

import re
from collections.abc import Awaitable, Mapping
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

from rag.agent.core.context import RunRegistry
from rag.agent.core.definition import AgentDefinition, ModelSelectionPolicy
from rag.agent.core.llm_context import (
    AgentLLMContextAssembler,
    AgentLLMContextOverflowError,
)
from rag.agent.core.llm_registry import ModelRegistry
from rag.agent.core.messages import ModelMessage, StopReason, ToolCall, ToolUseResult
from rag.agent.core.runtime_diagnostics import RuntimeDiagnostic
from rag.agent.core.runtime_ports import (
    RetrievalHintProvider,
    RetrievalHintUpdate,
)
from rag.agent.core.tool_schema import AgentMessageAssembler, OpenAIAdapter
from rag.agent.core.turn_contracts import ToolCallPlan
from rag.agent.loop.state import (
    LoopState,
    ModelTurnDraft,
    append_loop_diagnostic,
)
from rag.agent.tools.spec import ToolSpec
from rag.assembly.tokenizer import TokenAccountingService, TokenizerContract
from rag.providers.llm_gateway import LLMGateway
from rag.schema.llm import DEFAULT_LLM_STAGE_BUDGETS, LLMCallStage
from rag.schema.query import RetrievalSignals

_QUOTED_TERM_RE = re.compile(
    r"""["“”]([^"“”]+?)["“”]|"""
    r"""['‘’]([^'‘’]+?)['‘’]""",
)


def _extract_quoted_terms(text: str) -> list[str]:
    """从 query 中提取所有引号内的内容作为 quoted_terms。"""
    terms: list[str] = []
    seen: set[str] = set()
    for match in _QUOTED_TERM_RE.finditer(text):
        term = (match.group(1) or match.group(2)).strip()
        if term and term.lower() not in seen:
            terms.append(term)
            seen.add(term.lower())
    return terms


def _merge_quoted_terms(llm_terms: list[str], rule_terms: list[str]) -> list[str]:
    """融合规则提取和 LLM 输出的 quoted_terms。规则优先，去重且过滤空字符串。"""
    merged: list[str] = []
    seen: set[str] = set()
    for term in rule_terms + llm_terms:
        cleaned = term.strip()
        if cleaned and cleaned.lower() not in seen:
            merged.append(cleaned)
            seen.add(cleaned.lower())
    return merged


def _filter_non_empty(items: list[str]) -> list[str]:
    """过滤空字符串，保留有意义的 term。"""
    return [t for t in items if t.strip()]


def _validate_retrieval_signals(
    raw: dict[str, object] | None,
) -> tuple[RetrievalSignals, str]:
    """校验 LLM 输出为合法 RetrievalSignals。

    Returns:
        (signals, source) — source ∈ {"llm", "validation_failed", "rule_fallback"}
    """
    if raw is None or not isinstance(raw, dict):
        return RetrievalSignals(), "rule_fallback"

    validation_errors: list[str] = []
    allowed = {"special_targets", "quoted_terms", "allow_graph_expansion"}
    filtered: dict[str, object] = {
        k: v for k, v in raw.items() if k in allowed and v is not None
    }

    if "quoted_terms" in filtered:
        if not isinstance(filtered["quoted_terms"], list):
            validation_errors.append("quoted_terms is not a list")
            filtered["quoted_terms"] = []
    if "special_targets" in filtered:
        if not isinstance(filtered["special_targets"], list):
            validation_errors.append("special_targets is not a list")
            filtered["special_targets"] = []
    if "allow_graph_expansion" in filtered:
        if not isinstance(filtered["allow_graph_expansion"], bool):
            validation_errors.append("allow_graph_expansion is not a bool")
            filtered["allow_graph_expansion"] = False

    if validation_errors:
        return RetrievalSignals(), "validation_failed"

    try:
        signals = RetrievalSignals.model_validate(filtered)
        return signals, "llm"
    except Exception:
        return RetrievalSignals(), "validation_failed"


# ── Retrieval hints ──


class RetrievalHintDecision(BaseModel):
    reason: str
    retrieval_signals: dict[str, object] | None = None


class LLMRetrievalHintProvider(RetrievalHintProvider):
    """LLM-generated retrieval hints for the model-driven agent loop."""

    def __init__(
        self,
        generator: Any,
        *,
        kwargs: dict[str, Any] | None = None,
        gateway: LLMGateway | None = None,
        context_assembler: AgentLLMContextAssembler | None = None,
        definition: AgentDefinition | None = None,
    ) -> None:
        self._kwargs = kwargs or {}
        self._uses_async_gateway = gateway is not None
        self._gateway = gateway or _fallback_gateway(
            generator,
            LLMCallStage.RETRIEVAL_HINT,
        )
        _validate_shared_token_accounting(
            gateway=self._gateway,
            context_assembler=context_assembler,
        )
        self._context_assembler = context_assembler or _assembler_from_gateway(
            self._gateway,
            LLMCallStage.RETRIEVAL_HINT,
            kwargs=self._kwargs,
        )
        self._definition = definition or AgentDefinition(
            agent_type="retrieval_hint",
            description="Retrieval hint context",
            system_prompt="",
            allowed_tools=[],
        )

    def hint(
        self,
        state: LoopState,
    ) -> RetrievalHintUpdate | Awaitable[RetrievalHintUpdate]:
        if self._uses_async_gateway:
            return self._hint_with_gateway(state)
        assembler = self._context_assembler
        if assembler is None:
            raise RuntimeError("retrieval hint context assembler is not configured")
        try:
            assembled = assembler.assemble_retrieval_hint(
                definition=self._definition,
                state=state,
                output_schema=RetrievalHintDecision,
            )
            decision = self._gateway.generate_structured(
                stage=LLMCallStage.RETRIEVAL_HINT,
                prompt=assembled.prompt,
                schema=RetrievalHintDecision,
                kwargs=self._kwargs,
            ).value
        except AgentLLMContextOverflowError:
            raise
        except Exception:
            decision = None
        return self._build_hint_update(state, decision)

    async def _hint_with_gateway(
        self,
        state: LoopState,
    ) -> RetrievalHintUpdate:
        gateway = self._gateway
        if gateway is None:
            raise RuntimeError("retrieval hint gateway is not configured")
        run_id = state["run_config"].run_id
        try:
            ledger = RunRegistry.get(run_id).budget_ledger
        except KeyError:
            ledger = None
        try:
            assembler = self._context_assembler
            if assembler is None:
                raise RuntimeError("retrieval hint context assembler is not configured")
            assembled = assembler.assemble_retrieval_hint(
                definition=self._definition,
                state=state,
                output_schema=RetrievalHintDecision,
            )
            result = await gateway.agenerate_structured(
                stage=LLMCallStage.RETRIEVAL_HINT,
                prompt=assembled.prompt,
                schema=RetrievalHintDecision,
                ledger=ledger,
                lease_id=f"{run_id}:retrieval_hint:{uuid4().hex}",
                kwargs=self._kwargs,
            )
        except AgentLLMContextOverflowError:
            raise
        except Exception as exc:
            update = self._build_hint_update(state, None)
            update["retrieval_signals_debug"] = {
                **update["retrieval_signals_debug"],
                "signals_source": "llm_gateway_failed",
                "gateway_error": str(exc),
            }
            return update
        return self._build_hint_update(state, result.value)

    def _build_hint_update(
        self,
        state: LoopState,
        decision: RetrievalHintDecision | None,
    ) -> RetrievalHintUpdate:
        task = state.get("task", "")

        # 规则提取 quoted_terms（从原始 query），过滤空字符串
        rule_quoted_terms = _filter_non_empty(_extract_quoted_terms(task))

        # 校验 LLM 输出的 retrieval_signals，返回 (signals, source)
        signals, signals_source = _validate_retrieval_signals(
            decision.retrieval_signals if decision is not None else None
        )

        # 融合 quoted_terms：规则优先，过滤空字符串
        llm_quoted = _filter_non_empty(list(signals.quoted_terms))
        merged_quoted = _merge_quoted_terms(llm_quoted, rule_quoted_terms)

        # 用 model_copy 保留 metadata_filters / structure_constraints 等字段
        clean_special = _filter_non_empty(list(signals.special_targets))
        signals = signals.model_copy(update={
            "special_targets": clean_special,
            "quoted_terms": merged_quoted,
        })

        signals_debug: dict[str, object] = {
            "signals_source": signals_source,
            "special_targets": list(signals.special_targets),
            "quoted_terms": list(signals.quoted_terms),
            "allow_graph_expansion": signals.allow_graph_expansion,
            "has_metadata_filters": signals.metadata_filters.has_constraints(),
            "has_structure_constraints": signals.structure_constraints.has_constraints(),
            "rule_quoted_terms_count": len(rule_quoted_terms),
            "llm_quoted_terms_count": len(llm_quoted),
        }

        reason = decision.reason if decision is not None else "agent_research"

        update: RetrievalHintUpdate = {
            "decision_reason": reason,
            "retrieval_signals": signals,
            "retrieval_signals_debug": signals_debug,
        }

        return update


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
                decision.pause_reason
                or decision.needs_user_input
                or decision.stop_reason
                or decision.thought
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
        self._assembler = AgentMessageAssembler()

    async def next_turn(
        self,
        state: LoopState,
        *,
        definition: AgentDefinition,
        budget_remaining: int,
    ) -> ModelTurnDraft:
        if self._tool_specs:
            return await self._next_turn_with_tools(
                state,
                definition=definition,
                budget_remaining=budget_remaining,
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
        definition: AgentDefinition,
        budget_remaining: int,
    ) -> ModelTurnDraft:
        """OpenAI-compatible native tool calling path."""
        # 1. Flush completed tool call/result pairs into loop_messages
        #    so the model sees its own tool history on subsequent turns.
        _flush_tool_results_to_loop_messages(state)

        # 2. Build system message
        system_msg = self._assembler.build_system_message(
            definition=definition,
            state=state,
            budget_remaining=budget_remaining,
        )

        # 3. Convert conversation history + loop_messages
        conversation_msgs = _base_messages_to_model_messages(
            state.get("messages", [])
        )
        conversation_msgs.extend(state.get("loop_messages", []))
        all_messages = [system_msg, *conversation_msgs]

        # 4. Convert to OpenAI format
        openai_messages = OpenAIAdapter.messages(all_messages)
        openai_tools = OpenAIAdapter.tools(self._tool_specs)

        # 5. Call gateway
        run_id = state["run_config"].run_id
        try:
            ledger = RunRegistry.get(run_id).budget_ledger
        except KeyError:
            ledger = None

        result = await self._gateway.agenerate_with_tools(
            stage=LLMCallStage.TOOL_DECISION,
            messages=openai_messages,
            tools=openai_tools,
            ledger=ledger,
            lease_id=(
                f"{run_id}:loop_turn:{state.get('iteration', 0)}:"
                f"{uuid4().hex}"
            ),
            kwargs=self._kwargs,
        )

        # 6. Parse response
        tool_result = OpenAIAdapter.parse_tool_calls(result.value)

        # 7. Convert to ModelTurnDraft
        return _tool_use_result_to_draft(tool_result, state)

    async def _next_turn_legacy(
        self,
        state: LoopState,
        *,
        definition: AgentDefinition,
        budget_remaining: int,
    ) -> ModelTurnDraft:
        """Legacy structured-output path (no native tool calling)."""
        assembler = self._context_assembler
        if assembler is None:
            raise RuntimeError("loop model context assembler is not configured")
        assembled = assembler.assemble_loop_turn(
            definition=definition,
            state=state,
            budget_remaining=budget_remaining,
            output_schema=LoopModelDecision,
        )
        run_id = state["run_config"].run_id
        try:
            ledger = RunRegistry.get(run_id).budget_ledger
        except KeyError:
            ledger = None
        result = await self._gateway.agenerate_structured(
            stage=LLMCallStage.TOOL_DECISION,
            prompt=assembled.prompt,
            schema=LoopModelDecision,
            ledger=ledger,
            lease_id=(
                f"{run_id}:loop_turn:{state.get('iteration', 0)}:"
                f"{uuid4().hex}"
            ),
            kwargs=self._kwargs,
        )
        return parse_loop_model_turn(result.value)


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
                        id=tc.get("id", f"tc_{uuid4().hex[:12]}"),
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

    if state.get("answer_candidates"):
        return ModelTurnDraft(
            action="finish",
            final_answer=state["answer_candidates"][-1].text,
        )

    return ModelTurnDraft(
        action="pause",
        pause_reason="Model produced no text or tool calls.",
    )


def _flush_tool_results_to_loop_messages(state: LoopState) -> None:
    """Convert completed tool call/result pairs into loop_messages.

    Uses pending_tool_calls (ToolCallPlan) for original arguments and
    tool_results (ToolResult) for outputs.  Writes assistant(tool_calls)
    + tool(result) pairs into state["loop_messages"].  Deduplicates by
    tool_call_id so repeated calls are idempotent.
    """
    loop_messages: list[ModelMessage] = state.setdefault("loop_messages", [])
    existing_ids = {
        m.tool_call_id
        for m in loop_messages
        if m.role == "tool" and m.tool_call_id
    }

    # Index pending plans by id for argument lookup
    plan_by_id: dict[str, Any] = {}
    for plan in state.get("pending_tool_calls", []):
        tc_id = getattr(plan, "tool_call_id", None)
        if tc_id:
            plan_by_id[tc_id] = plan

    for tr in state.get("tool_results", []):
        tc_id = tr.tool_call_id
        if tc_id in existing_ids:
            continue

        plan = plan_by_id.get(tc_id)
        arguments = dict(plan.arguments) if plan is not None else {}

        # assistant message with tool_call
        loop_messages.append(
            ModelMessage(
                role="assistant",
                content="",
                tool_calls=(ToolCall(id=tc_id, name=tr.tool_name, input=arguments),),
            )
        )

        # tool result message
        if tr.status == "ok" and tr.output is not None:
            content = tr.output.model_dump_json(exclude_none=True)[:2000]
        elif tr.error is not None:
            content = f"Error: {tr.error.message}"
        else:
            content = f"Tool {tr.tool_name} returned no output."

        loop_messages.append(
            ModelMessage(role="tool", content=content, tool_call_id=tc_id)
        )


def create_loop_model_turn_provider(
    registry: ModelRegistry,
    selection: ModelSelectionPolicy,
    *,
    tool_registry: Any | None = None,
    definition: AgentDefinition | None = None,
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
        ),
        tool_specs=tool_specs,
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


# ── 从 ModelRegistry 批量创建 ──


def create_default_providers(
    registry: ModelRegistry,
    selection: ModelSelectionPolicy,
    definition: AgentDefinition | None = None,
) -> tuple[LLMRetrievalHintProvider, LLMLoopModelTurnProvider]:
    """Create retrieval-hint and model-turn providers for an agent loop."""

    def _resolve(
        node_model: str | None,
        node_name: str,
        temperature: float,
        max_tokens: int | None,
    ) -> tuple[Any, dict[str, Any]]:
        resolved = registry.resolve_for_node(node_model=node_model, node_name=node_name)
        kwargs = dict(resolved.kwargs)
        kwargs.setdefault("temperature", temperature)
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        return resolved.generator, kwargs

    hint_gen, hint_kwargs = _resolve(
        selection.retrieval_hint_model,
        "retrieval_hint",
        selection.retrieval_hint_temperature,
        selection.retrieval_hint_max_tokens,
    )
    decision_gen, decision_kwargs = _resolve(
        selection.tool_decision_model,
        "tool_decision",
        selection.tool_decision_temperature,
        selection.tool_decision_max_tokens,
    )
    hint_resolved = registry.resolve_for_node(
        node_model=selection.retrieval_hint_model,
        node_name="retrieval_hint",
    )
    decision_resolved = registry.resolve_for_node(
        node_model=selection.tool_decision_model,
        node_name="tool_decision",
    )
    return (
        LLMRetrievalHintProvider(
            hint_gen,
            kwargs=hint_kwargs,
            gateway=hint_resolved.gateway,
            context_assembler=_assembler_from_gateway(
                hint_resolved.gateway,
                LLMCallStage.RETRIEVAL_HINT,
                kwargs=hint_kwargs,
            ),
            definition=definition,
        ),
        LLMLoopModelTurnProvider(
            decision_gen,
            kwargs=decision_kwargs,
            gateway=decision_resolved.gateway,
            context_assembler=_assembler_from_gateway(
                decision_resolved.gateway,
                LLMCallStage.TOOL_DECISION,
                kwargs=decision_kwargs,
            ),
        ),
    )


def _assembler_from_gateway(
    gateway: LLMGateway | None,
    stage: LLMCallStage,
    *,
    kwargs: dict[str, Any] | None = None,
) -> AgentLLMContextAssembler | None:
    if gateway is None:
        return None
    return AgentLLMContextAssembler(
        token_accounting=gateway.token_accounting,
        stage_budgets={
            stage: gateway.effective_stage_budget(stage, kwargs=kwargs)
        },
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
    if (
        context_assembler is not None
        and context_assembler.token_accounting is not gateway.token_accounting
    ):
        raise ValueError(
            "AgentLLMContextAssembler and LLMGateway must share the same "
            "TokenAccountingService instance"
        )

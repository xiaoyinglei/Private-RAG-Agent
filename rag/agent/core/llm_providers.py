from __future__ import annotations

import re
from collections.abc import Awaitable, Mapping
from inspect import isawaitable
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field

from rag.agent.core.context import RunRegistry
from rag.agent.core.definition import AgentDefinition, ModelSelectionPolicy
from rag.agent.core.llm_context import (
    AgentLLMContextAssembler,
    AgentLLMContextOverflowError,
)
from rag.agent.core.llm_registry import ModelRegistry
from rag.agent.core.runtime_diagnostics import RuntimeDiagnostic
from rag.agent.core.runtime_ports import (
    RetrievalHintProvider,
    ToolDecisionProvider,
)
from rag.agent.core.turn_contracts import ThinkOutput, ToolCallPlan
from rag.agent.loop.state import (
    LoopState,
    ModelTurnDraft,
    append_loop_diagnostic,
)
from rag.agent.memory.injector import ContextBuilder
from rag.agent.memory.models import ContextBudgetSnapshot, InjectedContext
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
    ) -> dict[Any, Any] | Awaitable[dict[Any, Any]]:
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

    async def _hint_with_gateway(self, state: LoopState) -> dict[Any, Any]:
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
    ) -> dict[Any, Any]:
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

        update: dict[str, Any] = {
            "decision_reason": reason,
            "retrieval_signals": signals,
            "retrieval_signals_debug": signals_debug,
        }

        return update


# ── Tool decisions ──


class LoopModelDecision(BaseModel):
    """Provider payload accepting both the new and legacy decision vocabulary."""

    action: Literal["execute", "finish", "synthesize", "pause"]
    tool_calls: list[ToolCallPlan] = Field(default_factory=list)
    final_answer: str | None = None
    pause_reason: str | None = None
    needs_user_input: str | None = None
    stop_reason: str | None = None
    thought: str | None = None


def parse_loop_model_turn(
    value: ModelTurnDraft | LoopModelDecision | ThinkOutput | Mapping[str, object],
) -> ModelTurnDraft:
    """Normalize legacy model output without giving labels routing authority."""

    if isinstance(value, ModelTurnDraft):
        return value
    if isinstance(value, ThinkOutput):
        decision = LoopModelDecision(
            action=value.action,
            tool_calls=value.tool_calls,
            needs_user_input=value.needs_user_input,
            stop_reason=value.stop_reason,
            thought=value.thought,
        )
    elif isinstance(value, LoopModelDecision):
        decision = value
    else:
        decision = LoopModelDecision.model_validate(value)

    calls = tuple(decision.tool_calls)
    if calls:
        return ModelTurnDraft(action="execute", tool_calls=calls)
    if decision.action in {"finish", "synthesize"}:
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
    """Loop-specific provider returning a focused draft with no goal routing."""

    manages_llm_context = True

    def __init__(
        self,
        generator: Any,
        *,
        kwargs: dict[str, Any] | None = None,
        gateway: LLMGateway | None = None,
        context_assembler: AgentLLMContextAssembler | None = None,
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

    async def next_turn(
        self,
        state: LoopState,
        *,
        definition: AgentDefinition,
        budget_remaining: int,
    ) -> ModelTurnDraft:
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
        if result.value.action == "synthesize":
            append_loop_diagnostic(
                state,
                RuntimeDiagnostic(
                    code="legacy_synthesize_normalized",
                    component="loop_model_provider",
                    message=(
                        "Normalized legacy synthesize action to finish intent."
                    ),
                    degraded=False,
                ),
            )
        return parse_loop_model_turn(result.value)


class LegacyToolDecisionModelTurnProvider:
    """Adapt an explicit legacy decision provider at the service boundary."""

    def __init__(
        self,
        provider: ToolDecisionProvider,
        *,
        use_synthesis_builder: bool,
    ) -> None:
        self._provider = provider
        self._use_synthesis_builder = use_synthesis_builder

    async def next_turn(
        self,
        state: LoopState,
        *,
        definition: AgentDefinition,
        budget_remaining: int,
    ) -> ModelTurnDraft:
        if bool(getattr(self._provider, "manages_llm_context", False)):
            context = InjectedContext(
                sections=[],
                context_budget=ContextBudgetSnapshot(
                    max_context_tokens=0
                ),
            )
        else:
            context = ContextBuilder(
                max_context_tokens=(
                    state["run_config"].max_context_tokens
                    or DEFAULT_LLM_STAGE_BUDGETS[
                        LLMCallStage.TOOL_DECISION
                    ].max_input_tokens
                ),
            ).assemble_loop(
                definition=definition,
                state=state,
            )
            state["context_budget"] = context.context_budget
        raw = self._provider.decide(
            state,  # type: ignore[arg-type]
            definition=definition,
            budget_remaining=budget_remaining,
            context=context,
        )
        if isawaitable(raw):
            raw = await raw
        draft = parse_loop_model_turn(raw)
        if (
            draft.action == "finish"
            and not draft.final_answer
            and not self._use_synthesis_builder
            and state["answer_candidates"]
        ):
            return draft.model_copy(
                update={
                    "final_answer": state["answer_candidates"][-1].text
                }
            )
        return draft


def create_loop_model_turn_provider(
    registry: ModelRegistry,
    selection: ModelSelectionPolicy,
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
    return LLMLoopModelTurnProvider(
        resolved.generator,
        kwargs=kwargs,
        gateway=gateway,
        context_assembler=_assembler_from_gateway(
            gateway,
            LLMCallStage.TOOL_DECISION,
            kwargs=kwargs,
        ),
    )


class LLMToolDecisionProvider(ToolDecisionProvider):
    """Model tool-choice decision provider. Invalid output pauses visibly."""

    manages_llm_context = True

    def __init__(
        self,
        generator: Any,
        *,
        kwargs: dict[str, Any] | None = None,
        gateway: LLMGateway | None = None,
        context_assembler: AgentLLMContextAssembler | None = None,
    ) -> None:
        self._kwargs = kwargs or {}
        self._uses_async_gateway = gateway is not None
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

    def decide(
        self,
        state: LoopState,
        *,
        definition: AgentDefinition,
        budget_remaining: int,
        context: InjectedContext,
    ) -> ThinkOutput | dict[str, object] | Awaitable[ThinkOutput | dict[str, object]]:
        del context
        if self._uses_async_gateway:
            return self._decide_with_gateway(
                state,
                definition=definition,
                budget_remaining=budget_remaining,
            )
        assembler = self._context_assembler
        if assembler is None:
            raise RuntimeError("tool decision context assembler is not configured")
        try:
            assembled = assembler.assemble_tool_decision(
                definition=definition,
                state=state,
                budget_remaining=budget_remaining,
                output_schema=ThinkOutput,
            )
            decision = self._gateway.generate_structured(
                stage=LLMCallStage.TOOL_DECISION,
                prompt=assembled.prompt,
                schema=ThinkOutput,
                kwargs=self._kwargs,
            ).value
        except AgentLLMContextOverflowError:
            raise
        except Exception:
            decision = None
        if decision is None:
            return ThinkOutput(
                action="pause",
                thought="LLM tool-decision response could not be parsed",
                needs_user_input="Tool-decision provider failed to produce a valid decision",
                confidence=0.0,
            )
        return decision

    async def _decide_with_gateway(
        self,
        state: LoopState,
        *,
        definition: AgentDefinition,
        budget_remaining: int,
    ) -> ThinkOutput:
        gateway = self._gateway
        if gateway is None:
            raise RuntimeError("tool decision gateway is not configured")
        run_id = state["run_config"].run_id
        try:
            ledger = RunRegistry.get(run_id).budget_ledger
        except KeyError:
            ledger = None
        try:
            assembler = self._context_assembler
            if assembler is None:
                raise RuntimeError("tool decision context assembler is not configured")
            assembled = assembler.assemble_tool_decision(
                definition=definition,
                state=state,
                budget_remaining=budget_remaining,
                output_schema=ThinkOutput,
            )
            result = await gateway.agenerate_structured(
                stage=LLMCallStage.TOOL_DECISION,
                prompt=assembled.prompt,
                schema=ThinkOutput,
                ledger=ledger,
                lease_id=(
                    f"{run_id}:tool_decision:{state.get('iteration', 0)}:"
                    f"{uuid4().hex}"
                ),
                kwargs=self._kwargs,
            )
        except AgentLLMContextOverflowError:
            raise
        except Exception as exc:
            return ThinkOutput(
                action="pause",
                thought="LLM gateway rejected or failed the tool decision call",
                needs_user_input=f"Tool-decision LLM call failed: {exc}",
                confidence=0.0,
            )
        return result.value


# ── 从 ModelRegistry 批量创建 ──


def create_default_providers(
    registry: ModelRegistry,
    selection: ModelSelectionPolicy,
    definition: AgentDefinition | None = None,
) -> tuple[LLMRetrievalHintProvider, LLMToolDecisionProvider]:
    """Create retrieval-hint and tool-decision providers for an agent loop."""

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
        LLMToolDecisionProvider(
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

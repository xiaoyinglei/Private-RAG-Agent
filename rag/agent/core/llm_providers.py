from __future__ import annotations

import re
from collections.abc import Awaitable
from typing import Any

from pydantic import BaseModel

from rag.agent.core.definition import AgentDefinition, ModelSelectionPolicy
from rag.agent.core.llm_prompts import (
    build_retrieval_hint_prompt,
    build_tool_decision_prompt,
)
from rag.agent.core.llm_registry import ModelRegistry
from rag.agent.graphs.nodes.llm_decide import ToolDecisionProvider
from rag.agent.graphs.nodes.retrieval_hint import RetrievalHintProvider
from rag.agent.memory.models import InjectedContext
from rag.agent.state import AgentState, ThinkOutput
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


def _generate_structured[T: BaseModel](
    generator: Any,
    prompt: str,
    schema: type[T],
    *,
    kwargs: dict[str, Any] | None = None,
) -> T | None:
    """调用 Generator.generate_structured，解析失败返回 None。"""
    extra = kwargs or {}
    try:
        result = generator.generate_structured(prompt=prompt, schema=schema, **extra)
        if isinstance(result, schema):
            return result
        return schema.model_validate(result)
    except Exception:
        return None


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
    ) -> None:
        self._generator = generator
        self._kwargs = kwargs or {}

    def hint(self, state: AgentState) -> dict[Any, Any]:
        task = state.get("task", "")
        prompt = build_retrieval_hint_prompt(state)
        decision = _generate_structured(
            self._generator, prompt, RetrievalHintDecision, kwargs=self._kwargs
        )

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


class LLMToolDecisionProvider(ToolDecisionProvider):
    """Model tool-choice decision provider. Invalid output pauses visibly."""

    def __init__(self, generator: Any, *, kwargs: dict[str, Any] | None = None) -> None:
        self._generator = generator
        self._kwargs = kwargs or {}

    def decide(
        self,
        state: AgentState,
        *,
        definition: AgentDefinition,
        budget_remaining: int,
        context: InjectedContext,
    ) -> ThinkOutput | dict[str, object] | Awaitable[ThinkOutput | dict[str, object]]:
        prompt = build_tool_decision_prompt(
            state,
            budget_remaining=budget_remaining,
            context_text=context.as_text(),
            allowed_tools=definition.allowed_tools,
        )
        decision = _generate_structured(
            self._generator, prompt, ThinkOutput, kwargs=self._kwargs
        )
        if decision is None:
            return ThinkOutput(
                action="pause",
                thought="LLM tool-decision response could not be parsed",
                needs_user_input="Tool-decision provider failed to produce a valid decision",
                confidence=0.0,
            )
        return decision


# ── 从 ModelRegistry 批量创建 ──


def create_default_providers(
    registry: ModelRegistry,
    selection: ModelSelectionPolicy,
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
    return (
        LLMRetrievalHintProvider(hint_gen, kwargs=hint_kwargs),
        LLMToolDecisionProvider(decision_gen, kwargs=decision_kwargs),
    )

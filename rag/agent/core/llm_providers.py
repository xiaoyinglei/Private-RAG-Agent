from __future__ import annotations

import re
from collections.abc import Awaitable
from typing import Any

from pydantic import BaseModel

from rag.agent.core.definition import AgentDefinition, ModelSelectionPolicy
from rag.agent.core.llm_prompts import (
    build_evaluate_prompt,
    build_plan_prompt,
    build_route_prompt,
)
from rag.agent.core.llm_registry import ModelRegistry
from rag.agent.core.task import TaskDAG
from rag.agent.graphs.nodes.evaluate import EvaluateDecisionProvider
from rag.agent.graphs.nodes.plan import PlanProvider
from rag.agent.graphs.nodes.route import RouteProvider
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


# ── Router ──


class RouteDecision(BaseModel):
    route: str  # "fast_path" | "decompose" | "direct"
    reason: str
    retrieval_signals: dict[str, object] | None = None


class LLMRouteProvider(RouteProvider):
    """LLM 驱动的路由决策。失败时退回 direct。

    decompose_enabled: 子 Agent 编排是否可用。False 时 decompose 降级为 direct。
    """

    def __init__(
        self,
        generator: Any,
        *,
        kwargs: dict[str, Any] | None = None,
        decompose_enabled: bool = False,
    ) -> None:
        self._generator = generator
        self._kwargs = kwargs or {}
        self._decompose_enabled = decompose_enabled

    def route(self, state: AgentState) -> dict[Any, Any]:
        task = state.get("task", "")
        prompt = build_route_prompt(state)
        decision = _generate_structured(
            self._generator, prompt, RouteDecision, kwargs=self._kwargs
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

        route = decision.route if decision is not None else "direct"
        reason = decision.reason if decision is not None else "agent_research"
        if decision is None or route not in {"fast_path", "decompose", "direct"}:
            route = "direct"
            reason = "agent_research"

        update: dict[str, Any] = {
            "status": route,
            "route_reason": reason,
            "retrieval_signals": signals,
            "retrieval_signals_debug": signals_debug,
        }

        # decompose 降级：子 Agent 编排未启用时 → direct 循环
        if route == "decompose" and not self._decompose_enabled:
            update["status"] = "direct"
            update["route_reason"] = f"decompose_disabled: {reason}"
            update["decompose_disabled_single_agent_mode"] = True

        return update


# ── Evaluator ──


class LLMEvaluateDecisionProvider(EvaluateDecisionProvider):
    """LLM 驱动的评估决策。失败时返回 pause。"""

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
        prompt = build_evaluate_prompt(
            state,
            budget_remaining=budget_remaining,
            context_text=context.as_text(),
        )
        decision = _generate_structured(
            self._generator, prompt, ThinkOutput, kwargs=self._kwargs
        )
        if decision is None:
            return ThinkOutput(
                action="pause",
                thought="LLM evaluate response could not be parsed",
                needs_user_input="Evaluate provider failed to produce valid decision",
                confidence=0.0,
            )
        return decision


# ── Planner ──


class LLMPlanProvider(PlanProvider):
    """LLM 驱动的任务拆解。失败时由 plan_node 返回 failed。"""

    def __init__(self, generator: Any, *, kwargs: dict[str, Any] | None = None) -> None:
        self._generator = generator
        self._kwargs = kwargs or {}

    def create_plan(
        self,
        state: AgentState,
        *,
        definition: AgentDefinition,
    ) -> TaskDAG | dict[str, object] | Awaitable[TaskDAG | dict[str, object]]:
        prompt = build_plan_prompt(
            state,
            allowed_tools=definition.allowed_tools,
            max_depth=definition.max_depth,
        )
        plan = _generate_structured(
            self._generator, prompt, TaskDAG, kwargs=self._kwargs
        )
        if plan is None:
            raise ValueError("Plan provider failed to produce a valid TaskDAG")
        return plan


# ── 从 ModelRegistry 批量创建 ──


def create_default_providers(
    registry: ModelRegistry,
    selection: ModelSelectionPolicy,
) -> tuple[LLMRouteProvider, LLMEvaluateDecisionProvider, LLMPlanProvider]:
    """根据 ModelSelectionPolicy + ModelRegistry 创建三个 LLM provider。"""

    def _resolve(node_model: str | None, node_name: str, temperature: float) -> tuple[Any, dict[str, Any]]:
        resolved = registry.resolve_for_node(node_model=node_model, node_name=node_name)
        kwargs = dict(resolved.kwargs)
        kwargs.setdefault("temperature", temperature)
        return resolved.generator, kwargs

    router_gen, router_kwargs = _resolve(
        selection.route_model, "route", selection.route_temperature
    )
    evaluator_gen, evaluator_kwargs = _resolve(
        selection.evaluate_model, "evaluate", selection.evaluate_temperature
    )
    planner_gen, planner_kwargs = _resolve(
        selection.plan_model, "plan", selection.plan_temperature
    )

    return (
        LLMRouteProvider(router_gen, kwargs=router_kwargs),
        LLMEvaluateDecisionProvider(evaluator_gen, kwargs=evaluator_kwargs),
        LLMPlanProvider(planner_gen, kwargs=planner_kwargs),
    )

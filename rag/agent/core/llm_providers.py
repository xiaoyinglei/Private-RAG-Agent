from __future__ import annotations

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


class LLMRouteProvider(RouteProvider):
    """LLM 驱动的路由决策。失败时退回 direct。"""

    def __init__(self, generator: Any, *, kwargs: dict[str, Any] | None = None) -> None:
        self._generator = generator
        self._kwargs = kwargs or {}

    def route(self, state: AgentState) -> dict[Any, Any]:
        prompt = build_route_prompt(state)
        decision = _generate_structured(
            self._generator, prompt, RouteDecision, kwargs=self._kwargs
        )
        if decision is None or decision.route not in {"fast_path", "decompose", "direct"}:
            return {"status": "direct", "route_reason": "agent_research"}
        return {"status": decision.route, "route_reason": decision.reason}


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

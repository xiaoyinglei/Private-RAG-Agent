from __future__ import annotations

from collections.abc import Awaitable
from inspect import isawaitable
from typing import Protocol

from pydantic import ValidationError

from rag.agent.core.definition import AgentDefinition
from rag.agent.core.task import TaskDAG
from rag.agent.state import AgentState


class PlanProvider(Protocol):
    def create_plan(
        self,
        state: AgentState,
        *,
        definition: AgentDefinition,
    ) -> TaskDAG | dict[str, object] | Awaitable[TaskDAG | dict[str, object]]: ...


async def plan_node(
    state: AgentState,
    *,
    definition: AgentDefinition,
    plan_provider: PlanProvider | None = None,
) -> dict:
    if plan_provider is None:
        return {"status": "failed", "stop_reason": "plan_provider_missing"}

    try:
        raw_plan = plan_provider.create_plan(state, definition=definition)
        if isawaitable(raw_plan):
            raw_plan = await raw_plan
        plan = TaskDAG.model_validate(raw_plan)
    except ValidationError as exc:
        return {
            "status": "failed",
            "stop_reason": "invalid_task_dag",
            "needs_user_input": str(exc),
        }
    except Exception as exc:
        return {
            "status": "failed",
            "stop_reason": "plan_provider_failed",
            "needs_user_input": str(exc),
        }

    return {"status": "running", "plan": plan}


def route_after_plan(state: AgentState) -> str:
    if state.get("status") == "failed":
        return "synthesize"
    return "evaluate"

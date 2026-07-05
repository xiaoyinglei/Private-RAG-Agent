from __future__ import annotations

import pytest

from rag.agent.builtin.generic import GENERIC_AGENT
from rag.agent.builtin_registry import create_builtin_tool_registry
from rag.agent.core.context import AgentRunConfig
from rag.agent.loop.state import create_loop_state
from rag.agent.planning import AgentPlan
from rag.agent.planning import PlanStep as PlanningStep
from rag.agent.tools.catalog_assembly import build_tool_catalog
from rag.agent.tools.registry import ToolExecutionContext
from rag.agent.tools.runtime_registry_builder import RuntimeToolRegistryBuilder
from rag.schema.runtime import AccessPolicy


def _run_config() -> AgentRunConfig:
    return AgentRunConfig(
        run_id="runtime-tool-registry-builder",
        thread_id="runtime-tool-registry-builder",
        max_depth=1,
        access_policy=AccessPolicy.default(),
    )


@pytest.mark.anyio
async def test_runtime_tool_registry_builder_registers_update_plan_runner() -> None:
    base_registry = create_builtin_tool_registry()
    run_config = _run_config()
    state = create_loop_state(task="Explain policy", run_config=run_config)
    state["plan_state"].agent_plan = AgentPlan(
        objective="Explain policy",
        steps=[
            PlanningStep(
                step_id="step_existing",
                title="Inspect existing context",
                status="in_progress",
            )
        ],
    )

    runtime_registry = RuntimeToolRegistryBuilder(
        base_tool_registry=base_registry,
        policy=GENERIC_AGENT,
        catalog=build_tool_catalog(base_registry, GENERIC_AGENT),
    ).build(run_config)

    output = await runtime_registry.run(
        "update_plan",
        {
            "action": "add",
            "steps": [
                {
                    "id": "step_new",
                    "description": "Summarize result",
                    "status": "pending",
                }
            ],
            "summary": "Plan updated",
        },
        execution_context=ToolExecutionContext(
            run_config=run_config,
            state=state,
        ),
    )

    assert [step.id for step in output.steps] == ["step_existing", "step_new"]
    assert output.steps[0].description == "Inspect existing context"
    assert output.steps[0].status == "in_progress"
    assert output.summary == "Plan updated"


def test_runtime_tool_registry_builder_clones_base_registry_for_extra_runners() -> None:
    base_registry = create_builtin_tool_registry()

    runtime_registry = RuntimeToolRegistryBuilder(
        base_tool_registry=base_registry,
        policy=GENERIC_AGENT,
        catalog=build_tool_catalog(base_registry, GENERIC_AGENT),
    ).build(
        _run_config(),
        runners={"llm_generate": lambda payload: {"text": str(payload)}},
    )

    assert runtime_registry.has_runner("llm_generate")
    assert not base_registry.has_runner("llm_generate")

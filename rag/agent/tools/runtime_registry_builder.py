"""Per-run ToolRegistry assembly."""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal, cast

from pydantic import BaseModel

from rag.agent.capabilities.catalog import ToolCatalog, ToolCatalogEntry
from rag.agent.core.context import AgentRunConfig
from rag.agent.core.definition import AgentRuntimePolicy
from rag.agent.core.delegation import DelegatedAgentRunner
from rag.agent.core.llm_registry import ModelResolver
from rag.agent.core.runtime_ports import RetrievalHintProvider
from rag.agent.loop.runtime import ModelTurnProvider
from rag.agent.tools.mcp_adapter import MCPToolRegistry
from rag.agent.tools.registry import ToolRegistry, ToolRunner

logger = logging.getLogger(__name__)


@dataclass
class RuntimeToolRegistryBuilder:
    """Clone the base registry and inject request-scoped runners."""

    base_tool_registry: ToolRegistry
    policy: AgentRuntimePolicy
    catalog: ToolCatalog
    model_registry: ModelResolver | None = None
    model_turn_provider: ModelTurnProvider | None = None
    retrieval_hint_provider: RetrievalHintProvider | None = None
    task_delegated_runner: DelegatedAgentRunner | None = None
    task_delegated_runner_factory: Callable[[ToolRegistry], DelegatedAgentRunner] | None = None
    subagent_runner: DelegatedAgentRunner | None = None
    mcp_registry: MCPToolRegistry | None = None

    def build(
        self,
        run_config: AgentRunConfig,
        *,
        runners: Mapping[str, ToolRunner] | None = None,
        tools: list[Any] | None = None,
    ) -> ToolRegistry:
        runtime = self.base_tool_registry.clone()
        self._inject_model_llm_tool_runners(runtime)
        self._register_workspace_tools(runtime, tools)
        self._register_extra_runners(runtime, runners)
        self._register_update_plan_runner(runtime)
        self._assemble_tool_pool(runtime)
        self._register_task_runner(runtime, run_config)
        self._register_agent_as_tool_adapters(runtime, run_config)
        return runtime

    def _inject_model_llm_tool_runners(self, registry: ToolRegistry) -> None:
        if self.model_registry is None:
            return
        from rag.agent.core.llm_tool_runners import create_model_llm_tool_runners

        for tool_name, runner in create_model_llm_tool_runners(
            self.model_registry,
        ).items():
            if registry.has_runner(tool_name):
                continue
            try:
                registry.register_contextual_runner(tool_name, runner)
            except KeyError:
                pass

    @staticmethod
    def _register_workspace_tools(
        runtime: ToolRegistry,
        tools: list[Any] | None,
    ) -> None:
        if not tools:
            return
        for tool in tools:
            try:
                runtime.register_tool(tool)
            except Exception:
                logger.warning(
                    "Failed to register tool '%s'",
                    getattr(tool, "name", "?"),
                    exc_info=True,
                )

    @staticmethod
    def _register_extra_runners(
        runtime: ToolRegistry,
        runners: Mapping[str, ToolRunner] | None,
    ) -> None:
        if not runners:
            return
        for extra_name, extra_runner in runners.items():
            try:
                runtime.register_runner(extra_name, extra_runner)
            except KeyError:
                pass

    @staticmethod
    def _register_update_plan_runner(runtime: ToolRegistry) -> None:
        from rag.agent.tools.generic_tools import (
            PlanStep,
            UpdatePlanInput,
            UpdatePlanOutput,
            update_plan_spec,
        )

        def _to_tool_step(step: Any, *, fallback_index: int) -> PlanStep:
            sid = getattr(step, "id", None) or getattr(step, "step_id", None)
            description = (
                getattr(step, "description", None)
                or getattr(step, "title", None)
                or ""
            )
            status_value = getattr(step, "status", "pending")
            if status_value in {"pending", "in_progress", "completed", "blocked"}:
                status = cast(
                    Literal["pending", "in_progress", "completed", "blocked"],
                    status_value,
                )
            else:
                status = "pending"
            return PlanStep(
                id=str(sid or f"step-{fallback_index}"),
                description=str(description),
                status=status,
            )

        def _update_plan_runner(payload: Any, context: Any) -> Any:
            if isinstance(payload, dict):
                inp = UpdatePlanInput(**payload)
            else:
                inp = payload
            state = getattr(context, "state", {}) or {}
            plan_state = state.get("plan_state")
            existing: list[PlanStep] = []
            if plan_state and hasattr(plan_state, "agent_plan") and plan_state.agent_plan:
                existing = [
                    _to_tool_step(step, fallback_index=index + 1)
                    for index, step in enumerate(plan_state.agent_plan.steps)
                ]
            steps = list(existing)
            if inp.action == "add":
                for s in inp.steps:
                    sid = s.id or f"step-{len(steps) + 1}"
                    steps.append(
                        PlanStep(
                            id=sid,
                            description=s.description,
                            status=s.status,
                        )
                    )
            elif inp.action == "complete":
                ids = set(inp.step_ids)
                steps = [
                    PlanStep(
                        id=s.id,
                        description=s.description,
                        status="completed" if s.id in ids else s.status,
                    )
                    for s in steps
                ]
            elif inp.action == "update":
                by_id = {s.id: s for s in inp.steps if s.id}
                steps = [by_id.get(old.id, old) for old in existing]
            return UpdatePlanOutput(
                steps=steps,
                summary=inp.summary,
                message="plan updated",
            )

        try:
            runtime.get("update_plan")
        except KeyError:
            runtime.register(update_plan_spec)
        runtime.register_contextual_runner("update_plan", _update_plan_runner)

    def _assemble_tool_pool(self, runtime: ToolRegistry) -> None:
        self._inject_mcp_tools(runtime)

        deny = self.policy.tool_catalog_filter.deny
        if deny:
            for tool_name in sorted(deny):
                if tool_name in self.policy.allowed_tools:
                    self.policy.allowed_tools.remove(tool_name)
                if self.catalog.get(tool_name) is not None:
                    logger.info("Tool '%s' removed by deny rule", tool_name)

    def _inject_mcp_tools(self, runtime: ToolRegistry) -> None:
        if self.mcp_registry is None:
            return

        mcp_names = [s.name for s in self.mcp_registry.list_all_tools()]
        for name in mcp_names:
            if name not in self.policy.allowed_tools:
                self.policy.allowed_tools.append(name)

        for spec in self.mcp_registry.list_all_tools():
            try:
                runtime.get(spec.name)
            except KeyError:
                runtime.register(spec)

            try:
                runner = self.mcp_registry.get_runner(spec.name)
                runtime.register_contextual_runner(spec.name, runner)
            except KeyError:
                logger.warning("MCP tool '%s' has no runner — call skipped", spec.name)

            if runtime.get_formatter(spec.name) is None:
                from rag.agent.tools.formatters.mcp_tools import MCPToolFormatter

                runtime.register_formatter(MCPToolFormatter(spec.name))

            if self.catalog.get(spec.name) is None:
                card = spec.aci
                search_text = ToolCatalog.build_search_text(
                    spec.name,
                    spec.description,
                    "",
                    when_to_use=card.when_to_use if card else "",
                    when_not_to_use=card.when_not_to_use if card else "",
                    domains=card.domains if card else (),
                    file_types=card.file_types if card else (),
                    selection_tags=card.selection_tags if card else (),
                )
                self.catalog.register(
                    ToolCatalogEntry(
                        name=spec.name,
                        description=spec.description,
                        category="deferred",
                        search_text=search_text,
                        activation_group=card.activation_group if card else "mcp",
                        when_to_use=card.when_to_use if card else "",
                        when_not_to_use=card.when_not_to_use if card else "",
                        domains=card.domains if card else (),
                        selection_tags=card.selection_tags if card else (),
                        source="mcp",
                    ),
                )

    def _register_task_runner(
        self,
        runtime: ToolRegistry,
        run_config: AgentRunConfig,
    ) -> None:
        if not (
            runtime.has_runner("task")
            or "task" in {s.name for s in self.base_tool_registry.list_all()}
        ):
            return
        from rag.agent.tools.task_tool import TaskInput, TaskOutput, TaskToolRunner

        task_runner = TaskToolRunner(
            policy=self.policy,
            tool_registry=runtime,
            model_turn_provider=self.model_turn_provider,
            retrieval_hint_provider=self.retrieval_hint_provider,
            delegated_runner=(
                self.task_delegated_runner_factory(runtime)
                if self.task_delegated_runner_factory is not None
                else self.task_delegated_runner
            ),
        )

        async def _task_runner(payload: BaseModel) -> TaskOutput:
            input_data = TaskInput.model_validate(payload)
            return await task_runner.run(input_data, parent_config=run_config)

        runtime.register_runner("task", _task_runner)

    def _register_agent_as_tool_adapters(
        self,
        runtime: ToolRegistry,
        run_config: AgentRunConfig,
    ) -> None:
        if self.subagent_runner is None:
            return
        from rag.agent.core.agent_as_tool import AgentAsToolAdapter

        for spec in self.base_tool_registry.list_all():
            if not spec.name.startswith("agent_"):
                continue
            agent_type = spec.name[len("agent_") :]
            adapter = AgentAsToolAdapter(
                runner=self.subagent_runner,
                agent_type=agent_type,
                run_config=run_config,
            )
            runtime.register_runner(spec.name, adapter)


__all__ = ["RuntimeToolRegistryBuilder"]

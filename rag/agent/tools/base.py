"""BaseTool — self-contained tool: spec + execute in one class.

Every builtin tool inherits from BaseTool.  The class carries its own
metadata (name, description, permissions, ToolCard) and execution logic.
Registration is a single call: registry.register_tool(tool_instance).

MCP and dynamically-generated tools still use raw ToolSpec + runner
registration — they are the exception, not the rule.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel

from rag.agent.tools.card import ToolCard
from rag.agent.tools.spec import (
    ExecutionCategory,
    InterruptBehavior,
    RiskLevel,
    ToolError,
    ToolPermissions,
    ToolSpec,
)


class BaseTool(ABC):
    """A tool is a self-contained unit: metadata + execution logic.

    Subclasses define:
      - ToolSpec fields: name, description, permissions, etc.
      - execute(): the actual tool logic

    Registration: registry.register_tool(tool_instance)
      → calls to_spec() + registers a runner closure over execute()
    """

    # ── Tool identity (override in subclasses) ──
    name: str = ""
    description: str = ""

    # ── Contract models (override in subclasses) ──
    input_model: type[BaseModel] = BaseModel
    output_model: type[BaseModel] = BaseModel
    error_model: type[BaseModel] = ToolError

    # ── Behaviour (override in subclasses) ──
    permissions: ToolPermissions = ToolPermissions()
    execution_category: ExecutionCategory = ExecutionCategory.READ
    risk_level: RiskLevel | None = None  # None → ToolSpec infers from category
    interrupt_behavior: InterruptBehavior = InterruptBehavior.CANCEL
    timeout_seconds: float = 30.0
    max_retries: int = 0
    idempotent: bool = False
    concurrency_safe: bool = False
    work_budget_cost: int = 100
    max_result_size_chars: int = 64000

    # ── ACI metadata ──
    aci: ToolCard | None = None

    # ── Abstract ──

    @abstractmethod
    async def execute(
        self,
        input_data: BaseModel,
        context: Any | None = None,
    ) -> BaseModel:
        """Execute the tool.  Called by ToolRegistry via the runner closure."""
        ...

    # ── Spec generation ──

    def to_spec(self) -> ToolSpec:
        """Generate a ToolSpec from this tool's metadata.

        Called once during registration.  The resulting ToolSpec is stored
        in ToolRegistry and used for input/output validation, approval,
        and catalog indexing.
        """
        kwargs: dict[str, Any] = dict(
            name=self.name,
            description=self.description,
            input_model=self.input_model,
            output_model=self.output_model,
            error_model=self.error_model,
            permissions=self.permissions,
            execution_category=self.execution_category,
            interrupt_behavior=self.interrupt_behavior,
            timeout_seconds=self.timeout_seconds,
            max_retries=self.max_retries,
            idempotent=self.idempotent,
            concurrency_safe=self.concurrency_safe,
            work_budget_cost=self.work_budget_cost,
            max_result_size_chars=self.max_result_size_chars,
            aci=self.aci,
        )
        if self.risk_level is not None:
            kwargs["risk_level"] = self.risk_level
        return ToolSpec(**kwargs)

    # ── Runner adapter ──

    def as_runner(self):
        """Return a callable suitable for ToolRegistry.register_runner().

        The runner is a closure over self.execute().
        """
        tool = self

        async def _runner(input_data: BaseModel, context: Any | None = None) -> BaseModel:
            return await tool.execute(input_data, context)

        return _runner
